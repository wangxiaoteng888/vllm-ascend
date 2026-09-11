# SPDX-License-Identifier: Apache-2.0
"""Host staging must remain pinned until every PP H2D pull completes."""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import KVCacheRecvingThread as Base
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_d2rh_connector import KVCacheRecvingThread as Receiver


def handle_request(self, meta):
    try:
        if meta.get("fail"):
            raise RuntimeError("injected transfer failure")
    finally:
        if self._mark_request_task_done(meta["request_id"], meta["all_task_done"]):
            self.completed.append(meta["request_id"])


def receiver():
    obj = Receiver.__new__(Receiver)
    obj.request_task_counts = defaultdict(int)
    obj.request_task_counts_lock = threading.Lock()
    obj.finished_request_markers = set()
    obj._h2d_remote_request_ids = {}
    obj.freed = []
    obj.completed = []
    obj.block_map = {(0, 1, 0): 7}
    obj.remote_local_block_map = {"D-request": obj.block_map, "P-request": obj.block_map}
    obj.cpu_kvcache_manager = SimpleNamespace(free_block_map=obj.freed.append)
    return obj


def shards():
    return [dict(request_id="D-request", remote_request_id="P-request", all_task_done=last) for last in [False, True]]


class TestH2DLifetime(unittest.TestCase):
    def setUp(self):
        fake = patch.object(Base, "_handle_request", handle_request)
        fake.start()
        self.addCleanup(fake.stop)

    def test_last_submitted_finishes_first_must_keep_host_pinned(self):
        obj = receiver()
        first, last = shards()
        for item in [first, last]:
            obj._mark_request_task_submitted(item)
        obj._handle_request(last)
        self.assertEqual(obj.request_task_counts["D-request"], 1)
        self.assertEqual(obj.freed, [], "Host freed while another PP pull is pending")
        self.assertIn("P-request", obj.remote_local_block_map)
        obj._handle_request(first)
        self.assertEqual(obj.freed, [obj.block_map])
        self.assertEqual(obj.remote_local_block_map, {})
        self.assertEqual(obj.completed, ["D-request"])

    def test_in_order_releases_once(self):
        obj = receiver()
        items = shards()
        for item in items:
            obj._mark_request_task_submitted(item)
        obj._handle_request(items[0])
        self.assertEqual(obj.freed, [])
        obj._handle_request(items[1])
        self.assertEqual(obj.freed, [obj.block_map])

    def test_pp1_release(self):
        obj = receiver()
        item = shards()[-1]
        obj._mark_request_task_submitted(item)
        obj._handle_request(item)
        self.assertEqual(obj.freed, [obj.block_map])

    def test_early_first_completion_before_last_submission(self):
        obj = receiver()
        first, last = shards()
        obj._mark_request_task_submitted(first)
        obj._handle_request(first)
        self.assertEqual(obj.freed, [])
        obj._mark_request_task_submitted(last)
        obj._handle_request(last)
        self.assertEqual(obj.freed, [obj.block_map])

    def test_failed_last_pull_keeps_other_pull_pinned(self):
        obj = receiver()
        first, last = shards()
        last["fail"] = True
        for item in [first, last]:
            obj._mark_request_task_submitted(item)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            obj._handle_request(last)
        self.assertEqual(obj.freed, [])
        obj._handle_request(first)
        self.assertEqual(obj.freed, [obj.block_map])
