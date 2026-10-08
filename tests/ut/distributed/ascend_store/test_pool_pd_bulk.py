# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import queue
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401
from tests.ut.distributed.ascend_store.test_pool_pd import make_request, make_scheduler
from tests.ut.distributed.ascend_store.test_pool_pd_memcache import make_pd_request
from tests.ut.distributed.ascend_store.test_pool_worker import make_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import BatchResultShapeError
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import KVCacheStoreRecvingThread
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata


@pytest.fixture(params=["mooncake", "memcache"])
def worker(request):
    patches = unittest.TestCase()
    result = make_worker(patches, kv_role="kv_consumer", extra_config={"backend": request.param, "pool_pd": True})
    result.token_database.set_group_buffers({0: [1000, 2000, 3000, 4000]}, {0: [32, 64, 48, 16]})
    result.m_store = MagicMock()
    result.m_store.batch_get_start.side_effect = lambda keys: [0] * len(keys)
    result.m_store.batch_copy_get.side_effect = lambda keys, *_: [0] * len(keys)
    result.m_store.batch_get_end.return_value = 0
    result.m_store.batch_add_lease.side_effect = lambda keys, _: [0] * len(keys)
    result.m_store.batch_get_key_info.side_effect = lambda keys: [
        SimpleNamespace(gva_list=lambda: [10000], size=lambda: 160) for _ in keys
    ]
    result.m_store.batch_remove_lease.return_value = 0
    result.m_store.store.batch_copy.return_value = 0
    result.kv_recv_thread = MagicMock()
    try:
        yield result
    finally:
        patches.doCleanups()


@pytest.mark.parametrize("length", [1, 15, 16, 17, 31, 32, 33, 300])
def test_bulk_restore_uses_p_keys_and_all_layer_ranges(worker, length):
    request = make_pd_request(length, decode=True)
    worker._load_pd_snapshot(request)
    expected = [worker._make_layerwise_full_key(0, value) for value in request.pd_transfer["block_hashes"]]
    if length % 16 > 1:
        expected.append(worker._make_layerwise_full_key(0, request.pd_transfer["tail_id"]))
    if not expected:
        worker.m_store.batch_get_start.assert_not_called()
        worker.m_store.batch_add_lease.assert_not_called()
        worker.m_store.store.batch_copy.assert_not_called()
        return
    addresses = [
        [1000 + block * 32, 2000 + block * 64, 3000 + block * 48, 4000 + block * 16]
        for block in request.block_ids_by_group[0][: len(expected)]
    ]
    if worker.backend_name == "mooncake":
        worker.m_store.batch_copy_get.assert_called_once_with(
            expected, addresses, [[32, 64, 48, 16]] * len(expected), [[0, 32, 96, 144]] * len(expected)
        )
        worker.m_store.batch_get_end.assert_called_once_with(expected)
    else:
        worker.m_store.store.batch_copy.assert_called_once_with(
            [10000, 10032, 10096, 10144] * len(expected),
            [addr for row in addresses for addr in row],
            [32, 64, 48, 16] * len(expected),
            1,
        )
        assert worker.m_store.batch_add_lease.call_args.args[0] == expected
        worker.m_store.batch_remove_lease.assert_called_once_with(expected)


@pytest.mark.parametrize("length", [17, 33])
def test_bulk_restore_does_not_require_a_page_for_the_replayed_tail_token(worker, length):
    request = make_pd_request(length, decode=True)
    request.block_ids_by_group = [request.block_ids_by_group[0][:-1]]
    worker._load_pd_snapshot(request)
    acquire = worker.m_store.batch_get_start if worker.backend_name == "mooncake" else worker.m_store.batch_add_lease
    assert acquire.call_args.args[0] == [
        worker._make_layerwise_full_key(0, value) for value in request.pd_transfer["block_hashes"]
    ]


@pytest.mark.parametrize("length", [1, 16, 17])
def test_bulk_scheduler_waits_for_full_snapshot_including_tail(length):
    producer = make_scheduler()
    request = make_request(length)
    producer.get_num_new_matched_tokens(request, 0)
    _, request.kv_transfer_params = producer.request_finished(request, [])
    consumer = make_scheduler("kv_consumer", use_layerwise=False)
    assert consumer.get_num_new_matched_tokens(request, 0) == (length - 1, length > 1)
    blocks = [list(range(10, 10 + (length + 15) // 16))]
    if length > 1:
        meta = consumer._process_async_load_request(request.request_id, request, blocks)
        assert meta.pd_transfer == {
            key: value for key, value in request.kv_transfer_params["pool_pd"].items() if key != "status"
        }
        assert meta.block_ids_by_group == blocks
        assert meta.target_token_len == length
        assert not meta.can_save
    else:
        consumer.update_state_after_alloc(request, SimpleNamespace(), 0)
        assert consumer.load_specs[request.request_id].can_load


def test_bulk_scheduler_supports_synchronous_restore():
    producer = make_scheduler()
    request = make_request()
    producer.get_num_new_matched_tokens(request, 0)
    _, request.kv_transfer_params = producer.request_finished(request, [])
    consumer = make_scheduler("kv_consumer", use_layerwise=False, extra_config={"load_async": False})
    assert consumer.get_num_new_matched_tokens(request, 0) == (19, False)


def test_producer_still_requires_layerwise_saves():
    with pytest.raises(ValueError, match="layerwise P writes"):
        make_scheduler(use_layerwise=False)


@pytest.mark.parametrize("length", [1, 17])
def test_only_requests_with_external_tokens_are_queued(worker, length):
    request = make_pd_request(length, decode=True)
    meta = AscendConnectorMetadata(preempted_req_ids=set())
    meta.requests.append(request)
    worker.start_load_kv(meta)
    if length == 1:
        worker.kv_recv_thread.add_request.assert_not_called()
        worker.m_store.batch_get_start.assert_not_called()
        worker.m_store.batch_add_lease.assert_not_called()
    else:
        worker.kv_recv_thread.add_request.assert_called_once_with(request)
        worker.m_store.batch_get_start.assert_not_called()
        worker.m_store.batch_add_lease.assert_not_called()


def test_ongoing_decode_does_not_read_pool(worker):
    worker.start_load_kv(AscendConnectorMetadata(preempted_req_ids=set()))
    worker.m_store.batch_get_start.assert_not_called()
    worker.m_store.batch_add_lease.assert_not_called()
    worker.kv_recv_thread.add_request.assert_not_called()


@pytest.mark.parametrize("failure", ["lease", "shape", "copy", "release"])
def test_bulk_failure_releases_acquired_objects(worker, failure):
    request = make_pd_request(32, decode=True)
    first_key = worker._make_layerwise_full_key(0, request.pd_transfer["block_hashes"][0])
    if worker.backend_name == "mooncake":
        if failure in ("lease", "shape"):
            worker.m_store.batch_get_start.side_effect = None
            worker.m_store.batch_get_start.return_value = [0, -1] if failure == "lease" else [0]
        elif failure == "copy":
            worker.m_store.batch_copy_get.side_effect = None
            worker.m_store.batch_copy_get.return_value = [-1, 0]
        else:
            worker.m_store.batch_get_end.return_value = -1
    else:
        if failure in ("lease", "shape"):
            worker.m_store.batch_add_lease.side_effect = None
            worker.m_store.batch_add_lease.return_value = [0, -1] if failure == "lease" else [0]
        elif failure == "copy":
            worker.m_store.store.batch_copy.return_value = -1
        else:
            worker.m_store.batch_remove_lease.return_value = -1
    with pytest.raises((RuntimeError, BatchResultShapeError)):
        worker._load_pd_snapshot(request)
    release = worker.m_store.batch_get_end if worker.backend_name == "mooncake" else worker.m_store.batch_remove_lease
    if failure in ("lease", "shape"):
        release.assert_called_once_with([first_key])
    else:
        assert len(release.call_args.args[0]) == 2


def test_split_ranges_acquire_each_object_only_once(worker):
    worker.layerwise_max_transfer_blocks = 1
    worker.layerwise_max_transfer_bytes = 16
    worker._load_pd_snapshot(make_pd_request(18, decode=True))
    acquire = worker.m_store.batch_get_start if worker.backend_name == "mooncake" else worker.m_store.batch_add_lease
    assert acquire.call_count == 2
    copy = worker.m_store.batch_copy_get if worker.backend_name == "mooncake" else worker.m_store.store.batch_copy
    assert copy.call_count == 2
    for call in copy.call_args_list:
        lengths = call.args[2]
        assert (
            max(value for row in lengths for value in row) <= 16
            if worker.backend_name == "mooncake"
            else max(lengths) <= 16
        )


def test_bulk_receive_marks_finished_only_after_restore_succeeds(worker):
    receiver = object.__new__(KVCacheStoreRecvingThread)
    receiver.worker = worker
    receiver.request_queue = queue.Queue()
    receiver.set_finished_request = MagicMock()
    request = make_pd_request(decode=True)
    receiver.request_queue.put(request)
    receiver._handle_request(receiver.request_queue.get())
    receiver.set_finished_request.assert_called_once_with(request.req_id)
    worker._load_pd_snapshot = MagicMock(side_effect=RuntimeError("copy failed"))
    receiver.set_finished_request.reset_mock()
    receiver.request_queue.put(request)
    with pytest.raises(RuntimeError, match="copy failed"):
        receiver._handle_request(receiver.request_queue.get())
    receiver.set_finished_request.assert_not_called()
    assert receiver.request_queue.unfinished_tasks == 0


def test_bulk_receive_failure_is_propagated_to_engine(worker):
    worker.kv_recv_thread.raise_if_failed.side_effect = RuntimeError("restore failed")
    with pytest.raises(RuntimeError, match="restore failed"):
        worker.get_finished(set(), AscendConnectorMetadata(preempted_req_ids=set()))


@pytest.mark.parametrize("length", [16, 18])
def test_layerwise_written_bytes_round_trip_to_different_bulk_destination_pages(worker, length):
    request = make_pd_request(length, decode=True)
    sizes = [32, 64, 48, 16]
    source = [np.arange(32 * size, dtype=np.uint8).reshape(32, size) for size in sizes]
    destination = [np.zeros_like(buffer) for buffer in source]
    source_blocks = [2, 5]
    worker.token_database.set_group_buffers(
        {0: [buffer.ctypes.data for buffer in source]}, {0: sizes}, group_num_layers={0: 2}
    )
    ids = request.pd_transfer["block_hashes"] + ([request.pd_transfer["tail_id"]] if length % 16 else [])
    objects = {worker._make_layerwise_full_key(0, value): np.zeros(sum(sizes), dtype=np.uint8) for value in ids}
    # Reproduce P's layerwise range PUT layout using real source pointers;
    # D's block IDs and buffers intentionally differ from P's.
    for index, key in enumerate(objects):
        for layer in range(2):
            addresses, lengths, _ = worker.token_database.prepare_value_layer(
                index * 16, (index + 1) * 16, source_blocks, layer
            )
            offset = sum(sizes[: layer * 2])
            for address, size in zip(addresses, lengths, strict=True):
                ctypes.memmove(objects[key].ctypes.data + offset, address, size)
                offset += size
    worker.token_database.set_group_buffers({0: [buffer.ctypes.data for buffer in destination]}, {0: sizes})
    if worker.backend_name == "mooncake":

        def copy_ranges(keys, addresses, lengths, offsets):
            for key, addr_row, size_row, offset_row in zip(keys, addresses, lengths, offsets, strict=True):
                for address, size, offset in zip(addr_row, size_row, offset_row, strict=True):
                    ctypes.memmove(address, objects[key].ctypes.data + offset, size)
            return [0] * len(keys)

        worker.m_store.batch_copy_get.side_effect = copy_ranges
    else:
        worker.m_store.batch_get_key_info.side_effect = lambda keys: [
            SimpleNamespace(gva_list=lambda key=key: [objects[key].ctypes.data], size=lambda: sum(sizes))
            for key in keys
        ]

        def copy_gvas(gvas, addresses, lengths, direction):
            assert direction == 1
            for gva, address, size in zip(gvas, addresses, lengths, strict=True):
                ctypes.memmove(address, gva, size)
            return 0

        worker.m_store.store.batch_copy.side_effect = copy_gvas
    worker._load_pd_snapshot(request)
    for src, dst in zip(source, destination, strict=True):
        for index, block in enumerate(request.block_ids_by_group[0]):
            np.testing.assert_array_equal(dst[block], src[source_blocks[index]])
        np.testing.assert_array_equal(dst[0], 0)


def test_bulk_does_not_require_piecewise_or_eager_layerwise_load():
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pd_transfer import requires_layerwise_eager_load

    config = {"pool_pd": True, "use_layerwise": False}
    assert not AscendStoreConnector.requires_piecewise_for_cudagraph(config)
    assert not requires_layerwise_eager_load(
        SimpleNamespace(has_sync_kv_loads=True), SimpleNamespace(kv_connector_extra_config=config)
    )
