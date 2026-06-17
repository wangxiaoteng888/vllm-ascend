# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import logging
import math
import os
import queue
import random
import struct
import threading
import time
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import msgspec
import numpy as np
import numpy.typing as npt
import torch
import torch_npu
import zmq
from mooncake.engine import TransferEngine  # type: ignore
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_decode_context_model_parallel_rank,
    get_decode_context_model_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.distributed.utils import get_pp_indices
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.request import RequestStatus

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import (
    RegisterRegions,
    collect_storage_merged_register_regions,
    get_transfer_timeout_value,
    validate_register_region_count,
)
from vllm_ascend.utils import enable_custom_op

# isort: off
if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request
# isort: on

GET_META_MSG = b"get_meta_msg"
DONE_RECVING_MSG = b"done_recving_msg"
READY_SCHEDULER = b"ready_scheduler"
STAGING_FULL = b"staging_full"
START_PULL = b"START_PULL"

# ZMQ ports for D2RH (hop1) and scheduler ready signaling (hop1 done).
# port = BASE + dp_offset + (pp_rank + pcp_rank) * tp_size + tp_rank
D2RH_ZMQ_PORT_BASE = 8100
SCHEDULER_READY_ZMQ_PORT_BASE = 8200


class RemotePortInfo(TypedDict):
    num: int
    host: str


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]]
    block_size: int
    kv_caches_base_addr: list[list[int]]
    block_size_scale: list[list[int]]
    num_blocks: int
    block_lens: list[list[int]]
    block_strides: list[list[int]]
    local_ip: str = ""


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    num_external_tokens: int
    num_computed_tokens: int
    remote_block_ids: BlockIds
    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_pcp_size: int
    remote_dcp_size: int
    remote_ptp_size: int | None
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]
    num_prompt_blocks: int


@dataclass(frozen=True)
class GroupPull:
    group_id: int
    remote_tp_offset: int
    num_group_pulls: int
    prefill_pp_rank: int = 0
    is_group_transfer_end: bool = False


@dataclass(frozen=True)
class GroupTransferInfo:
    tokens_per_block: int
    blocks_per_window: int
    is_state_group: bool


def expand_kernel_block_ids(block_ids: list[int], scale: int) -> list[int]:
    return [bid * scale + offset for bid in block_ids for offset in range(scale)]


def trim_kernel_blocks_for_prefix_cache(
    local_block_ids: list[int],
    remote_block_ids: list[int],
    local_scale: int,
    remote_scale: int,
    block_size: int,
    group_compress_ratio: int,
    num_computed_tokens: int,
) -> tuple[list[int], list[int]]:
    """Pair local/remote kernel block ids for prefix-cache-aware KV transfer.

    Used by direct P->D, D2RH Hop1, and H2D Hop2. Skip remote-side kernel blocks
    by token offset, then pair by index up to min length.
    """
    kernel_local_block_ids = expand_kernel_block_ids(local_block_ids, local_scale)
    kernel_remote_block_ids = expand_kernel_block_ids(remote_block_ids, remote_scale)
    remote_kernel_block_size = block_size // remote_scale
    remote_kernel_token_size = remote_kernel_block_size * group_compress_ratio
    remote_start_idx = num_computed_tokens // remote_kernel_token_size
    kernel_remote_block_ids = kernel_remote_block_ids[remote_start_idx:]
    num_kernel_blocks = min(len(kernel_remote_block_ids), len(kernel_local_block_ids))
    return (
        kernel_local_block_ids[:num_kernel_blocks],
        kernel_remote_block_ids[:num_kernel_blocks],
    )


@dataclass
class StagingAllocation:
    """Host CPU staging blocks allocated for one request on one decode TP."""

    block_maps: list[dict[int, int]]
    allocated_blocks: list[list[int]]


class HostStagingManager:
    """Per-TP host block pool for D2RH CPU staging."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.inflight: dict[str, StagingAllocation] = {}

    @staticmethod
    def staging_demand(remote_block_ids: BlockIds) -> int:
        return sum(len(group_blocks) for group_blocks in remote_block_ids)

    def try_alloc(self, request_id: str, remote_block_ids: BlockIds) -> StagingAllocation | None:
        if request_id in self.inflight:
            return self.inflight[request_id]

        total_needed = self.staging_demand(remote_block_ids)
        if total_needed > len(self.free_blocks):
            logger.info(
                "Host staging full for request %s: need %d blocks, free %d",
                request_id,
                total_needed,
                len(self.free_blocks),
            )
            return None

        block_maps: list[dict[int, int]] = []
        allocated_blocks: list[list[int]] = []
        for group_remote_block_ids in remote_block_ids:
            cpu_block_ids = [self.free_blocks.popleft() for _ in range(len(group_remote_block_ids))]
            block_maps.append(dict(zip(group_remote_block_ids, cpu_block_ids)))
            allocated_blocks.append(cpu_block_ids)

        staging = StagingAllocation(block_maps=block_maps, allocated_blocks=allocated_blocks)
        self.inflight[request_id] = staging
        logger.info("Host staging allocated request %s blocks=%s", request_id, allocated_blocks)
        return staging

    def get(self, request_id: str) -> StagingAllocation | None:
        return self.inflight.get(request_id)

    def free(self, request_id: str) -> None:
        staging = self.inflight.pop(request_id, None)
        if staging is None:
            return
        for cpu_block_ids in staging.allocated_blocks:
            self.free_blocks.extend(cpu_block_ids)
        logger.info("Host staging freed request %s", request_id)


class HostListeningThread(threading.Thread):
    """Collect READY_SCHEDULER / STAGING_FULL signals from D2RH threads."""

    def __init__(
        self,
        all_requests: set[str],
        decode_tp_size: int,
        scheduler_ready_port: int,
    ):
        super().__init__(daemon=True, name="HostListeningThread")
        self.all_requests = all_requests
        self.decode_tp_size = max(decode_tp_size, 1)
        self.scheduler_ready_port = scheduler_ready_port
        self.ready_count: dict[str, int] = defaultdict(int)
        self.ready_request: set[str] = set()
        self.ready_lock = threading.Lock()
        self.host_ip = get_ip()

    def run(self) -> None:
        path = make_zmq_path("tcp", self.host_ip, self.scheduler_ready_port)
        logger.info("Starting scheduler ready listener on path: %s", path)
        try:
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self._run_busy_loop(sock)
        except Exception as e:
            logger.error("HostListeningThread exception: %s", e, exc_info=True)

    def _run_busy_loop(self, sock: zmq.Socket) -> None:  # type: ignore
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message format: %s", frames)
                    continue

                identity = frames[0]
                payload = [frame for frame in frames[1:] if frame != b""]
                if len(payload) != 1:
                    logger.error("Invalid message format: %s", frames)
                    continue

                msg = decoder.decode(payload[0])
                if msg[0] == READY_SCHEDULER:
                    request_id = msg[1]
                    with self.ready_lock:
                        self.ready_count[request_id] += 1
                        ready_cnt = self.ready_count[request_id]
                        if self.ready_count[request_id] >= self.decode_tp_size:
                            self.ready_request.add(request_id)
                            del self.ready_count[request_id]
                    logger.info(
                        "[D2RH-DEBUG] READY_SCHEDULER request_id=%s ready_count=%d/%d all_ready=%s",
                        request_id,
                        ready_cnt,
                        self.decode_tp_size,
                        request_id in self.ready_request,
                    )
                    self._send_ack(sock, identity, request_id)
                elif msg[0] == STAGING_FULL:
                    request_id = msg[1]
                    logger.info(
                        "CPU staging full for request %s, remove from all_requests to allow retry",
                        request_id,
                    )
                    with self.ready_lock:
                        self.all_requests.discard(request_id)
                        self.ready_count.pop(request_id, None)
                    self._send_ack(sock, identity, request_id)
                else:
                    logger.error("HostListeningThread got unexpected message %s", msg)
            except Exception as e:
                logger.error("HostListeningThread exception: %s", e, exc_info=True)

    @staticmethod
    def _send_ack(sock: zmq.Socket, identity: bytes, request_id: str) -> None:  # type: ignore
        while True:
            try:
                sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                break
            except zmq.Again:  # type: ignore
                logger.debug("Socket not ready, retrying ACK for request %s", request_id)
                time.sleep(0.01)


@dataclass
class SizedDict(OrderedDict):
    def __init__(self, max_size=16000, *args, **kwargs):
        self.max_size = max_size
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            self.popitem(last=False)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            value: dict[int, list[int]] = {}
            self[key] = value
            return value


class KVCacheTaskTracker:
    def __init__(self):
        super().__init__()

        self.done_task_lock = threading.Lock()
        self.finished_requests: set[str] = set()
        # Only used in prefill node. Tracks requests whose kv blocks freeing is
        # intentionally delayed. Each entry is a tuple of (request_id,
        # timestamp). If a request remains in this queue for too long, it will
        # be force-freed.
        self.delayed_free_requests: OrderedDict[str, float] = OrderedDict()
        self.reqs_to_process: set[str] = set()

    def add_req_to_process(self, request_id: str):
        self.reqs_to_process.add(request_id)

    def add_not_transfer_request(self, request_id: str):
        with self.done_task_lock:
            self.finished_requests.add(request_id)
            self.reqs_to_process.discard(request_id)

    def update_done_task_count(self, request_id: str):
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.finished_requests.add(request_id)
                self.reqs_to_process.discard(request_id)
                self.delayed_free_requests.pop(request_id, None)
            else:
                logger.warning(
                    "MooncakeConnector finish req not in reqs to process. "
                    "request_id=%s. "
                    "Possible cause: Request was already completed or not properly tracked. "
                    "Check: Verify request lifecycle and tracking logic.",
                    request_id,
                )

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        with self.done_task_lock:
            finished_requests = self.finished_requests.copy()
            expired_requests = self._retrieve_expired_requests()
            finished_requests.update(expired_requests)
            self.finished_requests.clear()
        return finished_requests

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        """Add a delayed free request."""
        with self.done_task_lock:
            if request_id in self.reqs_to_process:
                self.delayed_free_requests[request_id] = delay_start_time

    def _retrieve_expired_requests(self):
        """Retrieve all expired delayed requests."""
        expired_requests: set[str] = set()
        # Free delayed requests if they exceed the timeout
        current_time = time.time()
        while self.delayed_free_requests:
            request_id = next(iter(self.delayed_free_requests))
            delay_start_time = self.delayed_free_requests[request_id]
            if current_time - delay_start_time > envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:
                self.delayed_free_requests.popitem(last=False)
                self.reqs_to_process.discard(request_id)
                expired_requests.add(request_id)
                logger.info(
                    "Force freed expired request: %s. "
                    "Reason: Request exceeded timeout threshold (%s seconds). "
                    "Action: Resources have been forcibly released to prevent memory leak.",
                    request_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
            else:
                break
        return expired_requests


class KVCacheSendingThread(threading.Thread):
    def __init__(
        self,
        vllm_config: VllmConfig,
        tp_rank: int,
        prefill_tp_size: int,
        local_engine_id: str,
        side_channel_host: str,
        side_channel_port: int,
        metadata: MooncakeAgentMetadata,
        ready_event: threading.Event,
        kv_caches: dict[str, Any],
        pcp_rank: int,
    ):
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self.tp_rank = tp_rank
        self.prefill_tp_size = prefill_tp_size
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.local_engine_id = local_engine_id
        self.side_channel_host = side_channel_host
        self.side_channel_port = side_channel_port
        self.metadata = metadata
        self.ready_event = ready_event
        self.kv_caches = kv_caches
        self.pcp_rank = pcp_rank
        self.port_send_num: dict[str, int] = {}

        self.task_tracker = KVCacheTaskTracker()

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def add_not_transfer_request(self, request_id: str):
        self.task_tracker.add_not_transfer_request(request_id)

    def add_delayed_request(self, request_id: str, delay_start_time: float):
        return self.task_tracker.add_delayed_request(request_id, delay_start_time)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            # Listen for new requests for metadata. NOTE(rob): we need each rank
            # to have a unique port. This hack to keeps us moving. We will
            # switch when moving to etcd or where we have a single ZMQ socket in
            # the scheduler.
            device_index = self.pp_rank * self.tp_size + self.tp_rank + self.pcp_rank * self.prefill_tp_size
            handshake_port = self.side_channel_port + device_index
            path = make_zmq_path("tcp", self.side_channel_host, handshake_port)
            logger.info(
                "KVCacheSendingThread started listening on path: %s. Thread: tp_rank=%d, pp_rank=%d, pcp_rank=%d",
                path,
                self.tp_rank,
                self.pp_rank,
                self.pcp_rank,
            )
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.ready_event.set()
                self.run_busy_loop(sock)
        except Exception as e:
            logger.exception(
                "Mooncake KVCacheSendingThread encountered exception. "
                "Thread: tp_rank=%d, pp_rank=%d, listening_path=%s. "
                "Error: %s",
                self.tp_rank,
                self.pp_rank,
                path,
                e,
            )

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error(
                        "Invalid message format in KVCacheSendingThread. "
                        "Expected: at least 2 frames (identity + payload). "
                        "Actual: %d frames. "
                        "Frames: %s. "
                        "Check: Verify message sender implementation.",
                        len(frames),
                        frames,
                    )
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error(
                        "Invalid message format in KVCacheSendingThread. "
                        "Expected: exactly 1 payload frame. "
                        "Actual: %d payload frames. "
                        "Frames: %s. "
                        "Check: Verify message sender removes empty frames correctly.",
                        len(payload),
                        frames,
                    )
                    continue

                msg = decoder.decode(payload[0])
                if msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                elif msg[0] == DONE_RECVING_MSG:
                    logger.debug("Got DONE_RECVING_MSG for request %s", msg[1])
                    request_id = msg[1]
                    remote_port_send_num = msg[2]
                    if remote_port_send_num:
                        if request_id not in self.port_send_num:
                            self.port_send_num[request_id] = 0
                        self.port_send_num[request_id] += 1
                        device_index = self.pp_rank * self.tp_size + self.tp_rank + self.pcp_rank * self.prefill_tp_size
                        handshake_port = self.side_channel_port + device_index
                        if self.port_send_num[request_id] >= remote_port_send_num[handshake_port]["num"]:
                            self.task_tracker.update_done_task_count(request_id)
                            del self.port_send_num[request_id]
                    else:
                        self.task_tracker.update_done_task_count(request_id)
                    # Acknowledge the request completion.
                    while True:
                        try:
                            # Send ACK to the sender.
                            sock.send_multipart((identity, b"", b"ACK"), flags=zmq.NOBLOCK)  # type: ignore
                            break
                        except zmq.Again:  # type: ignore
                            # If the socket is not ready, retry sending.
                            logger.debug("Socket not ready, retrying to send ACK for request %s", msg[1])
                            time.sleep(0.01)
                else:
                    logger.error(
                        "Connection listener received unexpected message type. "
                        "Expected: GET_META_MSG or DONE_RECVING_MSG. "
                        "Actual: %s. "
                        "Full message: %s. "
                        "Check: Verify message protocol implementation.",
                        msg[0] if msg else "empty",
                        msg,
                    )
            except Exception as e:
                logger.error(
                    "Connection listener encountered exception during message processing. "
                    "Exception type: %s. "
                    "Error: %s. "
                    "Context: Processing frames from socket. "
                    "Check: Review message handling logic and socket state.",
                    type(e).__name__,
                    e,
                )


class D2RHThread(threading.Thread):
    """Hop1: pull KV from prefill NPU into host CPU staging."""

    def __init__(
        self,
        engine: TransferEngine,
        vllm_config: VllmConfig,
        tp_rank: int,
        tp_size: int,
        prefill_pp_size: int,
        cpu_kv_caches_base_addr: list[list[int]],
        block_len_per_addr: list[list[int]],
        block_stride_per_addr: list[list[int]],
        block_size_scale: list[list[int]],
        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]],
        host_staging: HostStagingManager,
        staging_allocations: dict[str, StagingAllocation],
        d2rh_handshake_port: int,
        scheduler_ready_port: int,
        prefill_pp_layer_partition: str | None = None,
    ):
        super().__init__(daemon=True, name=f"D2RHThread-TP{tp_rank}")
        self.engine = engine
        self.vllm_config = vllm_config
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._prefill_pp_size = prefill_pp_size
        self.cpu_kv_caches_base_addr = cpu_kv_caches_base_addr
        self.block_len_per_addr = block_len_per_addr
        self.block_stride_per_addr = block_stride_per_addr
        self.block_size_scale = block_size_scale
        self.kv_group2layeridx = kv_group2layeridx
        self.host_staging = host_staging
        self.staging_allocations = staging_allocations
        self.d2rh_handshake_port = d2rh_handshake_port
        self.scheduler_ready_port = scheduler_ready_port
        self.host_ip = get_ip()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.kv_caches_base_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_block_size_scale: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_block_stride_per_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_kv_group2layeridx: dict[str, dict[int, dict[int, tuple[dict[str, Any], list[int]]]]] = SizedDict()
        self.group_compress_ratios: dict[int, int] = {}
        for group_id, (group_spec, _) in self.kv_group2layeridx.items():
            compress_ratio = 1
            kv_cache_spec = group_spec.get("kv_cache_spec")
            if isinstance(kv_cache_spec, dict):
                for spec in kv_cache_spec.values():
                    if isinstance(spec, dict) and isinstance(spec.get("compress_ratio"), int):
                        compress_ratio = max(1, spec["compress_ratio"])
                        break
            self.group_compress_ratios[group_id] = compress_ratio
        self.block_size = vllm_config.cache_config.block_size
        hf_text_config = vllm_config.model_config.hf_text_config
        self.num_layers = getattr(hf_text_config, "num_hidden_layers", 0)
        self.num_speculative_tokens = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config is not None
            else 0
        )
        self.pp_layer_indices = {
            rank: get_prefill_pp_indices(self.num_layers, rank, self._prefill_pp_size, prefill_pp_layer_partition)
            for rank in range(self._prefill_pp_size)
        }
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)  # type: ignore
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0

    def add_request(self, req_meta: dict[str, Any]) -> None:
        self.request_queue.put(req_meta)

    def run(self) -> None:
        path = make_zmq_path("tcp", self.host_ip, self.d2rh_handshake_port)
        logger.info("Starting D2RH listener on tp_rank=%s path: %s", self.tp_rank, path)

        def zmq_listener_worker() -> None:
            try:
                with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                    self._run_start_pull_loop(sock)
            except Exception as e:
                logger.error("D2RH ZMQ listener crashed: %s", e, exc_info=True)

        threading.Thread(
            target=zmq_listener_worker,
            name=f"D2RH-ZMQListener-TP{self.tp_rank}",
            daemon=True,
        ).start()

        while True:
            req_meta = self.request_queue.get()
            try:
                self._transfer_kv_cache(req_meta)
            except Exception:
                logger.exception("D2RH transfer failed for request %s", req_meta.get("request_id"))
            finally:
                self.request_queue.task_done()

    def _run_start_pull_loop(self, sock: zmq.Socket) -> None:  # type: ignore
        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    continue
                identity = frames[0]
                payload = [frame for frame in frames[1:] if frame != b""]
                if len(payload) != 1:
                    continue
                msg = decoder.decode(payload[0])
                if msg[0] != START_PULL:
                    logger.error("D2RH got unexpected message %s", msg)
                    continue

                request_id = msg[1]
                pull_ack = b"ACK"
                try:
                    params = msg[2]
                    remote_request_id_preview = params.get("remote_request_id", request_id)
                    logger.info(
                        "[D2RH-DEBUG] START_PULL received tp_rank=%s d2rh_port=%s request_id=%s "
                        "remote_request_id=%s remote_block_ids=%s",
                        self.tp_rank,
                        self.d2rh_handshake_port,
                        request_id,
                        remote_request_id_preview,
                        params.get("remote_block_ids"),
                    )
                    remote_block_ids = params.get("remote_block_ids") or tuple()
                    if isinstance(remote_block_ids, list) and remote_block_ids and isinstance(remote_block_ids[0], list):
                        remote_block_ids = tuple(remote_block_ids)
                    elif isinstance(remote_block_ids, list) and remote_block_ids and isinstance(remote_block_ids[0], int):
                        remote_block_ids = (remote_block_ids,)

                    if HostStagingManager.staging_demand(remote_block_ids) == 0:
                        num_groups = len(remote_block_ids) if remote_block_ids else len(self.kv_group2layeridx)
                        empty_staging = StagingAllocation(
                            block_maps=[{} for _ in range(num_groups)],
                            allocated_blocks=[[] for _ in range(num_groups)],
                        )
                        remote_request_id = params.get("remote_request_id", request_id)
                        self.staging_allocations[request_id] = empty_staging
                        if remote_request_id != request_id:
                            self.staging_allocations[remote_request_id] = empty_staging
                        logger.info(
                            "[D2RH-DEBUG] staging allocated (empty) tp_rank=%s request_id=%s "
                            "remote_request_id=%s staging_keys=%s",
                            self.tp_rank,
                            request_id,
                            remote_request_id,
                            sorted(self.staging_allocations.keys()),
                        )
                        threading.Thread(
                            target=self._send_scheduler_ready,
                            args=(request_id,),
                            name=f"D2RH-Ready-{request_id}",
                            daemon=True,
                        ).start()
                    else:
                        staging = self.host_staging.try_alloc(request_id, remote_block_ids)
                        if staging is None:
                            pull_ack = STAGING_FULL
                            logger.info(
                                "[D2RH-DEBUG] START_PULL STAGING_FULL tp_rank=%s request_id=%s",
                                self.tp_rank,
                                request_id,
                            )
                        else:
                            remote_request_id = params.get("remote_request_id", request_id)
                            self.staging_allocations[request_id] = staging
                            if remote_request_id != request_id:
                                self.staging_allocations[remote_request_id] = staging
                            logger.info(
                                "[D2RH-DEBUG] staging allocated tp_rank=%s request_id=%s "
                                "remote_request_id=%s blocks=%s staging_keys=%s",
                                self.tp_rank,
                                request_id,
                                remote_request_id,
                                staging.allocated_blocks,
                                sorted(self.staging_allocations.keys()),
                            )
                            self.add_request(
                                {
                                    "request_id": request_id,
                                    "remote_request_id": remote_request_id,
                                    "remote_host": params["remote_host"],
                                    "remote_engine_id": params["remote_engine_id"],
                                    "remote_handshake_ports": params.get("remote_handshake_ports") or [],
                                    "remote_port_base": params["remote_port"],
                                    "remote_multi_nodes_meta_mapping": params.get("remote_multi_nodes_meta_mapping"),
                                    "remote_block_ids": remote_block_ids,
                                    "staging": staging,
                                    "num_computed_tokens": params.get("num_computed_tokens", 0),
                                }
                            )
                except Exception:
                    logger.exception("Failed to handle START_PULL for request %s", request_id)
                    pull_ack = STAGING_FULL

                while True:
                    try:
                        sock.send_multipart((identity, b"", pull_ack), flags=zmq.NOBLOCK)  # type: ignore
                        break
                    except zmq.Again:  # type: ignore
                        time.sleep(0.01)
            except Exception as e:
                logger.error("D2RH START_PULL loop exception: %s", e, exc_info=True)

    def _transfer_kv_cache(self, req_meta: dict[str, Any]) -> None:
        request_id = req_meta["request_id"]
        remote_request_id = req_meta.get("remote_request_id", request_id)
        logger.info(
            "[D2RH-DEBUG] Hop1 transfer begin tp_rank=%s request_id=%s remote_request_id=%s",
            self.tp_rank,
            request_id,
            remote_request_id,
        )
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        staging: StagingAllocation = req_meta["staging"]
        remote_handshake_ports = req_meta.get("remote_handshake_ports") or []
        remote_port_base = req_meta["remote_port_base"]
        default_remote_host = req_meta["remote_host"]
        default_remote_engine_id = req_meta["remote_engine_id"]
        remote_multi_nodes_meta_mapping = req_meta.get("remote_multi_nodes_meta_mapping")
        num_computed_tokens = req_meta.get("num_computed_tokens", 0)
        tp_num_need_pulls = max(len(remote_handshake_ports), 1)
        transfer_ok = True

        for inner_offset, remote_handshake_port in enumerate(remote_handshake_ports):
            remote_host, remote_engine_id = resolve_remote_host_for_handshake_port(
                remote_port_base,
                remote_handshake_port,
                default_remote_host,
                default_remote_engine_id,
                remote_multi_nodes_meta_mapping,
            )
            if (
                remote_engine_id not in self.kv_caches_base_addr
                or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]
            ):
                self._get_remote_metadata(remote_host, remote_handshake_port, remote_engine_id)

            remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
            remote_block_size_scale = self.remote_block_size_scale[remote_engine_id][remote_handshake_port]
            remote_block_stride_per_addr = self.remote_block_stride_per_addr[remote_engine_id][remote_handshake_port]
            te_rpc_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
            session_id = f"{remote_host}:{te_rpc_port}"

            local_list: list[int] = []
            peer_list: list[int] = []
            length_list: list[int] = []

            for group_idx, group_remote_block_ids in enumerate(remote_block_ids):
                if not group_remote_block_ids:
                    continue
                group_spec, layer_indices = self.kv_group2layeridx[group_idx]
                block_map = staging.block_maps[group_idx]
                is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
                if is_mamba_group:
                    transfer_block_idx = len(group_remote_block_ids) - self.num_speculative_tokens - 1
                    remote_block_id = group_remote_block_ids[transfer_block_idx]
                    cpu_block_id = block_map[remote_block_id]
                    for layer_idx in layer_indices:
                        self._append_mamba_d2rh_meta(
                            local_list,
                            peer_list,
                            length_list,
                            group_spec=group_spec,
                            cpu_layer_base_addr=self.cpu_kv_caches_base_addr[layer_idx],
                            remote_layer_base_addr=remote_kv_caches_base_addrs[layer_idx],
                            block_len=self.block_len_per_addr[layer_idx],
                            block_stride=self.block_stride_per_addr[layer_idx],
                            remote_block_stride=remote_block_stride_per_addr[layer_idx],
                            remote_block_id=remote_block_id,
                            cpu_block_id=cpu_block_id,
                            tp_num_need_pulls=tp_num_need_pulls,
                            remote_tp_offset=inner_offset,
                        )
                    continue

                local_scale = self.block_size_scale[layer_indices[0]][0]
                remote_scale = remote_block_size_scale[layer_indices[0]][0]
                num_computed_tokens = req_meta.get("num_computed_tokens", 0)
                kernel_cpu_block_ids, kernel_remote_block_ids = trim_kernel_blocks_for_prefix_cache(
                    [block_map[remote_block_id] for remote_block_id in group_remote_block_ids],
                    group_remote_block_ids,
                    local_scale,
                    remote_scale,
                    self.block_size,
                    self.group_compress_ratios[group_idx],
                    num_computed_tokens,
                )

                if tp_num_need_pulls == 1:
                    grouped_remote_block_ids, grouped_cpu_block_ids = group_concurrent_contiguous(
                        kernel_remote_block_ids, kernel_cpu_block_ids
                    )
                else:
                    grouped_remote_block_ids = [[block_id] for block_id in kernel_remote_block_ids]
                    grouped_cpu_block_ids = [[block_id] for block_id in kernel_cpu_block_ids]

                for layer_idx in layer_indices:
                    for cache_idx in range(len(self.cpu_kv_caches_base_addr[layer_idx])):
                        cpu_layer_base = self.cpu_kv_caches_base_addr[layer_idx][cache_idx]
                        remote_layer_base = remote_kv_caches_base_addrs[layer_idx][cache_idx]
                        block_len = self.block_len_per_addr[layer_idx][cache_idx]
                        block_stride = self.block_stride_per_addr[layer_idx][cache_idx]
                        remote_block_stride = remote_block_stride_per_addr[layer_idx][cache_idx]
                        inner_block_len = block_len // tp_num_need_pulls
                        transfer_remote_block_ids, transfer_cpu_block_ids = split_if_not_byte_contiguous(
                            grouped_remote_block_ids,
                            grouped_cpu_block_ids,
                            src_block_stride=remote_block_stride,
                            dst_block_stride=block_stride,
                            block_len=inner_block_len,
                        )
                        for remote_block_id, cpu_block_id in zip(transfer_remote_block_ids, transfer_cpu_block_ids):
                            local_list.append(
                                cpu_layer_base + cpu_block_id[0] * block_stride + inner_offset * inner_block_len
                            )
                            peer_list.append(remote_layer_base + remote_block_id[0] * remote_block_stride)
                            length_list.append(inner_block_len * len(cpu_block_id))

            if not local_list:
                continue
            ret = self.engine.batch_transfer_sync_read(session_id, local_list, peer_list, length_list)
            if ret < 0:
                logger.error(
                    "D2RH transfer failed request=%s session=%s port=%s",
                    remote_request_id,
                    session_id,
                    remote_handshake_port,
                )
                transfer_ok = False
                break
            self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port)

        if transfer_ok:
            logger.info(
                "[D2RH-DEBUG] Hop1 transfer done tp_rank=%s request_id=%s remote_request_id=%s",
                self.tp_rank,
                request_id,
                remote_request_id,
            )
            self._send_scheduler_ready(request_id)
        else:
            logger.error(
                "[D2RH-DEBUG] Hop1 transfer FAILED tp_rank=%s request_id=%s remote_request_id=%s",
                self.tp_rank,
                request_id,
                remote_request_id,
            )

    def _append_mamba_d2rh_meta(
        self,
        local_list: list[int],
        peer_list: list[int],
        length_list: list[int],
        group_spec: dict[str, Any],
        cpu_layer_base_addr: list[int],
        remote_layer_base_addr: list[int],
        block_len: list[int],
        block_stride: list[int],
        remote_block_stride: list[int],
        remote_block_id: int,
        cpu_block_id: int,
        tp_num_need_pulls: int,
        remote_tp_offset: int,
    ) -> None:
        remote_tp_size = self.tp_size * tp_num_need_pulls
        cpu_conv_addr, cpu_ssm_addr = cpu_layer_base_addr[:2]
        remote_conv_addr, remote_ssm_addr = remote_layer_base_addr[:2]
        cpu_conv_len, cpu_ssm_len = block_len[:2]
        cpu_conv_stride, cpu_ssm_stride = block_stride[:2]
        remote_conv_stride, remote_ssm_stride = remote_block_stride[:2]
        tp_ratio = tp_num_need_pulls
        remote_conv_len = cpu_conv_len // tp_ratio
        remote_ssm_len = cpu_ssm_len // tp_ratio

        if tp_ratio == 1:
            local_list.extend(
                [
                    cpu_conv_addr + cpu_block_id * cpu_conv_stride,
                    cpu_ssm_addr + cpu_block_id * cpu_ssm_stride,
                ]
            )
            peer_list.extend(
                [
                    remote_conv_addr + remote_block_id * remote_conv_stride,
                    remote_ssm_addr + remote_block_id * remote_ssm_stride,
                ]
            )
            length_list.extend([remote_conv_len, remote_ssm_len])
            return

        conv_shape = group_spec["shapes"][0]
        conv_dtype_size = group_spec["dtype_sizes"][0]
        hf_text_config = self.vllm_config.model_config.hf_text_config
        linear_key_head_dim = hf_text_config.linear_key_head_dim
        linear_num_key_heads = hf_text_config.linear_num_key_heads
        linear_value_head_dim = hf_text_config.linear_value_head_dim
        linear_num_value_heads = hf_text_config.linear_num_value_heads
        remote_num_key_heads = linear_num_key_heads // remote_tp_size
        remote_num_value_heads = linear_num_value_heads // remote_tp_size
        remote_conv_width = (
            remote_num_key_heads * 2 * linear_key_head_dim + remote_num_value_heads * linear_value_head_dim
        )
        remote_conv_offsets = [
            0,
            remote_num_key_heads * linear_key_head_dim,
            remote_num_key_heads * 2 * linear_key_head_dim,
        ]
        remote_conv_sizes = [
            remote_num_key_heads * linear_key_head_dim,
            remote_num_key_heads * linear_key_head_dim,
            remote_num_value_heads * linear_value_head_dim,
        ]
        for i in range(conv_shape[0]):
            for remote_conv_offset, remote_conv_size in zip(remote_conv_offsets, remote_conv_sizes):
                remote_addr_offset = (i * remote_conv_width + remote_conv_offset) * conv_dtype_size
                cpu_addr_offset = (
                    (i * remote_conv_width + remote_conv_offset) * tp_ratio + remote_tp_offset * remote_conv_size
                ) * conv_dtype_size
                local_list.append(cpu_conv_addr + cpu_block_id * cpu_conv_stride + cpu_addr_offset)
                peer_list.append(remote_conv_addr + remote_block_id * remote_conv_stride + remote_addr_offset)
                length_list.append(remote_conv_size * conv_dtype_size)
        local_list.append(
            cpu_ssm_addr + cpu_block_id * cpu_ssm_stride + remote_tp_offset * cpu_ssm_len // tp_num_need_pulls
        )
        peer_list.append(remote_ssm_addr + remote_block_id * remote_ssm_stride)
        length_list.append(remote_ssm_len)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int, remote_engine_id: str) -> None:
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id or remote_engine_id
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
            self.remote_block_size_scale[engine_id][remote_handshake_port] = agent_meta.block_size_scale
            self.remote_block_stride_per_addr[engine_id][remote_handshake_port] = agent_meta.block_strides
            self.remote_kv_group2layeridx[engine_id][remote_handshake_port] = agent_meta.kv_group2layeridx
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def _send_done_recv_signal(self, request_id: str, remote_host: str, remote_handshake_port: int) -> None:
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id, {}))
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}", timeout=self.timeout)
            if resp != b"ACK":
                raise RuntimeError(f"D2RH failed to receive ACK from prefill: {resp!r}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)

    def _send_scheduler_ready(self, request_id: str) -> None:
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(self.host_ip, self.scheduler_ready_port)
            data_bytes = self.encoder.encode((READY_SCHEDULER, request_id))
            ensure_zmq_send(sock, data_bytes, f"{self.host_ip}:{self.scheduler_ready_port}")
            resp = ensure_zmq_recv(sock, self.remote_poller, f"{self.host_ip}:{self.scheduler_ready_port}", timeout=self.timeout)
            if resp != b"ACK":
                raise RuntimeError(f"D2RH failed scheduler ready ACK: {resp!r}")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, self.host_ip, self.scheduler_ready_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()
        ctx = zmq.Context()  # type: ignore
        sock = make_zmq_socket(ctx=ctx, path=remote_path, socket_type=zmq.REQ, bind=False)  # type: ignore
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
        self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
        return sock

    def _return_remote_socket(self, sock: zmq.Socket, remote_host: str, remote_handshake_port: int) -> None:  # type: ignore
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


class KVCacheRecvingThread(threading.Thread):
    def __init__(
        self,
        tp_rank: int,
        tp_size: int,
        _prefill_pp_size: int,
        engine: TransferEngine,
        local_engine_id: str,
        local_handshake_port: int,
        side_channel_port: int,
        local_kv_caches_base_addr: list[list[int]],
        block_len_per_addr: list[list[int]],
        block_stride_per_addr: list[list[int]],
        is_hma_required=False,
        ready_event: threading.Event | None = None,
        vllm_config: VllmConfig | None = None,
        kv_caches: dict[str, Any] | None = None,
        prefill_pp_layer_partition: str | None = None,
        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] | None = None,
        block_size_scale: list[list[int]] | None = None,
        use_cpu_staging: bool = False,
        cpu_kv_caches_base_addr: list[list[int]] | None = None,
        cpu_te_rpc_port: int | None = None,
        host_staging: HostStagingManager | None = None,
        staging_allocations: dict[str, StagingAllocation] | None = None,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._prefill_pp_size = _prefill_pp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.engine = engine
        if ready_event is None:
            ready_event = threading.Event()
        self.ready_event = ready_event

        if kv_caches is None:
            kv_caches = {}
        self.kv_caches = kv_caches
        self.kv_caches_base_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.kv_caches_base_addr[local_engine_id][local_handshake_port] = local_kv_caches_base_addr
        self.block_len_per_addr = block_len_per_addr
        self.block_stride_per_addr = block_stride_per_addr
        if kv_group2layeridx is None:
            kv_group2layeridx = {}
        self.kv_group2layeridx = kv_group2layeridx
        self.group_compress_ratios: dict[int, int] = {}
        for group_id, (group_spec, _) in self.kv_group2layeridx.items():
            compress_ratio = 1
            kv_cache_spec = group_spec.get("kv_cache_spec")
            if isinstance(kv_cache_spec, dict):
                for spec in kv_cache_spec.values():
                    if isinstance(spec, dict) and isinstance(spec.get("compress_ratio"), int):
                        compress_ratio = max(1, spec["compress_ratio"])
                        break
            self.group_compress_ratios[group_id] = compress_ratio
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.remote_block_size_scale: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_block_stride_per_addr: dict[str, dict[int, list[list[int]]]] = SizedDict()
        self.remote_kv_group2layeridx: dict[str, dict[int, dict[int, tuple[dict[str, Any], list[int]]]]] = SizedDict()
        self.use_cpu_staging = use_cpu_staging
        self.cpu_kv_caches_base_addr = cpu_kv_caches_base_addr or []
        self.cpu_te_rpc_port = cpu_te_rpc_port or 0
        self.host_staging = host_staging
        self.staging_allocations = staging_allocations if staging_allocations is not None else {}

        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=32)

        self.task_tracker = KVCacheTaskTracker()

        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds

        assert vllm_config is not None
        self.vllm_config: VllmConfig = vllm_config
        self.model_config = self.vllm_config.model_config
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config is not None
            else 0
        )
        self.use_mla = self.model_config.is_deepseek_mla
        self.is_hma_required = is_hma_required
        self.block_size = self.vllm_config.cache_config.block_size
        try:
            hf_text_config = self.model_config.hf_text_config
            if hf_text_config is None:
                raise AttributeError
        except AttributeError:
            hf_text_config = self.model_config.hf_config
        self.num_layers = hf_text_config.num_hidden_layers
        if block_size_scale is None:
            block_size_scale = []
        self.block_size_scale = block_size_scale
        self.pp_layer_indices = {
            rank: get_prefill_pp_indices(self.num_layers, rank, self._prefill_pp_size, prefill_pp_layer_partition)
            for rank in range(self._prefill_pp_size)
        }
        self.proc_not_transfer_request: dict[str, bool] = {}
        self.failed_recv_requests: set[str] = set()
        self.invalid_block_ids: set[int] = set()
        self.failed_recv_requests_lock = threading.Lock()

        self.num_draft_layers = 0
        if self.vllm_config.speculative_config is not None:
            if self.vllm_config.speculative_config.method == "mtp":
                # all MTP layer use the same kv cache layer, so only need to transfer once
                self.num_draft_layers = 1
            elif (
                hasattr(self.vllm_config.speculative_config.draft_model_config, "hf_config")
                and getattr(self.vllm_config.speculative_config.draft_model_config.hf_config, "num_hidden_layers", None)
                is not None
            ):
                self.num_draft_layers = (
                    self.vllm_config.speculative_config.draft_model_config.hf_config.num_hidden_layers
                )

    def add_request(
        self,
        request_id: str,
        remote_request_id: str,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        group_pulls: list[GroupPull],
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        num_computed_tokens: int = 0,
        all_task_done: bool = False,
    ):
        """Add a new request to the queue for processing."""
        if remote_port_send_num is None:
            remote_port_send_num = {}
        trans_info = {
            "request_id": request_id,
            "local_block_ids": local_block_ids,
            "remote_block_ids": remote_block_ids,
            "group_pulls": group_pulls,
            "remote_engine_id": remote_engine_id,
            "remote_request_id": remote_request_id,
            "remote_host": remote_host,
            "remote_handshake_port": remote_handshake_port,
            "num_computed_tokens": num_computed_tokens,
            "remote_port_send_num": remote_port_send_num,
            "all_task_done": all_task_done,
        }
        logger.debug("Adding request %s to the queue.Trans info:%s", request_id, trans_info)
        self.request_queue.put(trans_info)

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def get_and_clear_invalid_block_ids(self) -> set[int]:
        """Get and clear block ids that failed to load."""
        with self.failed_recv_requests_lock:
            invalid_block_ids = self.invalid_block_ids
            self.invalid_block_ids = set()
        return invalid_block_ids

    def _is_failed_recv_request(self, request_id: str) -> bool:
        with self.failed_recv_requests_lock:
            return request_id in self.failed_recv_requests

    def _mark_failed_recv_request(self, request_id: str, local_block_ids: BlockIds) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.add(request_id)
            self.invalid_block_ids.update(local_block_ids[0])

    def _clear_failed_recv_request(self, request_id: str) -> None:
        with self.failed_recv_requests_lock:
            self.failed_recv_requests.discard(request_id)

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request. ")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error("Error in KVCacheTransferThread. error=%s. ", e)

    def _handle_request(self, req_meta: dict[str, Any]):
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        remote_port_send_num = req_meta["remote_port_send_num"]
        all_task_done = req_meta["all_task_done"]
        transfer_failed = self._is_failed_recv_request(request_id)

        try:
            if transfer_failed:
                self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                logger.warning("Skipping KV cache transfer for request. remote_request_id=%s. ", remote_request_id)
            else:
                try:
                    logger.debug("Starting to transfer KV cache for request %s.", remote_request_id)
                    if self.use_cpu_staging:
                        self._transfer_kv_cache_h2d(req_meta)
                    else:
                        self._transfer_kv_cache_all_groups(req_meta)
                    logger.debug("Finished transferring KV cache for request %s.", remote_request_id)
                except Exception as e:
                    transfer_failed = True
                    self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                    logger.exception("Failed to transfer KV cache for request %s: %s", remote_request_id, e)
        finally:
            if all_task_done:
                self.task_tracker.update_done_task_count(request_id)
                if request_id in self.proc_not_transfer_request:
                    del self.proc_not_transfer_request[request_id]
                self._clear_failed_recv_request(request_id)
            self.request_queue.task_done()
            if not self.use_cpu_staging:
                self._send_done_signal_to_free_remote_port(remote_request_id, remote_host, remote_port_send_num)
            if not self.use_cpu_staging:
                self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)
            elif sum(len(group_ids) for group_ids in req_meta.get("local_block_ids", tuple())) == 0:
                self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)

    def _wait_for_staging(self, request_id: str, remote_request_id: str) -> StagingAllocation:
        deadline = time.time() + 300.0
        wait_start = time.time()
        last_log_time = wait_start
        logger.info(
            "[D2RH-DEBUG] H2D wait staging begin tp_rank=%s request_id=%s remote_request_id=%s "
            "staging_keys=%s",
            self.tp_rank,
            request_id,
            remote_request_id,
            sorted(self.staging_allocations.keys()),
        )
        while time.time() < deadline:
            staging = self.staging_allocations.get(remote_request_id) or self.staging_allocations.get(request_id)
            if staging is not None:
                logger.info(
                    "[D2RH-DEBUG] H2D wait staging done tp_rank=%s request_id=%s remote_request_id=%s "
                    "waited_s=%.3f hit_key=%s",
                    self.tp_rank,
                    request_id,
                    remote_request_id,
                    time.time() - wait_start,
                    remote_request_id if remote_request_id in self.staging_allocations else request_id,
                )
                return staging
            now = time.time()
            if now - last_log_time >= 5.0:
                logger.info(
                    "[D2RH-DEBUG] H2D still waiting staging tp_rank=%s request_id=%s remote_request_id=%s "
                    "waited_s=%.1f staging_keys=%s",
                    self.tp_rank,
                    request_id,
                    remote_request_id,
                    now - wait_start,
                    sorted(self.staging_allocations.keys()),
                )
                last_log_time = now
            time.sleep(0.01)
        logger.error(
            "[D2RH-DEBUG] H2D wait staging TIMEOUT tp_rank=%s request_id=%s remote_request_id=%s "
            "waited_s=%.1f staging_keys=%s",
            self.tp_rank,
            request_id,
            remote_request_id,
            time.time() - wait_start,
            sorted(self.staging_allocations.keys()),
        )
        raise RuntimeError(
            f"CPU staging allocation missing for request {remote_request_id}. "
            "Ensure D2RHThread finished Hop1 before H2D."
        )

    def _transfer_kv_cache_h2d(self, req_meta: dict[str, Any]) -> None:
        """Hop2: pull KV from host CPU staging into local NPU."""
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        local_block_ids: BlockIds = req_meta["local_block_ids"]
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        group_pulls: list[GroupPull] = req_meta["group_pulls"]
        all_task_done = req_meta.get("all_task_done", False)

        num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)
        if num_local_blocks == 0:
            return

        logger.info(
            "[D2RH-DEBUG] H2D begin tp_rank=%s request_id=%s remote_request_id=%s "
            "num_local_blocks=%d all_task_done=%s staging_keys=%s",
            self.tp_rank,
            request_id,
            remote_request_id,
            num_local_blocks,
            all_task_done,
            sorted(self.staging_allocations.keys()),
        )
        staging = self._wait_for_staging(request_id, remote_request_id)
        if not any(staging.block_maps):
            return

        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        if (
            remote_engine_id not in self.kv_caches_base_addr
            or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]
        ):
            self._get_remote_metadata(remote_host, remote_handshake_port)
        remote_block_size_scale = self.remote_block_size_scale[remote_engine_id][remote_handshake_port]

        local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
        session_id = f"{get_ip()}:{self.cpu_te_rpc_port}"
        src_list: list[int] = []
        peer_list: list[int] = []
        length_list: list[int] = []
        attention_group_reformat_block_ids: list[tuple[tuple[int, list[list[int]], int, list[int]], bool]] = []

        def expand_block_ids(block_ids, scale):
            return [bid * scale + offset for bid in block_ids for offset in range(scale)]

        def pp_layer_indices(layer_indices: list[int], prefill_pp_rank: int) -> list[int]:
            first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
            if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
                end_layer_index += self.num_draft_layers
            return [layer_idx for layer_idx in layer_indices if first_layer_index <= layer_idx < end_layer_index]

        for group_pull in group_pulls:
            group_idx = group_pull.group_id
            group_spec, layer_indices = self.kv_group2layeridx[group_idx]
            layer_indices = pp_layer_indices(layer_indices, group_pull.prefill_pp_rank)
            if not layer_indices:
                continue
            tp_num_need_pulls = group_pull.num_group_pulls
            inner_offset = group_pull.remote_tp_offset
            block_map = staging.block_maps[group_idx]
            is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
            local_group_block_ids = local_block_ids[group_idx]
            remote_group_block_ids = remote_block_ids[group_idx]
            if not local_group_block_ids:
                continue

            if is_mamba_group:
                transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
                remote_block_id = remote_group_block_ids[transfer_block_idx]
                cpu_block_id = block_map[remote_block_id]
                for layer_idx in layer_indices:
                    self._append_mamba_transfer_meta(
                        src_list,
                        peer_list,
                        length_list,
                        group_spec=group_spec,
                        src_layer_base_addr=local_kv_caches_base_addrs[layer_idx],
                        dst_layer_base_addr=self.cpu_kv_caches_base_addr[layer_idx],
                        block_len=self.block_len_per_addr[layer_idx],
                        block_stride=self.block_stride_per_addr[layer_idx],
                        remote_block_stride=self.block_stride_per_addr[layer_idx],
                        remote_block_id=cpu_block_id,
                        local_block_id=local_group_block_ids[0],
                        tp_num_need_pulls=tp_num_need_pulls,
                        remote_tp_offset=inner_offset,
                    )
                continue

            local_scale = self.block_size_scale[layer_indices[0]][0]
            remote_scale = remote_block_size_scale[layer_indices[0]][0]
            num_computed_tokens = req_meta.get("num_computed_tokens", 0)
            cpu_logical_block_ids = [block_map[remote_block_id] for remote_block_id in remote_group_block_ids]
            kernel_local_block_ids, _ = trim_kernel_blocks_for_prefix_cache(
                local_group_block_ids,
                remote_group_block_ids,
                local_scale,
                remote_scale,
                self.block_size,
                self.group_compress_ratios[group_idx],
                num_computed_tokens,
            )
            kernel_cpu_block_ids = expand_kernel_block_ids(cpu_logical_block_ids, local_scale)[
                : len(kernel_local_block_ids)
            ]

            if tp_num_need_pulls == 1:
                grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                    kernel_cpu_block_ids, kernel_local_block_ids
                )
            else:
                grouped_remote_block_ids = [[block_id] for block_id in kernel_cpu_block_ids]
                grouped_local_block_ids = [[block_id] for block_id in kernel_local_block_ids]

            is_group_transfer_end = group_pull.is_group_transfer_end
            attention_group_reformat_block_ids.append(
                ((group_idx, grouped_local_block_ids, tp_num_need_pulls, layer_indices), is_group_transfer_end)
            )

            for layer_idx in layer_indices:
                for cache_idx in range(len(local_kv_caches_base_addrs[layer_idx])):
                    npu_layer_base = local_kv_caches_base_addrs[layer_idx][cache_idx]
                    cpu_layer_base = self.cpu_kv_caches_base_addr[layer_idx][cache_idx]
                    block_len = self.block_len_per_addr[layer_idx][cache_idx]
                    block_stride = self.block_stride_per_addr[layer_idx][cache_idx]
                    inner_block_len = block_len // tp_num_need_pulls
                    transfer_cpu_block_ids, transfer_npu_block_ids = split_if_not_byte_contiguous(
                        grouped_remote_block_ids,
                        grouped_local_block_ids,
                        src_block_stride=block_stride,
                        dst_block_stride=block_stride,
                        block_len=inner_block_len,
                    )
                    for cpu_block_id, npu_block_id in zip(transfer_cpu_block_ids, transfer_npu_block_ids):
                        src_list.append(npu_layer_base + npu_block_id[0] * block_stride + inner_offset * inner_block_len)
                        peer_list.append(cpu_layer_base + cpu_block_id[0] * block_stride)
                        length_list.append(inner_block_len * len(npu_block_id))

        if not src_list:
            return

        ret = self.engine.batch_transfer_sync_read(session_id, src_list, peer_list, length_list)
        if ret < 0:
            raise RuntimeError(f"H2D Mooncake transfer failed, ret: {ret}")

        if all_task_done and self.host_staging is not None:
            logger.info(
                "[D2RH-DEBUG] H2D free staging tp_rank=%s request_id=%s remote_request_id=%s",
                self.tp_rank,
                request_id,
                remote_request_id,
            )
            self.host_staging.free(request_id)
            self.host_staging.free(remote_request_id)
            self.staging_allocations.pop(request_id, None)
            self.staging_allocations.pop(remote_request_id, None)

        ready_attention_group_reformat_block_ids = [
            reformat_group
            for reformat_group, is_group_transfer_end in attention_group_reformat_block_ids
            if is_group_transfer_end
        ]
        if not ready_attention_group_reformat_block_ids:
            return

        gqa_reformat_groups = [
            (group_idx, grouped_local_block_ids, num_group_pulls, layer_indices)
            for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in ready_attention_group_reformat_block_ids
            if num_group_pulls > 1
        ]
        if self.is_hma_required:
            for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in gqa_reformat_groups:
                group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
                if group_kv_caches:
                    self.reformat_kv_cache_hybrid_linear_torch(grouped_local_block_ids, num_group_pulls, group_kv_caches)

    def _send_done_signal_to_free_remote_port(
        self, request_id: str, remote_host: str, remote_port_send_num: dict[int, RemotePortInfo]
    ):
        if self.side_channel_port != self.local_handshake_port or not remote_port_send_num:
            return
        if request_id not in self.proc_not_transfer_request:
            self.proc_not_transfer_request[request_id] = True
        if self.proc_not_transfer_request[request_id]:
            for remote_port in remote_port_send_num:
                if remote_port_send_num[remote_port]["num"] == 0:
                    remote_host_ = remote_port_send_num[remote_port]["host"]
                    self._send_done_recv_signal(request_id, remote_host_, remote_port, remote_port_send_num)
            self.proc_not_transfer_request[request_id] = False

    def _transfer_kv_cache_all_groups(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        remote_request_id = req_meta["remote_request_id"]
        local_block_ids: BlockIds = req_meta["local_block_ids"]
        remote_block_ids: BlockIds = req_meta["remote_block_ids"]
        group_pulls: list[GroupPull] = req_meta["group_pulls"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]

        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        num_local_blocks = sum(len(group_block_ids) for group_block_ids in local_block_ids)
        if num_local_blocks == 0:
            return

        # Check if we have the remote metadata cached.
        if (
            remote_engine_id not in self.kv_caches_base_addr
            or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]
        ):
            self._get_remote_metadata(remote_host, remote_handshake_port)
        remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
        local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port]
        remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
        remote_block_size_scale = self.remote_block_size_scale[remote_engine_id][remote_handshake_port]
        remote_block_stride_per_addr = self.remote_block_stride_per_addr[remote_engine_id][remote_handshake_port]
        session_id = f"{remote_host}:{remote_transfer_port}"

        req_start_time = time.perf_counter()
        src_list: list[int] = []
        dst_list: list[int] = []
        length_list: list[int] = []
        attention_group_reformat_block_ids: list[tuple[tuple[int, list[list[int]], int, list[int]], bool]] = []

        def expand_block_ids(block_ids, scale):
            return [bid * scale + offset for bid in block_ids for offset in range(scale)]

        def pp_layer_indices(layer_indices: list[int], prefill_pp_rank: int) -> list[int]:
            first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
            if self.vllm_config.speculative_config is not None and prefill_pp_rank == self._prefill_pp_size - 1:
                end_layer_index += self.num_draft_layers
            return [layer_idx for layer_idx in layer_indices if first_layer_index <= layer_idx < end_layer_index]

        for group_pull in group_pulls:
            group_idx = group_pull.group_id
            group_spec, layer_indices = self.kv_group2layeridx[group_idx]
            layer_indices = pp_layer_indices(layer_indices, group_pull.prefill_pp_rank)
            if not layer_indices:
                continue
            tp_num_need_pulls = group_pull.num_group_pulls
            inner_offset = group_pull.remote_tp_offset
            is_mamba_group = group_spec["kv_cache_spec_type"] == "MambaSpec"
            local_group_block_ids = local_block_ids[group_idx]
            remote_group_block_ids = remote_block_ids[group_idx]
            if not local_group_block_ids:
                continue
            if not is_mamba_group:
                is_group_transfer_end = group_pull.is_group_transfer_end
                local_scale = self.block_size_scale[layer_indices[0]][0]
                remote_scale = remote_block_size_scale[layer_indices[0]][0]
                num_computed_tokens = req_meta.get("num_computed_tokens", 0)
                kernel_local_block_ids, kernel_remote_block_ids = trim_kernel_blocks_for_prefix_cache(
                    local_group_block_ids,
                    remote_group_block_ids,
                    local_scale,
                    remote_scale,
                    self.block_size,
                    self.group_compress_ratios[group_idx],
                    num_computed_tokens,
                )

                if tp_num_need_pulls == 1:
                    grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                        kernel_remote_block_ids, kernel_local_block_ids
                    )
                else:
                    grouped_remote_block_ids = [[block_id] for block_id in kernel_remote_block_ids]
                    grouped_local_block_ids = [[block_id] for block_id in kernel_local_block_ids]
                attention_group_reformat_block_ids.append(
                    (
                        (group_idx, grouped_local_block_ids, tp_num_need_pulls, layer_indices),
                        is_group_transfer_end,
                    )
                )
            else:
                # For MambaSpec num block should equal on P node and D node
                if len(local_group_block_ids) != len(remote_group_block_ids):
                    raise RuntimeError("For MambaSpec num block should equal on P node and D node.")
                transfer_block_idx = len(remote_group_block_ids) - self.num_speculative_tokens - 1
                grouped_remote_block_ids = [[remote_group_block_ids[transfer_block_idx]]]
                grouped_local_block_ids = [[local_group_block_ids[0]]]

            if is_mamba_group:
                for layer_idx in layer_indices:
                    start_meta_idx = len(src_list)
                    self._append_mamba_transfer_meta(
                        src_list,
                        dst_list,
                        length_list,
                        group_spec=group_spec,
                        src_layer_base_addr=local_kv_caches_base_addrs[layer_idx],
                        dst_layer_base_addr=remote_kv_caches_base_addrs[layer_idx],
                        block_len=self.block_len_per_addr[layer_idx],
                        block_stride=self.block_stride_per_addr[layer_idx],
                        remote_block_stride=remote_block_stride_per_addr[layer_idx],
                        remote_block_id=grouped_remote_block_ids[0][0],
                        local_block_id=grouped_local_block_ids[0][0],
                        tp_num_need_pulls=tp_num_need_pulls,
                        remote_tp_offset=inner_offset,
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        for src, dst, length in zip(
                            src_list[start_meta_idx:], dst_list[start_meta_idx:], length_list[start_meta_idx:]
                        ):
                            logger.debug(
                                "Mooncake mamba transfer meta: request_id=%s group_idx=%s layer_idx=%s "
                                "local_block_id=%s remote_block_id=%s tp_num_need_pulls=%s "
                                "remote_tp_offset=%s  session_id=%s",
                                remote_request_id,
                                group_idx,
                                layer_idx,
                                grouped_local_block_ids[0][0],
                                grouped_remote_block_ids[0][0],
                                tp_num_need_pulls,
                                inner_offset,
                                session_id,
                            )
                continue

            for layer_idx in layer_indices:
                for cache_idx in range(len(local_kv_caches_base_addrs[layer_idx])):
                    src_layer_base_addr = local_kv_caches_base_addrs[layer_idx][cache_idx]
                    dst_layer_base_addr = remote_kv_caches_base_addrs[layer_idx][cache_idx]
                    block_len = self.block_len_per_addr[layer_idx][cache_idx]
                    block_stride = self.block_stride_per_addr[layer_idx][cache_idx]
                    remote_block_stride = remote_block_stride_per_addr[layer_idx][cache_idx]
                    inner_block_len = block_len // tp_num_need_pulls
                    transfer_remote_block_ids, transfer_local_block_ids = split_if_not_byte_contiguous(
                        grouped_remote_block_ids,
                        grouped_local_block_ids,
                        src_block_stride=remote_block_stride,
                        dst_block_stride=block_stride,
                        block_len=inner_block_len,
                    )
                    for remote_block_id, local_block_id in zip(transfer_remote_block_ids, transfer_local_block_ids):
                        src = src_layer_base_addr + local_block_id[0] * block_stride + inner_offset * inner_block_len
                        dst = dst_layer_base_addr + remote_block_id[0] * remote_block_stride
                        length = inner_block_len * len(local_block_id)
                        src_list.append(src)
                        dst_list.append(dst)
                        length_list.append(length)
                    logger.debug(
                        "Mooncake kv transfer meta: request_id=%s group_idx=%s layer_idx=%s local_block_ids=%s "
                        "remote_block_ids=%s tp_num_need_pulls=%s remote_tp_offset=%s session_id=%s",
                        remote_request_id,
                        group_idx,
                        layer_idx,
                        grouped_local_block_ids,
                        grouped_remote_block_ids,
                        tp_num_need_pulls,
                        inner_offset,
                        session_id,
                    )
        if not src_list:
            return

        logger.debug(
            "Mooncake transfer request=%s session id=%s src=%s dst=%s length=%s",
            remote_request_id,
            session_id,
            src_list,
            dst_list,
            length_list,
        )
        ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
        if ret < 0:
            logger.error(
                "Mooncake transfer failed for request. remote_request_id=%s, ret=%d. ",
                req_meta["remote_request_id"],
                ret,
            )
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms. local_ip %s local_device_id %s remote_session_id %s",
            remote_request_id,
            req_transfer_elapsed,
            get_ip(),
            self.tp_rank,
            session_id,
        )

        ready_attention_group_reformat_block_ids = []
        for reformat_group, is_group_transfer_end in attention_group_reformat_block_ids:
            if is_group_transfer_end:
                ready_attention_group_reformat_block_ids.append(reformat_group)
        if not ready_attention_group_reformat_block_ids:
            return

        gqa_reformat_groups = [
            (group_idx, grouped_local_block_ids, num_group_pulls, layer_indices)
            for (
                group_idx,
                grouped_local_block_ids,
                num_group_pulls,
                layer_indices,
            ) in ready_attention_group_reformat_block_ids
            if num_group_pulls > 1
        ]

        if self.is_hma_required:
            for group_idx, grouped_local_block_ids, num_group_pulls, layer_indices in gqa_reformat_groups:
                group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
                if not group_kv_caches:
                    continue
                self.reformat_kv_cache_hybrid_linear_torch(grouped_local_block_ids, num_group_pulls, group_kv_caches)
            return

        uniform_num_pulls = {num_group_pulls for _, _, num_group_pulls, _ in ready_attention_group_reformat_block_ids}
        if len(uniform_num_pulls) != 1:
            raise RuntimeError(
                f"Non-hybrid Mooncake KV reformat expects uniform group pulls, but got {uniform_num_pulls}."
            )

        num_group_pulls = next(iter(uniform_num_pulls))
        need_cat_cache = num_group_pulls > 1
        need_nz_cache = get_ascend_config().enable_kv_nz
        if not (need_cat_cache or need_nz_cache):
            return

        use_fused_op = ascend_envs.VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
        for group_idx, reformat_block_ids, _, layer_indices in ready_attention_group_reformat_block_ids:
            group_kv_caches = self._get_group_kv_caches(group_idx, layer_indices)
            if not group_kv_caches:
                continue
            if use_fused_op and enable_custom_op():
                if need_cat_cache:
                    self.reformat_kv_cache_with_fused_op(reformat_block_ids, num_group_pulls, group_kv_caches)
                if need_nz_cache:
                    self.reformat_kv_cache(reformat_block_ids, num_group_pulls, False, need_nz_cache, group_kv_caches)
            else:
                self.reformat_kv_cache(
                    reformat_block_ids,
                    num_group_pulls,
                    need_cat_cache,
                    need_nz_cache,
                    group_kv_caches,
                )

    @torch.no_grad()
    def reformat_kv_cache_hybrid_linear_torch(
        self, block_ids: list[list[int]], tp_num_need_pulls: int, group_kv_caches
    ):
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        if not flat_block_ids or tp_num_need_pulls == 1:
            return
        device = list(self.kv_caches.values())[0][0].device
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.long, device=device)
        num_blocks = block_ids_tensor.numel()

        def _transpose_cache_by_block(cache: torch.Tensor):
            # The transferred cache is laid out as
            # [block, split, token, head_per_split, dim]. Restore it to
            # [block, token, split, head_per_split, dim] in the selected blocks.
            selected = cache.index_select(0, block_ids_tensor)
            block_size = cache.shape[1]
            transposed = (
                selected.reshape(num_blocks, tp_num_need_pulls, block_size, -1)
                .transpose(1, 2)
                .contiguous()
                .reshape_as(selected)
            )
            cache.index_copy_(0, block_ids_tensor, transposed)

        for _, (k_cache_layer, v_cache_layer) in group_kv_caches.items():
            _transpose_cache_by_block(k_cache_layer)
            _transpose_cache_by_block(v_cache_layer)

    def _append_mamba_transfer_meta(
        self,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        group_spec: dict[str, Any],
        src_layer_base_addr: list[int],
        dst_layer_base_addr: list[int],
        block_len: list[int],
        block_stride: list[int],
        remote_block_stride: list[int],
        remote_block_id: int,
        local_block_id: int,
        tp_num_need_pulls: int,
        remote_tp_offset: int,
    ) -> None:
        remote_tp_size = self.tp_size * tp_num_need_pulls
        assert remote_tp_size >= self.tp_size, "Mamba prefill TP size must be >= decode TP size."
        assert remote_tp_size % self.tp_size == 0, "Mamba prefill TP size must be divisible by decode TP size."

        remote_conv_addr, remote_ssm_addr = dst_layer_base_addr[:2]
        local_conv_addr, local_ssm_addr = src_layer_base_addr[:2]
        local_conv_len, local_ssm_len = block_len[:2]
        local_conv_stride, local_ssm_stride = block_stride[:2]
        remote_conv_stride, remote_ssm_stride = remote_block_stride[:2]

        tp_ratio = tp_num_need_pulls
        remote_conv_len = local_conv_len // tp_ratio
        remote_ssm_len = local_ssm_len // tp_ratio

        if tp_ratio == 1:
            src_list.extend(
                [
                    local_conv_addr + local_block_id * local_conv_stride,
                    local_ssm_addr + local_block_id * local_ssm_stride,
                ]
            )
            dst_list.extend(
                [
                    remote_conv_addr + remote_block_id * remote_conv_stride,
                    remote_ssm_addr + remote_block_id * remote_ssm_stride,
                ]
            )
            length_list.extend([remote_conv_len, remote_ssm_len])
            return

        conv_shape = group_spec["shapes"][0]
        conv_dtype_size = group_spec["dtype_sizes"][0]

        linear_key_head_dim = self.vllm_config.model_config.hf_text_config.linear_key_head_dim
        linear_num_key_heads = self.vllm_config.model_config.hf_text_config.linear_num_key_heads
        linear_value_head_dim = self.vllm_config.model_config.hf_text_config.linear_value_head_dim
        linear_num_value_heads = self.vllm_config.model_config.hf_text_config.linear_num_value_heads
        remote_num_key_heads = linear_num_key_heads // remote_tp_size
        remote_num_value_heads = linear_num_value_heads // remote_tp_size
        remote_conv_width = (
            remote_num_key_heads * 2 * linear_key_head_dim + remote_num_value_heads * linear_value_head_dim
        )
        remote_conv_offsets = [
            0,
            remote_num_key_heads * linear_key_head_dim,
            remote_num_key_heads * 2 * linear_key_head_dim,
        ]
        remote_conv_sizes = [
            remote_num_key_heads * linear_key_head_dim,
            remote_num_key_heads * linear_key_head_dim,
            remote_num_value_heads * linear_value_head_dim,
        ]

        for i in range(conv_shape[0]):
            for remote_conv_offset, remote_conv_size in zip(remote_conv_offsets, remote_conv_sizes):
                remote_addr_offset = (i * remote_conv_width + remote_conv_offset) * conv_dtype_size
                local_addr_offset = (
                    (i * remote_conv_width + remote_conv_offset) * tp_ratio + remote_tp_offset * remote_conv_size
                ) * conv_dtype_size
                src_list.append(local_conv_addr + local_block_id * local_conv_stride + local_addr_offset)
                dst_list.append(remote_conv_addr + remote_block_id * remote_conv_stride + remote_addr_offset)
                length_list.append(remote_conv_size * conv_dtype_size)

        src_list.append(
            local_ssm_addr
            + local_block_id * local_ssm_stride
            + remote_tp_offset * local_ssm_len // tp_num_need_pulls
        )
        dst_list.append(remote_ssm_addr + remote_block_id * remote_ssm_stride)
        length_list.append(remote_ssm_len)

    def _get_group_kv_caches(self, group_idx: int, layer_indices: list[int] | None = None) -> dict[str, Any]:
        if layer_indices is None:
            _, layer_indices = self.kv_group2layeridx[group_idx]
        layer_index_set = set(layer_indices)
        num_attn_module = 2 if self.vllm_config.model_config.hf_text_config.model_type == "longcat_flash" else 1
        from vllm.v1.worker.utils import extract_layer_index

        def layer_in_group(layer_name: str) -> bool:
            if "mtp" in layer_name:
                return any(layer_idx >= self.num_layers for layer_idx in layer_index_set)
            return extract_layer_index(layer_name, num_attn_module) in layer_index_set

        return {
            layer_name: layer_cache for layer_name, layer_cache in self.kv_caches.items() if layer_in_group(layer_name)
        }

    @staticmethod
    def _get_kv_cache_dims_from_tensors(kv_caches: dict[str, Any]) -> tuple[int, int, int]:
        """Return (num_kv_heads, k_head_dim, v_head_dim) from registered KV cache tensors."""
        k_cache, v_cache = next(iter(kv_caches.values()))
        return int(k_cache.shape[-2]), int(k_cache.shape[-1]), int(v_cache.shape[-1])

    def reformat_kv_cache_with_fused_op(
        self,
        block_ids: list[list[int]],
        tp_num_need_pulls: int,
        kv_caches: dict[str, Any] | None = None,
    ):
        if kv_caches is None:
            kv_caches = self.kv_caches
        k_cache = list(kv_caches.values())[0][0]
        device = k_cache.device
        num_kv_head, head_dim, _ = self._get_kv_cache_dims_from_tensors(kv_caches)
        block_size = self.vllm_config.cache_config.block_size
        layers = len(kv_caches)
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int64, device=device)

        k_caches = []
        v_caches = []
        for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
            k_caches.append(k_cache_layer)
            v_caches.append(v_cache_layer)

        torch.ops._C_ascend.transpose_kv_cache_by_block(
            k_caches, v_caches, block_ids_tensor, block_size, num_kv_head, head_dim, tp_num_need_pulls, layers
        )

    def reformat_kv_cache(
        self,
        block_ids: list[list[int]],
        tp_num_need_pulls: int,
        need_cat_cache: bool = False,
        need_nz_cache: bool = False,
        kv_caches: dict[str, Any] | None = None,
    ):
        if kv_caches is None:
            kv_caches = self.kv_caches
        k_cache = list(kv_caches.values())[0][0]
        dtype = k_cache.dtype
        device = k_cache.device
        num_kv_heads, k_head_dim, v_head_dim = self._get_kv_cache_dims_from_tensors(kv_caches)

        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
        num_blocks = len(flat_block_ids)
        num_tokens = num_blocks * self.block_size

        # Create device tensors for copy operations
        block_table = block_ids_tensor.view(1, -1)
        block_len_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)

        k_buffer = torch.empty((num_tokens, num_kv_heads, k_head_dim), dtype=dtype, device=device)
        v_buffer = torch.empty((num_tokens, num_kv_heads, v_head_dim), dtype=dtype, device=device)

        # Create slot mapping for reshape operations
        block_offsets = torch.arange(0, self.block_size, dtype=torch.int32, device=device)
        slot_mapping = (
            block_offsets.reshape((1, self.block_size)) + block_ids_tensor.reshape((num_blocks, 1)) * self.block_size
        ).flatten()

        # FIXME: Right now, if we skip synchronization at this point, the system
        # will crash in GQA scenarios. However, we still haven't identified the
        # root cause.
        torch.npu.synchronize()

        # Process each layer in the KV cache
        for _, (k_cache_layer, v_cache_layer) in kv_caches.items():
            # Load cache data into buffers
            torch_npu.atb.npu_paged_cache_load(
                k_cache_layer,
                v_cache_layer,
                block_table,
                block_len_tensor,
                seq_starts=seq_start_tensor,
                key=k_buffer,
                value=v_buffer,
            )
            if need_cat_cache:
                self._cat_kv_cache(
                    k_cache_layer,
                    v_cache_layer,
                    k_buffer,
                    v_buffer,
                    tp_num_need_pulls,
                    num_blocks,
                    num_tokens,
                    slot_mapping,
                    num_kv_heads,
                )
            if need_nz_cache:
                self._nz_kv_cache(
                    k_cache_layer,
                    v_cache_layer,
                    k_buffer,
                    v_buffer,
                    slot_mapping,
                    num_kv_heads,
                    k_head_dim,
                    v_head_dim,
                )
        # Clean up buffers
        del k_buffer, v_buffer

    def _cat_kv_cache(
        self,
        k_cache_layer,
        v_cache_layer,
        k_buffer,
        v_buffer,
        tp_num_need_pulls,
        num_blocks,
        num_tokens,
        slot_mapping,
        num_kv_heads: int,
    ):
        def _transpose_kv_cache_between_head(buffer: torch.Tensor) -> torch.Tensor:
            buffer = buffer.view(num_blocks, tp_num_need_pulls, self.block_size, -1)
            buffer.transpose_(1, 2)
            return buffer.contiguous().view(num_tokens, num_kv_heads, -1)

        # Transpose KV cache
        k_buffer = _transpose_kv_cache_between_head(k_buffer)
        v_buffer = _transpose_kv_cache_between_head(v_buffer)

        # Reshape and cache the processed buffers
        torch_npu._npu_reshape_and_cache(
            key=k_buffer, value=v_buffer, key_cache=k_cache_layer, value_cache=v_cache_layer, slot_indices=slot_mapping
        )

    def _nz_kv_cache(
        self,
        k_cache_layer,
        v_cache_layer,
        k_buffer,
        v_buffer,
        slot_mapping,
        num_kv_heads: int,
        k_head_dim: int,
        v_head_dim: int,
    ):
        nz_fmt_last_dim = 16
        k_cache_layer = k_cache_layer.view(
            -1, k_head_dim * num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        v_cache_layer = v_cache_layer.view(
            -1, v_head_dim * num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        torch_npu.npu_scatter_pa_kv_cache(k_buffer, v_buffer, k_cache_layer, v_cache_layer, slot_mapping)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            assert engine_id != self.local_engine_id, (
                f"Conflict engine id {engine_id} with local engine id {self.local_engine_id}."
            )
            if agent_meta.kv_group2layeridx != self.kv_group2layeridx:
                logger.warning(
                    "Remote kv_group2layeridx is inconsistent with local. remote=%s, local=%s. ",
                    agent_meta.kv_group2layeridx,
                    self.kv_group2layeridx,
                )
            self.remote_kv_group2layeridx[engine_id][remote_handshake_port] = agent_meta.kv_group2layeridx
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
            self.remote_block_size_scale[engine_id][remote_handshake_port] = agent_meta.block_size_scale
            self.remote_block_stride_per_addr[engine_id][remote_handshake_port] = agent_meta.block_strides
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _send_done_recv_signal(
        self,
        request_id: str,
        remote_host: str,
        remote_handshake_port: int,
        remote_port_send_num: dict[int, RemotePortInfo],
    ):
        logger.debug(
            "Sending done recving signal for request %s to %s:%d", request_id, remote_host, remote_handshake_port
        )
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            data_bytes = self.encoder.encode((DONE_RECVING_MSG, request_id, remote_port_send_num))
            ensure_zmq_send(sock, data_bytes, f"{remote_host}:{remote_handshake_port}")
            resp = ensure_zmq_recv(
                sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}", timeout=self.timeout
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Received response for request %s: %s", request_id, resp.decode("utf-8"))
            if resp != b"ACK":
                logger.error(
                    "Failed to receive ACK for request. request_id=%s, source=%s:%d. ",
                    request_id,
                    remote_host,
                    remote_handshake_port,
                )
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning("Unexpected error occurred in socket. error=%s. ", e)
        finally:
            if sock is not None:
                self._return_remote_socket(sock, remote_host, remote_handshake_port)
                logger.debug("Returned socket to pool for %s:%d", remote_host, remote_handshake_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        """Get a socket to the remote host."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()

            ctx = zmq.Context()  # type: ignore
            sock = make_zmq_socket(
                ctx=ctx,
                path=remote_path,
                socket_type=zmq.REQ,  # type: ignore
                bind=False,
            )
            sock.setsockopt(
                zmq.SNDTIMEO,  # type: ignore
                int(self.timeout * 1000),
            )
            self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
            return sock

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        """Return the remote socket to the pool."""
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.requests: dict[str, ReqMeta] = {}
        self.requests_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
            num_computed_tokens=kv_transfer_params.get("num_computed_tokens", 0),
            remote_block_ids=kv_transfer_params["remote_block_ids"],
            remote_engine_id=kv_transfer_params["remote_engine_id"],
            remote_request_id=kv_transfer_params["remote_request_id"],
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_pcp_size=kv_transfer_params.get("remote_pcp_size", 1),
            remote_dcp_size=kv_transfer_params.get("remote_dcp_size", 1),
            remote_ptp_size=kv_transfer_params.get("remote_ptp_size"),
            remote_multi_nodes_meta_mapping=kv_transfer_params.get("remote_multi_nodes_meta_mapping", {}),
            num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
        )


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                vllm_config, str(self.engine_id), kv_cache_config
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id), kv_cache_config)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Get the block ids whose KV load failed."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        """MooncakeConnector does not save explicitly."""
        pass

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """
        Get the KVConnector handshake metadata for this connector.
        This metadata is used for out-of-band connector handshake
        between P/D workers.

        Returns:
            KVConnectorHandshakeMetadata: the handshake metadata.
            None if no handshake metadata is available.
        """
        assert self.connector_worker is not None
        return self.connector_worker.xfer_handshake_metadata

    def set_xfer_handshake_metadata(self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        assert self.connector_scheduler is not None
        self.connector_scheduler.set_xfer_handshake_metadata(metadata)


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        init_ascend_config(vllm_config)
        self.ascend_config = get_ascend_config()
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id = engine_id
        self.local_ip = get_ip()
        logger.info("Initializing Mooncake Scheduler %s", engine_id)

        self.side_channel_host = get_ip()
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.max_device_id = (
            vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.data_parallel_size
            * self.pcp_size
            * vllm_config.parallel_config.pipeline_parallel_size
        )

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        # Requests that need to start recv.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[str, tuple[Request, BlockIds, int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

        # master-slave meta information for cross-nodes
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.kv_cache_groups = kv_cache_config.kv_cache_groups
        self.use_hybrid = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(not isinstance(g.kv_cache_spec, FullAttentionSpec) for g in kv_cache_config.kv_cache_groups)
            and len(kv_cache_config.kv_cache_groups) > 1
        )
        self.use_compress = self._model_uses_compress()
        self.group_transfer_info = [self._get_group_transfer_info(group) for group in kv_cache_config.kv_cache_groups]
        self.need_truncate = self.use_compress or any(info.is_state_group for info in self.group_transfer_info)

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        prefill_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("prefill", {})
        decode_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("decode", {})
        self._prefill_tp_size = prefill_parallel_config.get("tp_size", vllm_config.parallel_config.tensor_parallel_size)
        self._prefill_pp_size = prefill_parallel_config.get("pp_size", 1)
        self._decode_tp_size = decode_parallel_config.get("tp_size", vllm_config.parallel_config.tensor_parallel_size)
        self._use_d2rh = bool(decode_parallel_config.get("use_d2rh", False))
        self.num_key_value_heads = vllm_config.model_config.hf_text_config.num_key_value_heads
        self.is_deepseek_mla = vllm_config.model_config.is_deepseek_mla
        self.use_sparse = hasattr(vllm_config.model_config.hf_text_config, "index_topk")
        self.local_host = get_ip()
        self.encoder = msgspec.msgpack.Encoder()
        self.remote_sockets: dict[str, deque[zmq.Socket]] = defaultdict(deque)  # type: ignore
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0
        self.remote_sockets_lock = threading.Lock()

        if self.kv_role == "kv_consumer" and self._use_d2rh:
            self.all_requests: set[str] = set()
            self.listeningthread = HostListeningThread(
                self.all_requests,
                self._decode_tp_size,
                get_scheduler_ready_zmq_port(vllm_config),
            )
            self.listeningthread.start()
        else:
            self.all_requests = set()
            self.listeningthread = None

    def _build_start_pull_params(
        self,
        request_id: str,
        params: dict[str, Any],
        decode_tp_rank: int,
        num_computed_tokens: int,
        prompt_len: int,
    ) -> dict[str, Any]:
        prefill_tp_size = params.get("remote_ptp_size") or self._prefill_tp_size
        remote_request_id = params.get("remote_request_id", request_id)
        remote_ranks_per_decode = get_remote_ranks_for_req(
            remote_request_id,
            prefill_tp_size,
            self._decode_tp_size,
            self._prefill_pp_size,
            self.num_key_value_heads,
            self.is_deepseek_mla,
            self.use_sparse,
        )
        p_ranks = remote_ranks_per_decode[decode_tp_rank]
        base_port = params["remote_port"]
        remote_handshake_ports = [base_port + p_rank for p_rank in p_ranks]
        remote_host, remote_engine_id = resolve_remote_host_for_handshake_port(
            base_port,
            remote_handshake_ports[0],
            params["remote_host"],
            params["remote_engine_id"],
            params.get("remote_multi_nodes_meta_mapping"),
        )
        remote_block_ids = params.get("remote_block_ids") or tuple()
        if isinstance(remote_block_ids, list):
            if remote_block_ids and isinstance(remote_block_ids[0], list):
                remote_block_ids = tuple(remote_block_ids)
            elif remote_block_ids and isinstance(remote_block_ids[0], int):
                remote_block_ids = (remote_block_ids,)
            else:
                remote_block_ids = tuple()
        effective_prompt_len = self._state_prefill_token_count(prompt_len)
        transfer_remote_block_ids = self._get_transfer_block_ids(remote_block_ids, effective_prompt_len)

        pull_params = copy.copy(params)
        pull_params["remote_host"] = remote_host
        pull_params["remote_engine_id"] = remote_engine_id
        pull_params["remote_handshake_ports"] = remote_handshake_ports
        pull_params["remote_block_ids"] = transfer_remote_block_ids
        pull_params["decode_tp_rank"] = decode_tp_rank
        # Keep token-level skip in D2RH/H2D via remote_start_idx (kernel granularity).
        pull_params["num_computed_tokens"] = num_computed_tokens
        return pull_params

    def _send_start_pull(self, request_id: str, params: dict[str, Any], d2rh_port: int) -> bytes:
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(self.local_host, d2rh_port)
            data_bytes = self.encoder.encode((START_PULL, request_id, params))
            ensure_zmq_send(sock, data_bytes, f"{self.local_host}:{d2rh_port}")
            return ensure_zmq_recv(
                sock,
                self.remote_poller,
                f"{self.local_host}:{d2rh_port}",
                timeout=self.timeout,
            )
        finally:
            if sock is not None:
                self._return_remote_socket(sock, self.local_host, d2rh_port)

    def _get_remote_socket(self, remote_host: str, remote_handshake_port: int) -> zmq.Socket:  # type: ignore
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            if self.remote_sockets[remote_path]:
                return self.remote_sockets[remote_path].popleft()
        ctx = zmq.Context()  # type: ignore
        sock = make_zmq_socket(ctx=ctx, path=remote_path, socket_type=zmq.REQ, bind=False)  # type: ignore
        sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))  # type: ignore
        self.remote_poller.register(sock, zmq.POLLIN)  # type: ignore
        return sock

    def _return_remote_socket(
        self,
        sock: zmq.Socket,  # type: ignore
        remote_host: str,
        remote_handshake_port: int,
    ) -> None:
        remote_path = make_zmq_path("tcp", remote_host, remote_handshake_port)
        with self.remote_sockets_lock:
            self.remote_sockets[remote_path].append(sock)

    def _model_uses_compress(self) -> bool:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        return isinstance(compress_ratios, (list, tuple, dict))

    def _get_group_transfer_info(self, group: Any) -> GroupTransferInfo:
        specs = self._get_group_unique_specs(group)
        first_spec = specs[0] if specs else group.kv_cache_spec
        block_size = getattr(group.kv_cache_spec, "block_size", getattr(first_spec, "block_size", self.block_size))
        is_state_group = any(isinstance(spec, MambaSpec) for spec in specs)
        sliding_window = 0
        compress_ratio = 1
        for spec in specs:
            if isinstance(spec, SlidingWindowSpec):
                sliding_window = spec.sliding_window
            elif hasattr(spec, "compress_ratio"):
                compress_ratio = spec.compress_ratio

        return GroupTransferInfo(
            tokens_per_block=block_size * max(1, int(compress_ratio)),
            blocks_per_window=cdiv(sliding_window, block_size) + 1 if sliding_window else 0,
            is_state_group=is_state_group,
        )

    def _get_group_unique_specs(self, group: Any) -> list[Any]:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            return [group.kv_cache_spec]

        specs = []
        for layer_name in group.layer_names:
            layer_spec = group.kv_cache_spec.kv_cache_specs[layer_name]
            if layer_spec not in specs:
                specs.append(layer_spec)
        return specs

    def _get_transfer_block_ids(self, block_ids: BlockIds, prompt_len: int) -> BlockIds:
        """Return the P-side blocks that contain transferable prompt KV.

        Attention KV can have extra blocks allocated by MTP. State groups such
        as Mamba are not context-block aligned with attention KV, so keep them
        unchanged and only clip attention-like groups.
        """
        if len(block_ids) == 0:
            return block_ids

        assert len(block_ids) == len(self.group_transfer_info), "Number of KV cache groups must match"

        transfer_block_ids = []
        for blocks, group_info in zip(block_ids, self.group_transfer_info):
            if group_info.is_state_group:
                transfer_block_ids.append(blocks)
            elif group_info.blocks_per_window > 0:
                window_blocks = blocks[-group_info.blocks_per_window :]
                if len(window_blocks) > 1:
                    window_blocks = [block_id for block_id in window_blocks if block_id != 0]
                transfer_block_ids.append(window_blocks)
            else:
                num_prompt_blocks = cdiv(prompt_len, group_info.tokens_per_block)
                transfer_block_ids.append(blocks[:num_prompt_blocks])
        return tuple(transfer_block_ids)

    def _get_swa_transfer_block_ids(self, block_ids: BlockIds) -> BlockIds:
        """Clip SWA groups to their window tail and drop placeholder block 0."""
        if len(block_ids) == 0:
            return block_ids

        assert len(block_ids) == len(self.group_transfer_info), "Number of KV cache groups must match"

        transfer_block_ids = []
        for blocks, group_info in zip(block_ids, self.group_transfer_info):
            if group_info.is_state_group or group_info.blocks_per_window == 0:
                transfer_block_ids.append(blocks)
            else:
                window_blocks = blocks[-group_info.blocks_per_window :]
                transfer_block_ids.append([block_id for block_id in window_blocks if block_id != 0])
        return tuple(transfer_block_ids)

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        # logger.info(f"[===] enter _state_prefill_token_count")
        if self.need_truncate and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        # logger.info(f"[===] enter _truncate_request_for_prefill")
        params = request.kv_transfer_params
        if (
            params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if params is not None and params.get("do_remote_prefill"):
            actual = self._state_prefill_token_count(request.num_prompt_tokens)
            params["num_computed_tokens"] = num_computed_tokens
            count = max(actual - num_computed_tokens, 0)
            if count <= 0:
                return 0, False

            if self._use_d2rh and self.kv_role == "kv_consumer":
                if request.request_id not in self.all_requests:
                    try:
                        got_staging_full = False
                        prompt_len = request.num_prompt_tokens
                        remote_request_id = params.get("remote_request_id", request.request_id)
                        parallel_config = self.vllm_config.parallel_config
                        logger.info(
                            "[D2RH-DEBUG] scheduler START_PULL plan request_id=%s remote_request_id=%s "
                            "decode_tp_size=%d runtime_tp_size=%d runtime_pp_size=%d runtime_pcp_size=%d "
                            "local_host=%s",
                            request.request_id,
                            remote_request_id,
                            self._decode_tp_size,
                            parallel_config.tensor_parallel_size,
                            parallel_config.pipeline_parallel_size,
                            parallel_config.prefill_context_parallel_size,
                            self.local_host,
                        )
                        for decode_tp_rank in range(self._decode_tp_size):
                            d2rh_port = get_d2rh_zmq_port(self.vllm_config, decode_tp_rank)
                            worker_expected_port = get_d2rh_zmq_port(
                                self.vllm_config, decode_tp_rank, pp_rank=0, pcp_rank=0
                            )
                            pull_params = self._build_start_pull_params(
                                request.request_id,
                                params,
                                decode_tp_rank,
                                num_computed_tokens,
                                prompt_len,
                            )
                            resp = self._send_start_pull(request.request_id, pull_params, d2rh_port)
                            logger.info(
                                "[D2RH-DEBUG] scheduler START_PULL sent request_id=%s remote_request_id=%s "
                                "decode_tp_rank=%d d2rh_port=%d worker_expected_port=%d target=%s:%d "
                                "num_computed_tokens=%d remote_block_ids=%s resp=%s",
                                request.request_id,
                                remote_request_id,
                                decode_tp_rank,
                                d2rh_port,
                                worker_expected_port,
                                self.local_host,
                                d2rh_port,
                                num_computed_tokens,
                                pull_params.get("remote_block_ids"),
                                resp.decode("utf-8", errors="replace"),
                            )
                            if resp == STAGING_FULL:
                                got_staging_full = True
                                break
                            if resp != b"ACK":
                                raise RuntimeError(f"Failed START_PULL ACK: {resp!r}")
                        if got_staging_full:
                            if self.listeningthread is not None:
                                with self.listeningthread.ready_lock:
                                    self.all_requests.discard(request.request_id)
                                    self.listeningthread.ready_count.pop(request.request_id, None)
                            return 0, False
                        self.all_requests.add(request.request_id)
                    except RuntimeError as e:
                        logger.warning("START_PULL failed for request %s: %s", request.request_id, e)
                        if self.listeningthread is not None:
                            with self.listeningthread.ready_lock:
                                if request.request_id in self.listeningthread.ready_request:
                                    self.all_requests.discard(request.request_id)
                        return 0, False
                else:
                    return 0, False

            return count, True

        if params is not None and params.get("do_remote_decode") and self.need_truncate:
            self._truncate_request_for_prefill(request)

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: num_external_tokens=%s, kv_transfer_params=%s",
            num_external_tokens,
            params,
        )

        if params is not None and (params.get("do_remote_prefill", False) or params.get("do_remote_decode", False)):
            self._reqs_in_batch.add(request.request_id)
        if params is not None and params.get("do_remote_prefill"):
            if params.get("remote_block_ids"):
                if all(p in params for p in ("remote_engine_id", "remote_host", "remote_port", "remote_request_id")):
                    local_block_ids = blocks.get_unhashed_block_ids_all_groups() if num_external_tokens > 0 else []
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (request, local_block_ids, num_external_tokens)
                else:
                    logger.warning("Got invalid KVTransferParams. params=%s. ", params)
            else:
                assert num_external_tokens == 0
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids, num_external_tokens) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            # For the case where there are no remote blocks to pull
            # (block_ids is empty), we don't need to schedule
            # an async read on the worker side.
            meta.add_new_req(
                request_id=req_id,
                local_block_ids=block_ids,
                num_external_tokens=num_external_tokens,
                kv_transfer_params=req.kv_transfer_params,
            )

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        meta.requests_to_send = self._reqs_need_send
        self._reqs_need_send = {}
        meta.reqs_in_batch = self._reqs_in_batch
        self._reqs_in_batch = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, request_status=%s, kv_transfer_params=%s", request.status, params
        )

        if (
            params is None
            or not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        computed_block_ids = block_ids
        computed_block_lens = [len(block_id_list) for block_id_list in computed_block_ids]
        delay_free_blocks = sum(computed_block_lens) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", len(computed_block_ids), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        num_prompt_blocks = math.ceil(request.num_prompt_tokens / self.block_size)
        computed_block_ids = self._get_transfer_block_ids(computed_block_ids, request.num_prompt_tokens)
        computed_block_ids = self._get_swa_transfer_block_ids(computed_block_ids)

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_pcp_size=self.pcp_size,
            remote_dcp_size=self.dcp_size,
            remote_ptp_size=self.tp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
        )

    def set_xfer_handshake_metadata(self, metadata: dict[int, KVConnectorHandshakeMetadata]) -> None:
        """
        Set the KV connector handshake metadata for this connector.

        Args:
            metadata (dict): the handshake metadata to set.
        """
        for local_rank, rank_metadata in metadata.items():
            self.multi_nodes_meta_mapping[str(local_rank)] = {
                "host": rank_metadata.local_ip,
                "engine_id": rank_metadata.engine_id,
            }


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: KVCacheConfig):
        self._get_prefill_decode_size(vllm_config)
        os.environ["ASCEND_TRANSFER_TIMEOUT"] = str(get_transfer_timeout_value())
        if self._prefill_tp_size < self._decode_tp_size:
            raise ValueError(
                f"prefill_tp_size: {self._prefill_tp_size} must be greater than"
                f" or equal to the decode_tp_size: {self._decode_tp_size}"
            )

        # Metadata.
        self.vllm_config = vllm_config
        self.ascend_config = get_ascend_config()
        self.engine_id = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.tp_group = get_tp_group()
        self.pp_rank = get_pp_group().rank_in_group
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
        self.dp_size = vllm_config.parallel_config.data_parallel_size_local
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.side_channel_host = get_ip()
        self.pcp_size = get_pcp_group().world_size
        self.total_layers = vllm_config.model_config.get_total_num_hidden_layers()
        # Assert that pp_size and pcp_size cannot both be greater than 1
        assert not (self.pp_size > 1 and self.pcp_size > 1), "pp and pcp cannot open in same time"
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        self.dcp_size = get_decode_context_model_parallel_world_size()
        self.dcp_rank = get_decode_context_model_parallel_rank() if self.dcp_size > 1 else 0

        self.max_device_id = self.tp_size * self.dp_size * self.pcp_size * self.pp_size
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_key_value_heads = self.vllm_config.model_config.hf_text_config.num_key_value_heads

        # kv cache config
        self.kv_cache_config = kv_cache_config
        self.num_blocks: int = kv_cache_config.num_blocks
        self.kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        self.use_hybrid = (
            not self.vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(not isinstance(g.kv_cache_spec, FullAttentionSpec) for g in self.kv_cache_config.kv_cache_groups)
            and len(self.kv_cache_config.kv_cache_groups) > 1
        )
        self._is_hma_required = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and any(
            not isinstance(g.kv_cache_spec, FullAttentionSpec) for g in kv_cache_config.kv_cache_groups
        )
        self._layer_specs = {
            layer: group.kv_cache_spec for group in kv_cache_config.kv_cache_groups for layer in group.layer_names
        }

        # Handshake base port
        self.side_channel_port = (
            vllm_config.kv_transfer_config.kv_port
            + vllm_config.parallel_config.data_parallel_rank
            * vllm_config.parallel_config.tensor_parallel_size
            * vllm_config.parallel_config.pipeline_parallel_size
            * self.pcp_size
        )
        device_index = (self.pp_rank + self.pcp_rank) * self.tp_size + self.tp_rank
        self.handshake_port = self.side_channel_port + device_index
        self.sockets: dict = {}
        self.engine = global_te.get_transfer_engine(self.side_channel_host, device_name=None)
        self.te_rpc_port = self.engine.get_rpc_port()

        # Background thread for sending or receiving KV caches.
        self.kv_send_thread: KVCacheSendingThread | None = None
        self.kv_recv_thread: KVCacheRecvingThread | None = None

        # Handshake metadata of this worker
        self.xfer_handshake_metadata: MooncakeAgentMetadata | None = None

        # kv_transfer variables
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        if self.vllm_config.model_config.is_deepseek_mla:
            self.tp_num_need_pulls = 1
        else:
            num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
            num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
            self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
        self.local_remote_block_port_mapping: dict[str, list[list[int]] | None] = {}
        self.remote_port_send_num: dict[str, dict[int, RemotePortInfo]] = {}
        self.staging_allocations: dict[str, StagingAllocation] = {}
        self.cpu_caches_hold: list[torch.Tensor] = []
        self.cpu_kv_caches_base_addr: list[list[int]] = []
        self.host_staging: HostStagingManager | None = None
        self.d2rh_thread: D2RHThread | None = None

    def _get_prefill_decode_size(self, vllm_config: VllmConfig):
        # get prefill tp and dp size from extra config
        prefill_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("prefill", {})

        assert "tp_size" in prefill_parallel_config
        self._prefill_tp_size = prefill_parallel_config["tp_size"]

        assert "dp_size" in prefill_parallel_config
        self._prefill_dp_size = prefill_parallel_config["dp_size"]
        # get prefill pp size from extra config
        self._prefill_pp_size = prefill_parallel_config.get("pp_size", 1)
        # get decode tp and dp size from extra config
        decode_parallel_config: dict[str, Any] = vllm_config.kv_transfer_config.get_from_extra_config("decode", {})
        assert "tp_size" in decode_parallel_config
        self._decode_tp_size = decode_parallel_config["tp_size"]
        assert "dp_size" in decode_parallel_config
        self._decode_dp_size = decode_parallel_config["dp_size"]
        # get prefill pp size from extra config
        self._decode_pp_size = decode_parallel_config.get("pp_size", 1)
        assert self._decode_pp_size == 1, "decode pp size must be 1"
        self._use_d2rh = bool(decode_parallel_config.get("use_d2rh", False))
        self._prefill_pp_layer_partition = prefill_parallel_config.get("pp_layer_partition")

    @staticmethod
    def _serialize_kv_group_spec(group_spec: Any) -> dict[str, Any]:
        def to_msgpackable(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                return {str(k): to_msgpackable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_msgpackable(item) for item in value]
            try:
                builtins_value = msgspec.to_builtins(value)
                if builtins_value is value:
                    return repr(value)
                return to_msgpackable(builtins_value)
            except TypeError:
                return repr(value)

        kv_cache_spec = group_spec.kv_cache_spec
        spec = kv_cache_spec
        if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
            spec = kv_cache_spec.kv_cache_specs
        serialized = {
            "layer_names": list(group_spec.layer_names),
            "kv_cache_spec_type": type(kv_cache_spec).__name__,
            "kv_cache_spec": to_msgpackable(spec),
        }
        if isinstance(kv_cache_spec, MambaSpec):
            serialized["shapes"] = [list(shape) for shape in kv_cache_spec.shapes]
            serialized["dtype_sizes"] = [
                torch.tensor([], dtype=dtype).element_size()
                for dtype in kv_cache_spec.dtypes  # type: ignore[misc]
            ]
        return serialized

    def _build_kv_group2layeridx(self) -> dict[int, tuple[dict[str, Any], list[int]]]:
        from vllm.v1.worker.utils import extract_layer_index

        kv_group2layeridx: dict[int, tuple[dict[str, Any], list[int]]] = {}
        num_attn_module = 2 if self.vllm_config.model_config.hf_text_config.model_type == "longcat_flash" else 1
        next_mtp_layer_idx = self.total_layers
        for group_id, group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            layer_indices = []
            for layer_name in group_spec.layer_names:
                if "mtp" in layer_name:
                    layer_idx = next_mtp_layer_idx
                    next_mtp_layer_idx += 1
                else:
                    layer_idx = extract_layer_index(layer_name, num_attn_module)
                layer_indices.append(layer_idx)
            kv_group2layeridx[group_id] = (self._serialize_kv_group_spec(group_spec), layer_indices)
        return kv_group2layeridx

    def _has_mamba_group(self) -> bool:
        return any(group_spec["kv_cache_spec_type"] == "MambaSpec" for group_spec, _ in self.kv_group2layeridx.values())

    @staticmethod
    def _as_kv_cache_tuple(kv_cache_tuple: Any) -> list[torch.Tensor]:
        if isinstance(kv_cache_tuple, (list, tuple)):
            return list(kv_cache_tuple)
        return [kv_cache_tuple]

    def _get_layer_spec(self, layer_name: str) -> Any:
        layer_spec = self._layer_specs[layer_name]
        if isinstance(layer_spec, UniformTypeKVCacheSpecs):
            layer_spec = layer_spec.kv_cache_specs[layer_name]
        return layer_spec

    def _get_mamba_conv_padding(self, layer_spec: Any) -> int:
        if not isinstance(layer_spec, MambaSpec):
            return 0
        conv_nbytes = torch.tensor([], dtype=layer_spec.dtypes[0]).element_size()  # type: ignore[misc]
        conv_shape = torch.Size(layer_spec.shapes[0])
        return self.num_blocks * conv_shape.numel() * conv_nbytes

    def _get_registered_kv_tensor_buffers(self, kv_caches: dict[str, torch.Tensor]) -> tuple[list[int], list[int]]:
        ptrs: list[int] = []
        lengths: list[int] = []

        conv_padding = 0
        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            shared_addrs: list[int] = []
            has_mtp = False
            for layer_name in kv_cache_tensor.shared_by:
                has_mtp = has_mtp or "mtp" in layer_name
                layer_spec = self._get_layer_spec(layer_name)
                conv_padding = max(conv_padding, self._get_mamba_conv_padding(layer_spec))
                for single_kv_cache in self._as_kv_cache_tuple(kv_caches[layer_name]):
                    shared_addrs.append(single_kv_cache.data_ptr())

            if not shared_addrs:
                continue
            base_addr = min(shared_addrs)
            if has_mtp:
                base_addr -= conv_padding
            assert base_addr % (2 * 1024 * 1024) == 0, f"Tensor start addr {base_addr} is not align with 2M."
            ptrs.append(base_addr)
            lengths.append(kv_cache_tensor.size)

        return ptrs, lengths

    def _get_registered_kv_tensor_buffers_hybrid(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> tuple[list[int], list[int]]:
        ptrs: list[int] = []
        lengths: list[int] = []

        for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
            shared_addrs: list[int] = []
            for layer_name in kv_cache_tensor.shared_by:
                for single_kv_cache in self._as_kv_cache_tuple(kv_caches[layer_name]):
                    shared_addrs.append(single_kv_cache.data_ptr())

            if not shared_addrs:
                continue
            base_addr = min(shared_addrs)
            assert base_addr % (2 * 1024 * 1024) == 0, f"Tensor start addr {base_addr} is not align with 2M."
            ptrs.append(base_addr)
            lengths.append(kv_cache_tensor.size)

        return ptrs, lengths

    def _get_registered_layer_buffers(self, kv_caches: dict[str, torch.Tensor]) -> tuple[list[int], list[int]]:
        ptrs: list[int] = []
        lengths: list[int] = []

        for kv_cache_tuple in kv_caches.values():
            for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
                ptrs.append(single_kv_cache.data_ptr())
                lengths.append(single_kv_cache.element_size() * math.prod(single_kv_cache.shape))

        return ptrs, lengths

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        self.use_mla = self.vllm_config.model_config.is_deepseek_mla
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")

        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.kv_caches = kv_caches
        # Maps each KV cache group to its serialized group spec and physical
        # layer indices: {group_id: (group_spec, [layer_idx0, layer_idx1, ...])}.
        self.kv_group2layeridx = self._build_kv_group2layeridx()
        has_mamba_group = self._has_mamba_group()
        layer_name_to_idx = {
            layer_name: layer_idx
            for _, (group_spec, layer_indices) in self.kv_group2layeridx.items()
            for layer_name, layer_idx in zip(group_spec["layer_names"], layer_indices)
        }
        metadata_layers = max(layer_name_to_idx.values(), default=-1) + 1
        # Per-layer registered KV cache base addresses:
        # [layer_idx][cache_idx] -> data_ptr of one cache tensor, e.g. K/V.
        self.kv_caches_base_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer block scaling between logical KV blocks and tensor blocks:
        # [layer_idx][cache_idx] -> cache tensor num_blocks / logical num_blocks.
        self.block_size_scale: list[list[int]] = [[] for _ in range(metadata_layers)]
        # Per-layer byte length of one tensor block:
        # [layer_idx][cache_idx] -> element_size * prod(block_shape).
        self.block_len_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        self.block_shape_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]
        self.block_stride_per_addr: list[list[int]] = [[] for _ in range(metadata_layers)]

        for layer_name, kv_cache_tuple in kv_caches.items():
            layer_idx = layer_name_to_idx[layer_name]
            for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
                tensor_num_blocks = single_kv_cache.shape[0]
                block_size_scale = tensor_num_blocks // self.num_blocks
                block_shape = single_kv_cache.shape[1:]
                self.block_len_per_addr[layer_idx].append(single_kv_cache.element_size() * math.prod(block_shape))
                self.block_stride_per_addr[layer_idx].append(single_kv_cache.stride(0) * single_kv_cache.element_size())
                self.block_shape_per_addr[layer_idx].append(single_kv_cache.shape)
                self.block_size_scale[layer_idx].append(block_size_scale)
                self.kv_caches_base_addr[layer_idx].append(single_kv_cache.data_ptr())

        self.cpu_kv_caches_base_addr = [[] for _ in range(metadata_layers)]
        cpu_register_ptrs: list[int] = []
        cpu_register_lengths: list[int] = []

        if self.kv_role == "kv_consumer" and self._use_d2rh:
            self.host_staging = HostStagingManager(self.num_blocks)
            for layer_name, kv_cache_tuple in kv_caches.items():
                layer_idx = layer_name_to_idx[layer_name]
                for single_kv_cache in self._as_kv_cache_tuple(kv_cache_tuple):
                    cpu_cache = torch.empty(
                        single_kv_cache.shape,
                        dtype=single_kv_cache.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    self.cpu_caches_hold.append(cpu_cache)
                    self.cpu_kv_caches_base_addr[layer_idx].append(cpu_cache.data_ptr())
                    cpu_register_ptrs.append(cpu_cache.data_ptr())
                    cpu_register_lengths.append(
                        single_kv_cache.element_size() * math.prod(single_kv_cache.shape)
                    )

        if has_mamba_group:
            ptrs, lengths = self._get_registered_kv_tensor_buffers(kv_caches)
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        elif self.use_hybrid:
            ptrs, lengths = self._get_registered_kv_tensor_buffers_hybrid(kv_caches)
            register_regions = RegisterRegions(ptrs=ptrs, lengths=lengths)
        else:
            register_regions = collect_storage_merged_register_regions(kv_caches)

        if self.kv_role == "kv_consumer" and self._use_d2rh and cpu_register_ptrs:
            register_regions = RegisterRegions(
                ptrs=register_regions.ptrs + cpu_register_ptrs,
                lengths=register_regions.lengths + cpu_register_lengths,
            )

        validate_register_region_count(register_regions)
        global_te.register_buffer(register_regions.ptrs, register_regions.lengths)

        logger.debug(
            "Mooncake register kv caches metadata: kv_group2layeridx=%s, kv_caches_base_addr=%s, "
            "block_len_per_addr=%s, block_stride_per_addr=%s, block_shape_per_addr=%s, "
            "block_size_scale=%s, ptrs=%s, lengths=%s",
            self.kv_group2layeridx,
            self.kv_caches_base_addr,
            self.block_len_per_addr,
            self.block_stride_per_addr,
            self.block_shape_per_addr,
            self.block_size_scale,
            register_regions.ptrs,
            register_regions.lengths,
        )
        # After KV Caches registered, start the sending or receiving thread.
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            kv_group2layeridx=self.kv_group2layeridx,
            block_size=self.block_size,
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_size_scale=self.block_size_scale,
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_addr,
            block_strides=self.block_stride_per_addr,
            local_ip=get_ip(),
        )
        self.xfer_handshake_metadata = metadata

        ready_event = threading.Event()
        if self.kv_role == "kv_producer":
            self.kv_send_thread = KVCacheSendingThread(
                self.vllm_config,
                self.tp_rank,
                self._prefill_tp_size,
                self.engine_id,
                self.side_channel_host,
                self.side_channel_port,
                metadata,
                ready_event,
                self.kv_caches,
                self.pcp_rank,
            )
            self.kv_send_thread.start()
        else:
            if self._use_d2rh and self.host_staging is not None:
                d2rh_port = get_d2rh_zmq_port(self.vllm_config, self.tp_rank, self.pp_rank, self.pcp_rank)
                scheduler_ready_port = get_scheduler_ready_zmq_port(self.vllm_config)
                scheduler_would_use_port = get_d2rh_zmq_port(self.vllm_config, self.tp_rank)
                logger.info(
                    "[D2RH-DEBUG] worker D2RH thread init tp_rank=%s pp_rank=%s pcp_rank=%s "
                    "d2rh_port=%s scheduler_would_use_port=%s port_match=%s "
                    "decode_tp_size=%s runtime_tp_size=%s host=%s",
                    self.tp_rank,
                    self.pp_rank,
                    self.pcp_rank,
                    d2rh_port,
                    scheduler_would_use_port,
                    d2rh_port == scheduler_would_use_port,
                    self._decode_tp_size,
                    self.tp_size,
                    self.side_channel_host,
                )
                self.d2rh_thread = D2RHThread(
                    engine=self.engine,
                    vllm_config=self.vllm_config,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    prefill_pp_size=self._prefill_pp_size,
                    cpu_kv_caches_base_addr=self.cpu_kv_caches_base_addr,
                    block_len_per_addr=self.block_len_per_addr,
                    block_stride_per_addr=self.block_stride_per_addr,
                    block_size_scale=self.block_size_scale,
                    kv_group2layeridx=self.kv_group2layeridx,
                    host_staging=self.host_staging,
                    staging_allocations=self.staging_allocations,
                    d2rh_handshake_port=d2rh_port,
                    scheduler_ready_port=scheduler_ready_port,
                    prefill_pp_layer_partition=self._prefill_pp_layer_partition,
                )
                self.d2rh_thread.start()

            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank,
                self.tp_size,
                self._prefill_pp_size,
                self.engine,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                self.kv_caches_base_addr,
                self.block_len_per_addr,
                self.block_stride_per_addr,
                self._is_hma_required,
                ready_event,
                self.vllm_config,
                self.kv_caches,
                self._prefill_pp_layer_partition,
                self.kv_group2layeridx,
                self.block_size_scale,
                use_cpu_staging=self.host_staging is not None,
                cpu_kv_caches_base_addr=self.cpu_kv_caches_base_addr,
                cpu_te_rpc_port=self.te_rpc_port,
                host_staging=self.host_staging,
                staging_allocations=self.staging_allocations,
            )
            self.kv_recv_thread.start()
        start_wait_time = time.time()
        thread = self.kv_send_thread if self.kv_role == "kv_producer" else self.kv_recv_thread
        assert thread is not None
        while not ready_event.is_set():
            if not thread.is_alive():
                raise RuntimeError("KV Cache sending/receiving thread failed to start.")
            if time.time() - start_wait_time > 5 * 60:
                raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
            time.sleep(3)

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending = (
            self.kv_send_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.kv_role == "kv_producer"
            else set()
        )
        done_recving = (
            self.kv_recv_thread.get_and_clear_finished_requests(  # type: ignore[union-attr]
            )
            if self.kv_role == "kv_consumer"
            else set()
        )
        if self.tp_rank == 0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Number of completed KV cache send requests: %d, receive requests: %d",
                    len(done_sending),
                    len(done_recving),
                )
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self.kv_role == "kv_consumer" and self.kv_recv_thread is not None:
            return self.kv_recv_thread.get_and_clear_invalid_block_ids()
        return set()

    def _get_kv_split_metadata(
        self,
        req_id: str,
        meta: ReqMeta,
    ) -> tuple[list[list[int]], list[BlockIds], list[BlockIds]]:
        """Build per-transfer port and block-id metadata for remote KV reads.

        Args:
            req_id: Remote request id used as the stable hash key when choosing
                prefill TP ranks.
            meta: Request-level transfer metadata from the scheduler. It
                contains remote/local block ids, remote P-side port base,
                remote P-side PCP/DCP/PTP sizes, and prompt/prefix-cache
                token counts.

        Returns:
            A tuple of three aligned lists. Index ``i`` describes one transfer
            shard for this local D-side rank:
            * remote_handshake_port_list[i]: remote P worker handshake ports
              to pull from. The inner list length is the number of TP pulls
              needed for that shard.
            * local_block_ids_list[i]: local block ids, grouped by KV cache
              group, where received blocks are written.
            * remote_block_ids_list[i]: remote block ids, grouped by KV cache
              group, where blocks are read from.

        In PCP/DCP scenarios, prompt blocks can be split across multiple remote
        P workers. This method also accounts for unequal P/D prefix-cache hits
        by reducing the number of remote blocks that still need to be pulled.
        """
        prefill_tp_size: int = meta.remote_ptp_size if meta.remote_ptp_size is not None else self._prefill_tp_size

        if meta.remote_pcp_size * meta.remote_dcp_size * self.pcp_size * self.dcp_size == 1:
            if self._is_hma_required:
                chosen_rank_list, _ = self._get_hybrid_remote_rank_group_pulls(req_id, prefill_tp_size)
            else:
                chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)
            remote_handshake_port_list = [[x + meta.remote_port for x in chosen_rank_list]]
            local_block_ids_list = [meta.local_block_ids for _ in remote_handshake_port_list]
            remote_block_ids_list = [meta.remote_block_ids for _ in remote_handshake_port_list]
            return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

        def context_parallel_parameters_check():
            assert (meta.remote_pcp_size * meta.remote_dcp_size) % (self.pcp_size * self.dcp_size) == 0
            if not (self.use_mla or self.use_sparse):
                p_node_heads_per_rank = math.ceil(self.num_key_value_heads / prefill_tp_size)
                d_node_heads_per_rank = math.ceil(self.num_key_value_heads / self.tp_size)
                assert d_node_heads_per_rank % p_node_heads_per_rank == 0

        def get_kv_head_groups(tp_size):
            if self.use_mla or self.use_sparse:
                kv_head_groups = []
                kv_head_ids = [0]
                kv_head_groups.append(tuple(kv_head_ids))
                return kv_head_groups
            if self.num_key_value_heads // tp_size >= 1:
                kv_head_groups = []
                for tp_rank in range(tp_size):
                    kv_head_ids = [
                        head_idx + tp_rank * (self.num_key_value_heads // tp_size)
                        for head_idx in range(self.num_key_value_heads // tp_size)
                    ]
                    kv_head_groups.append(tuple(kv_head_ids))
                return kv_head_groups
            if tp_size // self.num_key_value_heads > 1:
                kv_head_groups = []
                for kv_head_ids_ in range(self.num_key_value_heads):
                    kv_head_groups.append(tuple([kv_head_ids_]))
                return kv_head_groups

        def get_cp_group_meta(tp_size, pcp_size, dcp_size, port_base):
            # key is kv_head_group, value is cp_groups and which cp_groups to select
            cp_group_meta: dict = {}
            kv_head_groups = get_kv_head_groups(tp_size)
            dcp_repeat_num = tp_size // len(kv_head_groups) // dcp_size

            for kv_head_group_idx, kv_head_group in enumerate(kv_head_groups):
                if kv_head_group not in cp_group_meta:
                    cp_group_meta[kv_head_group] = {}
                    cp_group_meta[kv_head_group]["cp_groups"] = []
                    cp_group_meta[kv_head_group]["select_cp_groups_id"] = 0
                kv_head_group_offset = tp_size // len(kv_head_groups) * kv_head_group_idx
                for dcp_repeat_idx in range(dcp_repeat_num):
                    # len(cp_group) == pcp_size * dcp_size
                    cp_group = []
                    dcp_repeat_offset = dcp_size * dcp_repeat_idx
                    for pcp_rank in range(pcp_size):
                        pcp_rank_offset = tp_size * pcp_rank
                        for dcp_rank in range(dcp_size):
                            cp_group.append(
                                dcp_rank + port_base + pcp_rank_offset + dcp_repeat_offset + kv_head_group_offset
                            )
                    cp_group_meta[kv_head_group]["cp_groups"].append(cp_group)

            return cp_group_meta

        def get_local_remote_block_port_mappings():
            context_parallel_parameters_check()
            p_node_cp_group_meta = get_cp_group_meta(
                prefill_tp_size, meta.remote_pcp_size, meta.remote_dcp_size, meta.remote_port
            )
            d_node_cp_group_meta = get_cp_group_meta(self.tp_size, self.pcp_size, self.dcp_size, self.side_channel_port)
            local_remote_block_port_mappings: dict[int, list[list[int]]] = {}
            for d_node_head_key in d_node_cp_group_meta:
                for p_node_head_key in p_node_cp_group_meta:
                    if not set(p_node_head_key).issubset(set(d_node_head_key)):
                        continue
                    d_node_head_group = d_node_cp_group_meta[d_node_head_key]
                    p_node_head_group = p_node_cp_group_meta[p_node_head_key]
                    for d_cp_group in d_node_head_group["cp_groups"]:
                        select_cp_groups_id = p_node_head_group["select_cp_groups_id"]
                        p_cp_groups = p_node_head_group["cp_groups"]
                        p_cp_group = p_cp_groups[select_cp_groups_id]
                        p_node_head_group["select_cp_groups_id"] = (
                            select_cp_groups_id + 1 if select_cp_groups_id + 1 < len(p_cp_groups) else 0
                        )
                        for d_idx, d_port in enumerate(d_cp_group):
                            if d_port not in local_remote_block_port_mappings:
                                local_remote_block_port_mappings[d_port] = []
                            p_port_remote_list = []
                            for p_idx, p_port in enumerate(p_cp_group):
                                if p_idx % len(d_cp_group) == d_idx:
                                    p_port_remote_list.append(p_port)
                            local_remote_block_port_mappings[d_port].append(p_port_remote_list)

            logger.info(
                "p_node_cp_group_meta is:: %s. d_node_cp_group_meta is:: %s. "
                "local_remote_block_port_mappings is:: %s. ",
                p_node_cp_group_meta,
                d_node_cp_group_meta,
                local_remote_block_port_mappings,
            )

            return local_remote_block_port_mappings

        def get_remote_port_send_num(
            local_remote_block_port_mappings: dict[int, list[list[int]]],
        ) -> dict[int, RemotePortInfo]:
            remote_port_send_num: dict[int, RemotePortInfo] = {}
            for port in range(prefill_tp_size * meta.remote_pcp_size):
                remote_host_info = meta.remote_multi_nodes_meta_mapping.get(str(port), None)
                if remote_host_info is None:
                    remote_host = meta.remote_host
                else:
                    remote_host = remote_host_info["host"]
                remote_port_send_num[meta.remote_port + port] = {"num": 0, "host": remote_host}

            for remote_port_head_list in local_remote_block_port_mappings.values():
                for remote_port_list in remote_port_head_list:
                    for remote_port in remote_port_list:
                        remote_port_send_num[remote_port]["num"] += 1
            return remote_port_send_num

        if meta.remote_engine_id not in self.local_remote_block_port_mapping:
            self.local_remote_block_port_mapping[meta.remote_engine_id] = None

        if self.local_remote_block_port_mapping[meta.remote_engine_id] is None:
            local_remote_block_port_mappings = get_local_remote_block_port_mappings()
            self.local_remote_block_port_mapping[meta.remote_engine_id] = local_remote_block_port_mappings[
                self.handshake_port
            ]
            self.remote_port_send_num[meta.remote_engine_id] = get_remote_port_send_num(
                local_remote_block_port_mappings
            )

        local_remote_block_port_mapping = copy.deepcopy(self.local_remote_block_port_mapping[meta.remote_engine_id])

        num_external_blocks = math.ceil(meta.num_external_tokens / self.block_size)

        kv_group_items = list(self.kv_group2layeridx.items())
        sequence_group_idx = next(
            (
                group_idx
                for group_idx, (group_spec, _) in kv_group_items
                if group_spec["kv_cache_spec_type"] != "MambaSpec"
            ),
            0,
        )
        assert math.ceil(num_external_blocks / (self.pcp_size * self.dcp_size)) == len(
            meta.local_block_ids[sequence_group_idx]
        ), (
            f"num_external_blocks({num_external_blocks}), cp_size({self.pcp_size * self.dcp_size}), "
            f"local_block_ids_len ({len(meta.local_block_ids[sequence_group_idx])})"
        )
        assert meta.num_prompt_blocks >= num_external_blocks, (
            f"meta.num_prompt_blocks({meta.num_prompt_blocks}), num_external_blocks({num_external_blocks})"
        )

        remote_cp_size = meta.remote_pcp_size * meta.remote_dcp_size
        remote_block_nums_all = [meta.num_prompt_blocks // remote_cp_size] * remote_cp_size
        num_remain_blocks = meta.num_prompt_blocks % remote_cp_size
        for i in range(num_remain_blocks):
            remote_block_nums_all[i] += 1
        last_block_location = (num_remain_blocks + remote_cp_size - 1) % remote_cp_size

        # Considering prefix cache, the remote_block_nums_all should be revised
        num_prefix_cached_blocks = meta.num_prompt_blocks - num_external_blocks
        remote_block_nums_all = [num - num_prefix_cached_blocks // remote_cp_size for num in remote_block_nums_all]
        num_remain_blocks = num_prefix_cached_blocks % remote_cp_size
        for i in range(num_remain_blocks):
            remote_block_nums_all[i] -= 1

        # make sure the last block (which may be unfull) of P nodes is put to the last block of D node
        remote_block_nums: list[int] = []
        final_block_idx: int | None = None
        local_cp_rank = self.dcp_rank + self.pcp_rank * self.dcp_size
        local_cp_size = self.dcp_size * self.pcp_size
        for cp_rank, block_num in enumerate(remote_block_nums_all):
            if cp_rank % local_cp_size == local_cp_rank:
                if last_block_location == cp_rank:
                    final_block_idx = len(remote_block_nums)
                remote_block_nums.append(block_num)

        assert local_remote_block_port_mapping is not None
        if final_block_idx is not None:
            final_block_num = remote_block_nums.pop(final_block_idx)
            remote_block_nums.append(final_block_num)
            for mapping in local_remote_block_port_mapping:
                final_block_port = mapping.pop(final_block_idx)
                mapping.append(final_block_port)

        remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = [], [], []
        for idx in range(len(local_remote_block_port_mapping[0])):
            mapping_list = []
            for mapping in local_remote_block_port_mapping:
                mapping_list.append(mapping[idx])
            remote_handshake_port_list.append(mapping_list)

        # the local_block_ids_list and remote_block_ids_list are related with remote_handshake_port_list
        # such as: local_block_ids_list[[1],[2],[5],[6]], remote_block_ids_list[[1],[1],[1],[1]],
        # remote_handshake_port_list[[30000],[30001],[30004],[30005]]
        # D rank will get remote block 1 in port 30004 and save it in local block 5
        local_block_offset = 0
        for remote_kv_id in range(len(remote_handshake_port_list)):
            num_blocks_to_pull = remote_block_nums[remote_kv_id]
            group_remote_block_ids: list[list[int]] = []
            group_local_block_ids: list[list[int]] = []
            is_final_shard = remote_kv_id == len(remote_handshake_port_list) - 1
            for group_idx, (group_spec, _) in kv_group_items:
                if group_spec["kv_cache_spec_type"] == "MambaSpec":
                    # Mamba state is not context-block sharded like attention
                    # KV. Transfer the final state from the final PCP/DCP shard.
                    group_remote_block_ids.append(list(meta.remote_block_ids[group_idx]) if is_final_shard else [])
                    group_local_block_ids.append(list(meta.local_block_ids[group_idx]) if is_final_shard else [])
                    continue
                group_remote_block_ids.append(list(meta.remote_block_ids[group_idx][:num_blocks_to_pull]))
                group_local_block_ids.append(
                    list(meta.local_block_ids[group_idx][local_block_offset : local_block_offset + num_blocks_to_pull])
                )
            remote_block_ids_list.append(tuple(group_remote_block_ids))
            local_block_ids_list.append(tuple(group_local_block_ids))
            local_block_offset += num_blocks_to_pull

        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        assert tp_num_need_pulls == len(remote_handshake_port_list[0]), (
            f"tp_num_need_pulls: {tp_num_need_pulls}, remote_handshake_port_list: {remote_handshake_port_list}"
        )

        return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

    def _get_group_pulls_metadata(
        self,
        req_id: str,
        remote_handshake_port_list: list[list[int]],
        prefill_tp_size: int,
        remote_base_port: int,
    ) -> list[list[list[GroupPull]]]:
        """Build per-port KV cache group pull descriptors.

        Args:
            req_id: Remote request id used to reproduce hybrid-attention rank
                selection for the same request.
            remote_handshake_port_list: Output from ``_get_kv_split_metadata``.
                Each outer item is one transfer shard; each inner item is a
                remote P worker handshake port.
            prefill_tp_size: Effective remote prefill TP size. This may come
                from ``meta.remote_ptp_size`` when P and D use different TP
                sizes.
            remote_base_port: Remote P-side handshake base port. A remote
                worker rank is ``remote_handshake_port - remote_base_port``.

        Returns:
            A three-level list aligned with ``remote_handshake_port_list``:
            ``result[shard_idx][remote_port_idx]`` is the list of ``GroupPull``
            entries for that remote port. Each ``GroupPull`` identifies the KV
            cache group, the remote TP offset to read, the number of pulls
            needed to assemble that group, the prefill PP rank, and whether
            this pull is the final pull for the group. The final-pull flag is
            used by the receiver to decide when group reformatting can run.
        """
        if self._is_hma_required:
            _, rank_group_pulls = self._get_hybrid_remote_rank_group_pulls(req_id, prefill_tp_size)
            num_pp_tp_ranks = prefill_tp_size * self._prefill_pp_size
            return [
                [
                    rank_group_pulls[(remote_handshake_port - remote_base_port) % num_pp_tp_ranks]
                    for remote_handshake_port in remote_ports
                ]
                for remote_ports in remote_handshake_port_list
            ]

        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        group_ids = [group_id for group_id, (_, layer_indices) in self.kv_group2layeridx.items() if layer_indices]

        def make_group_pulls(remote_tp_offset: int, prefill_pp_rank: int) -> list[GroupPull]:
            return [
                GroupPull(
                    group_id=group_id,
                    remote_tp_offset=remote_tp_offset,
                    num_group_pulls=tp_num_need_pulls,
                    prefill_pp_rank=prefill_pp_rank,
                    is_group_transfer_end=remote_tp_offset == tp_num_need_pulls - 1,
                )
                for group_id in group_ids
            ]

        group_pulls_list = []
        for pcp_dcp_rank, remote_ports in enumerate(remote_handshake_port_list):
            if len(remote_ports) == 1:
                remote_tp_offsets = [pcp_dcp_rank % tp_num_need_pulls]
                prefill_pp_ranks = [
                    ((remote_ports[0] - remote_base_port) % (prefill_tp_size * self._prefill_pp_size))
                    // prefill_tp_size
                ]
            else:
                assert len(remote_ports) % tp_num_need_pulls == 0, (
                    f"tp_num_need_pulls: {tp_num_need_pulls}, remote_ports: {remote_ports}"
                )
                remote_tp_offsets = [rank_idx % tp_num_need_pulls for rank_idx in range(len(remote_ports))]
                prefill_pp_ranks = [
                    ((remote_port - remote_base_port) % (prefill_tp_size * self._prefill_pp_size)) // prefill_tp_size
                    for remote_port in remote_ports
                ]
            group_pulls_list.append(
                [
                    make_group_pulls(remote_tp_offset, prefill_pp_rank)
                    for remote_tp_offset, prefill_pp_rank in zip(remote_tp_offsets, prefill_pp_ranks)
                ]
            )
        return group_pulls_list

    def _get_hybrid_remote_rank_group_pulls(
        self,
        req_id: str,
        prefill_tp_size: int,
    ) -> tuple[list[int], dict[int, list[GroupPull]]]:
        rank_group_pulls: OrderedDict[int, list[GroupPull]] = OrderedDict()

        def add_group_pull(remote_rank: int, group_pull: GroupPull) -> None:
            rank_group_pulls.setdefault(remote_rank, []).append(group_pull)

        for group_id, (group_spec, layer_indices) in self.kv_group2layeridx.items():
            if not layer_indices:
                continue

            if group_spec["kv_cache_spec_type"] == "MambaSpec":
                assert prefill_tp_size % self.tp_size == 0, (
                    f"Hybrid Mamba prefill tp size({prefill_tp_size}) must be divisible by "
                    f"decode tp size({self.tp_size})."
                )
                num_group_pulls = prefill_tp_size // self.tp_size
                for pp_rank in range(self._prefill_pp_size):
                    pp_rank_offset = pp_rank * prefill_tp_size
                    local_tp_offset = self.tp_rank * num_group_pulls
                    for remote_tp_offset in range(num_group_pulls):
                        remote_rank = pp_rank_offset + local_tp_offset + remote_tp_offset
                        add_group_pull(
                            remote_rank,
                            GroupPull(
                                group_id=group_id,
                                remote_tp_offset=remote_tp_offset,
                                num_group_pulls=num_group_pulls,
                                prefill_pp_rank=pp_rank,
                                is_group_transfer_end=remote_tp_offset == num_group_pulls - 1,
                            ),
                        )
                continue

            num_group_pulls = self._get_attention_group_num_need_pulls(group_spec, prefill_tp_size)
            chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)
            assert len(chosen_rank_list) == num_group_pulls * self._prefill_pp_size, (
                f"chosen_rank_list({chosen_rank_list}) does not match num_group_pulls({num_group_pulls}) "
                f"and prefill pp size({self._prefill_pp_size})."
            )
            for rank_idx, remote_rank in enumerate(chosen_rank_list):
                prefill_pp_rank = rank_idx // num_group_pulls
                add_group_pull(
                    remote_rank,
                    GroupPull(
                        group_id=group_id,
                        remote_tp_offset=rank_idx % num_group_pulls,
                        num_group_pulls=num_group_pulls,
                        prefill_pp_rank=prefill_pp_rank,
                        is_group_transfer_end=rank_idx % num_group_pulls == num_group_pulls - 1,
                    ),
                )

        return list(rank_group_pulls), dict(rank_group_pulls)

    def _get_attention_group_num_need_pulls(self, group_spec: dict[str, Any], prefill_tp_size: int) -> int:
        num_key_value_heads = self._get_attention_group_num_key_value_heads(group_spec)
        num_d_block_heads = max(1, num_key_value_heads // self.tp_size)
        num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
        return num_d_block_heads // num_p_block_heads

    def _get_attention_group_num_key_value_heads(self, group_spec: dict[str, Any]) -> int:
        kv_cache_spec = group_spec.get("kv_cache_spec", {})
        if isinstance(kv_cache_spec, dict):
            for key in ("num_kv_heads", "num_key_value_heads"):
                num_key_value_heads = kv_cache_spec.get(key)
                if isinstance(num_key_value_heads, int):
                    return num_key_value_heads
        return self.num_key_value_heads

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        """Start loading KV blocks from remote engine."""
        for req_id in metadata.reqs_in_batch:
            if self.kv_send_thread is not None:
                self.kv_send_thread.task_tracker.add_req_to_process(req_id)
            if self.kv_recv_thread is not None:
                self.kv_recv_thread.task_tracker.add_req_to_process(req_id)

        for req_id, meta in metadata.requests.items():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "start_load_kv for request %s from remote engine %s. "
                    "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                    req_id,
                    meta.remote_engine_id,
                    len(meta.local_block_ids),
                    len(meta.remote_block_ids),
                )

            remote_req_id = meta.remote_request_id
            prefill_tp_size: int = meta.remote_ptp_size if meta.remote_ptp_size is not None else self._prefill_tp_size

            (
                remote_handshake_port_list,
                local_block_ids_list,
                remote_block_ids_list,
            ) = self._get_kv_split_metadata(remote_req_id, meta)
            group_pulls_list = self._get_group_pulls_metadata(
                remote_req_id,
                remote_handshake_port_list,
                prefill_tp_size,
                meta.remote_port,
            )

            for pcp_dcp_rank, remote_ports in enumerate(remote_handshake_port_list):
                for remote_tp_offset, remote_handshake_port in enumerate(remote_ports):
                    assert self.kv_recv_thread is not None
                    remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                        meta.remote_port,
                        remote_handshake_port,
                        meta.remote_host,
                        meta.remote_engine_id,
                        meta.remote_multi_nodes_meta_mapping,
                    )
                    remote_port_send_num = (
                        self.remote_port_send_num[meta.remote_engine_id]
                        if meta.remote_pcp_size * meta.remote_dcp_size > 1
                        else None
                    )
                    all_task_done = (
                        pcp_dcp_rank == len(remote_handshake_port_list) - 1
                        and remote_tp_offset == len(remote_ports) - 1
                    )
                    logger.info(
                        "[D2RH-DEBUG] start_load_kv queue H2D tp_rank=%s request_id=%s "
                        "remote_request_id=%s remote_handshake_port=%s all_task_done=%s "
                        "staging_keys=%s",
                        self.tp_rank,
                        req_id,
                        remote_req_id,
                        remote_handshake_port,
                        all_task_done,
                        sorted(self.staging_allocations.keys()),
                    )
                    self.kv_recv_thread.add_request(
                        request_id=req_id,
                        remote_request_id=remote_req_id,
                        local_block_ids=local_block_ids_list[pcp_dcp_rank],
                        remote_block_ids=remote_block_ids_list[pcp_dcp_rank],
                        group_pulls=group_pulls_list[pcp_dcp_rank][remote_tp_offset],
                        remote_engine_id=remote_engine_id,
                        remote_host=remote_host,
                        remote_handshake_port=remote_handshake_port,
                        remote_port_send_num=remote_port_send_num,
                        num_computed_tokens=meta.num_computed_tokens,
                        all_task_done=all_task_done,
                    )

        if self.kv_send_thread is not None and self.pcp_size * self.dcp_size == 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                if self.tp_rank in self._prefill_get_remote_rank(req_id):
                    self.kv_send_thread.add_delayed_request(req_id, delay_start_time)
                else:
                    self.kv_send_thread.add_not_transfer_request(req_id)

        if self.kv_send_thread is not None and self.pcp_size * self.dcp_size > 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                self.kv_send_thread.add_delayed_request(req_id, delay_start_time)

    def _get_tp_num_need_pulls(self, prefill_tp_size: int | None) -> int:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size

        if prefill_tp_size == self._prefill_tp_size:
            return self.tp_num_need_pulls

        if self.vllm_config.model_config.is_deepseek_mla:
            tp_num_need_pulls = 1
        else:
            num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
            num_p_block_heads = max(1, self.num_key_value_heads // prefill_tp_size)
            tp_num_need_pulls = num_d_block_heads // num_p_block_heads
        return tp_num_need_pulls

    def _get_remote_host_info_by_port(
        self,
        base_port: int,
        remote_handshake_port: int,
        remote_host: str,
        remote_engine_id: str,
        remote_multi_nodes_meta_mapping: dict,
    ):
        rank = str(remote_handshake_port - base_port)
        if remote_multi_nodes_meta_mapping is None or remote_multi_nodes_meta_mapping.get(rank) is None:
            return remote_host, remote_engine_id
        info = remote_multi_nodes_meta_mapping[rank]
        return info.get("host", remote_host), info.get("engine_id", remote_engine_id)

    def _prefill_get_remote_rank(self, req_id: str) -> list[int]:
        return sum(self._get_remote_ranks_for_req(req_id), [])

    def _get_remote_rank(self, req_id: str, prefill_tp_size: int | None = None) -> list[int]:
        return self._get_remote_ranks_for_req(req_id, prefill_tp_size)[self.tp_rank]

    def _get_remote_tp_ranks(
        self, tp_ori_data: np.ndarray, rand_group_index: list[int], num_groups: int, prefill_tp_size: int
    ) -> list[list[int]]:
        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        # random split prefill tp list
        tp_sampled_nums = []
        if (
            prefill_tp_size > self.num_key_value_heads
            or self.vllm_config.model_config.is_deepseek_mla
            or self.use_sparse
        ):
            tp_ori_data = tp_ori_data.reshape(-1, num_groups)
            chosen_group = tp_ori_data[:, [rand_group_index]]
            flattened = chosen_group.reshape(-1).tolist()
            tp_sampled_nums = [
                flattened[i : i + tp_num_need_pulls] for i in range(0, len(flattened), tp_num_need_pulls)
            ]
        # non-random split
        else:
            group_size = prefill_tp_size // self._decode_tp_size
            for i in range(self._decode_tp_size):
                slice = tp_ori_data[i * group_size : (i + 1) * group_size]
                tp_sampled_nums.append(slice.tolist())
        return tp_sampled_nums

    def _get_remote_ranks_for_req(self, req_id: str, prefill_tp_size: int | None = None) -> list[list[int]]:
        if prefill_tp_size is None:
            prefill_tp_size = self._prefill_tp_size

        # Divide the ports according to the TP within the PP
        sampled_nums = []
        if prefill_tp_size == self._decode_tp_size:
            sampled_nums = list(
                map(
                    lambda tp: [tp + pp * prefill_tp_size for pp in range(self._prefill_pp_size)],
                    range(prefill_tp_size),
                )
            )
            return sampled_nums
        # use deepseek mla, num_key_value_heads == 128, but consider as 1
        if self.vllm_config.model_config.is_deepseek_mla or self.use_sparse:
            num_kv_head = 1
        else:
            num_kv_head = self.num_key_value_heads
        ori_data = np.arange(prefill_tp_size * self._prefill_pp_size)
        seed = string_to_int64_hash(req_id)
        rand = random.Random(seed)
        # random split prefill tp list
        ori_data_2d = ori_data.reshape(self._prefill_pp_size, -1)
        num_groups = max(
            1, len(ori_data_2d[0]) // num_kv_head
        )  # The number of redundant copies for each KV head within the PP stage
        rand_group_index = rand.sample(
            range(num_groups), (max(self._decode_tp_size // num_kv_head, 1))
        )  # random choose a group
        all_results = [
            self._get_remote_tp_ranks(ori_data_2d[pp_index], rand_group_index, num_groups, prefill_tp_size)
            for pp_index in range(self._prefill_pp_size)
        ]
        for group_index in range(len(all_results[0])):
            group = []
            for pp_index in range(self._prefill_pp_size):
                group.extend(all_results[pp_index][group_index])
            sampled_nums.append(group)
        return sampled_nums


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:  # type: ignore
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ, zmq.DEALER):  # type: ignore
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None  # type: ignore
    try:
        ctx = zmq.Context()  # type: ignore
        yield make_zmq_socket(ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER)  # type: ignore
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def group_concurrent_contiguous(
    src: list[int],
    dst: list[int],
    src_block_stride: int = 1,
    dst_block_stride: int = 1,
    block_len: int = 1,
) -> tuple[list[list[int]], list[list[int]]]:
    """Group block ids that are contiguous in both id space and memory."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

    if src_indices.size == 0:
        return [], []

    src_byte_contiguous = np.diff(src_indices) * src_block_stride == block_len
    dst_byte_contiguous = np.diff(dst_indices) * dst_block_stride == block_len
    brk = np.where(~(src_byte_contiguous & dst_byte_contiguous))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def split_if_not_byte_contiguous(
    src_groups: list[list[int]],
    dst_groups: list[list[int]],
    src_block_stride: int,
    dst_block_stride: int,
    block_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    if src_block_stride == block_len and dst_block_stride == block_len:
        return src_groups, dst_groups

    src = [bid for group in src_groups for bid in group]
    dst = [bid for group in dst_groups for bid in group]
    return group_concurrent_contiguous(
        src,
        dst,
        src_block_stride=src_block_stride,
        dst_block_stride=dst_block_stride,
        block_len=block_len,
    )


def string_to_int64_hash(input_str):
    """
    Hash the string using SHA-256 and convert it into an int64 integer.
    """
    hashed_bytes = hashlib.sha256(input_str.encode("utf-8")).digest()
    trunked_bytes = hashed_bytes[:8]
    uint64_value = struct.unpack("<Q", trunked_bytes)[0]
    return uint64_value


def ensure_zmq_send(
    socket: zmq.Socket,  # type: ignore
    data: bytes,
    path: str,
    max_retries: int = 3,
):
    retries_left = max_retries
    while True:
        try:
            socket.send(data)
            return
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Send failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Send failed after all retries. error=%s. ", e)
                raise RuntimeError(f"Failed to send data to {path} after {max_retries} retries: {e}")


def ensure_zmq_recv(
    socket: zmq.Socket,  # type: ignore
    poller: zmq.Poller,  # type: ignore
    path: str,
    timeout: float = 1.0,
    max_retries: int = 3,
) -> bytes:
    retries_left = max_retries
    while True:
        try:
            if dict(poller.poll(int(timeout * 1000))):  # milliseconds
                data = socket.recv()
                return data
            else:
                raise zmq.ZMQError("Receive timeout")  # type: ignore
        except zmq.ZMQError as e:  # type: ignore
            retries_left -= 1
            if retries_left > 0:
                logger.warning("Receive failed. error=%s, attempts_left=%d. ", e, retries_left)
                time.sleep(0.1)
            else:
                logger.error("Receive failed after all retries. source=%s, error=%s. ", path, e)
                raise RuntimeError(f"Failed to receive data after {max_retries} retries: {e}")


# decode node should know pp_partition_layer in prefill node,
# it is configured in kv_transfer_config by partition_list_str,
# default using vllm layer split algorithm.
def get_prefill_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int, partition_list_str: str | None = None
) -> tuple[int, int]:
    if partition_list_str is None:
        return get_pp_indices(num_hidden_layers, pp_rank, pp_size)
    else:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError("Invalid partition string: {}".format(partition_list_str)) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
        return (start_layer, end_layer)


def get_parallel_device_index(
    tp_rank: int,
    tp_size: int,
    pp_rank: int = 0,
    pcp_rank: int = 0,
) -> int:
    return (pp_rank + pcp_rank) * tp_size + tp_rank


def get_dp_port_offset(
    dp_rank: int,
    tp_size: int,
    pp_size: int,
    pcp_size: int,
) -> int:
    return dp_rank * tp_size * pp_size * pcp_size


def get_d2rh_zmq_port(
    vllm_config: VllmConfig,
    tp_rank: int,
    pp_rank: int = 0,
    pcp_rank: int = 0,
) -> int:
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device_index = get_parallel_device_index(tp_rank, tp_size, pp_rank, pcp_rank)
    dp_offset = get_dp_port_offset(dp_rank, tp_size, pp_size, pcp_size)
    return D2RH_ZMQ_PORT_BASE + dp_offset + device_index


def get_scheduler_ready_zmq_port(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pp_size = parallel_config.pipeline_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    dp_offset = get_dp_port_offset(dp_rank, tp_size, pp_size, pcp_size)
    return SCHEDULER_READY_ZMQ_PORT_BASE + dp_offset


def resolve_remote_host_for_handshake_port(
    base_port: int,
    remote_handshake_port: int,
    remote_host: str,
    remote_engine_id: str,
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]] | None,
) -> tuple[str, str]:
    rank = str(remote_handshake_port - base_port)
    if remote_multi_nodes_meta_mapping is None or remote_multi_nodes_meta_mapping.get(rank) is None:
        return remote_host, remote_engine_id
    info = remote_multi_nodes_meta_mapping[rank]
    return info.get("host", remote_host), info.get("engine_id", remote_engine_id)


def compute_tp_num_need_pulls(
    num_key_value_heads: int,
    decode_tp_size: int,
    prefill_tp_size: int,
    is_deepseek_mla: bool,
) -> int:
    if is_deepseek_mla:
        return 1
    num_d_block_heads = max(1, num_key_value_heads // decode_tp_size)
    num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
    return num_d_block_heads // num_p_block_heads


def get_remote_tp_ranks_for_req(
    tp_ori_data: np.ndarray,
    rand_group_index: list[int],
    num_groups: int,
    prefill_tp_size: int,
    decode_tp_size: int,
    num_key_value_heads: int,
    is_deepseek_mla: bool,
    use_sparse: bool,
) -> list[list[int]]:
    tp_num_need_pulls = compute_tp_num_need_pulls(
        num_key_value_heads, decode_tp_size, prefill_tp_size, is_deepseek_mla
    )
    if prefill_tp_size > num_key_value_heads or is_deepseek_mla or use_sparse:
        tp_ori_data = tp_ori_data.reshape(-1, num_groups)
        chosen_group = tp_ori_data[:, rand_group_index]
        flattened = chosen_group.reshape(-1).tolist()
        return [flattened[i : i + tp_num_need_pulls] for i in range(0, len(flattened), tp_num_need_pulls)]

    group_size = prefill_tp_size // decode_tp_size
    tp_sampled_nums: list[list[int]] = []
    for i in range(decode_tp_size):
        slice_data = tp_ori_data[i * group_size : (i + 1) * group_size]
        tp_sampled_nums.append(slice_data.tolist())
    return tp_sampled_nums


def get_remote_ranks_for_req(
    req_id: str,
    prefill_tp_size: int,
    decode_tp_size: int,
    prefill_pp_size: int,
    num_key_value_heads: int,
    is_deepseek_mla: bool,
    use_sparse: bool,
) -> list[list[int]]:
    if prefill_tp_size == decode_tp_size:
        return [[tp + pp * prefill_tp_size for pp in range(prefill_pp_size)] for tp in range(prefill_tp_size)]

    num_kv_head = 1 if is_deepseek_mla or use_sparse else num_key_value_heads
    ori_data = np.arange(prefill_tp_size * prefill_pp_size)
    rand = random.Random(string_to_int64_hash(req_id))
    ori_data_2d = ori_data.reshape(prefill_pp_size, -1)
    num_groups = max(1, len(ori_data_2d[0]) // num_kv_head)
    rand_group_index = rand.sample(range(num_groups), max(decode_tp_size // num_kv_head, 1))
    all_results = [
        get_remote_tp_ranks_for_req(
            ori_data_2d[pp_index],
            rand_group_index,
            num_groups,
            prefill_tp_size,
            decode_tp_size,
            num_key_value_heads,
            is_deepseek_mla,
            use_sparse,
        )
        for pp_index in range(prefill_pp_size)
    ]
    sampled_nums: list[list[int]] = []
    for group_index in range(len(all_results[0])):
        group: list[int] = []
        for pp_index in range(prefill_pp_size):
            group.extend(all_results[pp_index][group_index])
        sampled_nums.append(group)
    return sampled_nums
