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
                        if self.ready_count[request_id] >= self.decode_tp_size:
                            self.ready_request.add(request_id)
                            del self.ready_count[request_id]
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
                        else:
                            remote_request_id = params.get("remote_request_id", request_id)
                            self.staging_allocations[request_id] = staging
                            if remote_request_id != request_id:
                                self.staging_allocations[remote_request_id] = staging
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
            self._send_scheduler_ready(request_id)

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
        while time.time() < deadline:
            staging = self.staging_allocations.get(remote_request_id) or self.staging_allocations.get(request_id)
            if staging is not None:
                return staging
            time.sleep(0.01)
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

