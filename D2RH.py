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
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
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
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from vllm_ascend import envs as ascend_envs
from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.distributed.kv_transfer.utils.utils import get_transfer_timeout_value
from vllm_ascend.utils import enable_custom_op, is_vl_model

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
START_PULL = b"START_PULL"
BLOCK_DICT = {}


class RemotePortInfo(TypedDict):
    num: int
    host: str


class MooncakeAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    engine_id: str
    te_rpc_port: int
    kv_caches_base_addr: list[int]
    num_blocks: int
    local_ip: str = ""


@dataclass
class ReqMeta:
    local_block_ids: list[int]
    num_external_tokens: int
    remote_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_engine_id: str
    remote_request_id: str
    remote_pcp_size: int
    remote_dcp_size: int
    remote_ptp_size: int | None
    remote_multi_nodes_meta_mapping: dict[str, dict[str, Any]]
    num_prompt_blocks: int


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
                logger.error(
                    "MooncakeConnector finish req not in reqs to process."
                    "If it is a P node, this request may have been force freed."
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
                logger.info("Force freed request: %s", request_id)
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
            logger.info("Starting listening on path: %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.ready_event.set()
                self.run_busy_loop(sock)
        except Exception as e:
            logger.error("Mooncake KVCacheSendingThread exception: %s", e, exc_info=True)

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        encoder = msgspec.msgpack.Encoder()
        encoded_data = encoder.encode(self.metadata)
        size_in_bytes = len(encoded_data)
        logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message format: %s", frames)
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid message format: %s", frames)
                    continue

                msg = decoder.decode(payload[0])
                if msg[0] == GET_META_MSG:
                    sock.send_multipart((identity, b"", encoded_data))
                    logger.info(f"get meta_message")
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
                    logger.error("Connection listener got unexpected message %s", msg)
            except Exception as e:
                logger.error("Connection listener got exception %s: %s", type(e), e)

class cpu_kvcache_manager():
    def __init__(
            self,
            num_blocks
    ):
        self.free_queue = deque(range(num_blocks))
        self.used_set = set()

    def alloc_blocks(self, req_need_blocks):
        now_block = len(self.free_queue)
        if req_need_blocks > len(self.free_queue):
            logger.info(f"[===] Dont have enough blocks {req_need_blocks=} {now_block=}")
            return None
        else:
            local_block_ids = [self.free_queue.popleft() for _ in range(req_need_blocks)]
            self.used_set.update(local_block_ids)
            return local_block_ids

    def free_blocks(self, req_block_ids):
        logger.info(f"[===] need free blocks{req_block_ids=}")
        flat_block_ids = [block_id for sublist in req_block_ids for block_id in sublist]
        self.free_queue.extend(flat_block_ids)
        self.used_set.difference_update(flat_block_ids)



class HostListeningThread(threading.Thread):
    def __init__(
        self,
    ):
        super().__init__(daemon=True, name="KVCacheSendingThread")
        self.port_send_num: dict[str, int] = {}

        self.task_tracker = KVCacheTaskTracker()
        self.ready_request = set()
        self.ready_lock = threading.Lock()
        self.host_ip = get_ip()
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
    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()


    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            # Listen for new requests for metadata. NOTE(rob): we need each rank
            # to have a unique port. This hack to keeps us moving. We will
            # switch when moving to etcd or where we have a single ZMQ socket in
            # the scheduler.
            handshake_port = 8200
            path = make_zmq_path("tcp", self.host_ip, handshake_port)
            logger.info("Starting listening on path: %s", path)
            with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                self.run_busy_loop(sock)
        except Exception as e:
            logger.error("Mooncake KVCacheSendingThread exception: %s", e, exc_info=True)

    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        # encoder = msgspec.msgpack.Encoder()
        # encoded_data = encoder.encode(self.metadata)
        # size_in_bytes = len(encoded_data)
        # logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message format: %s", frames)
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid message format: %s", frames)
                    continue

                msg = decoder.decode(payload[0])
                if msg[0] == READY_SCHEDULER:
                    logger.debug("Got READY_SCHEDULER for request %s", msg[1])
                    request_id = msg[1]
                    with self.ready_lock:
                        self.ready_request.add(request_id)
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
                    logger.error("Connection listener got unexpected message %s", msg)
            except Exception as e:
                logger.error("Connection listener got exception %s: %s", type(e), e)



class D2RHThread(threading.Thread):
    def __init__(
            self,
            ptrs,
            engine,
            cpu_kvcache_manager
            ):
        super().__init__(daemon=True, name="D2RHThread")
        self.kv_caches_base_addr: dict[str, dict[int, list[int]]] = SizedDict()
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.request_queue: queue.Queue[Any] = queue.Queue()
        self.remote_sockets_lock = threading.Lock()
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.timeout = 1.0
        self.remote_poller = zmq.Poller()
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.ptrs = ptrs
        self.engine = engine
        self.host_ip = get_ip()
        self.cpu_kvcache_manager = cpu_kvcache_manager
    
    def add_request(
        self,
        request_id,
        remote_host,
        remote_port,
        remote_engine_id,
        grouped_remote_block_ids,
        grouped_local_block_ids
    ):
        self.request_queue.put(
            {
                "request_id":request_id,
                "remote_host":remote_host,
                "remote_port":remote_port,
                "remote_engine_id":remote_engine_id,
                "grouped_remote_block_ids":grouped_remote_block_ids,
                "grouped_local_block_ids":grouped_local_block_ids
            }
        )

    # def run(self):
    #     """Run the thread to handle KV cache transfer requests."""
    #     try:
    #         # Listen for new requests for metadata. NOTE(rob): we need each rank
    #         # to have a unique port. This hack to keeps us moving. We will
    #         # switch when moving to etcd or where we have a single ZMQ socket in
    #         # the scheduler.
    #         handshake_port = 8100
    #         path = make_zmq_path("tcp", self.host_ip, handshake_port)
    #         logger.info("Starting listening on path: %s", path)
    #         with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
    #             self.run_busy_loop(sock)
    #     except Exception as e:
    #         logger.error("Mooncake KVCacheSendingThread exception: %s", e, exc_info=True)
    #     while True:
    #         # try:
    #         request_data = self.request_queue.get()
    #         if request_data is None:
    #             logger.warning("Received a None request!")
    #             self.request_queue.task_done()
    #             continue
    #         self._handle_request(request_data)
    #         # except Exception as e:
    #         #     logger.error(f"Error in KVCacheTransferThread: {e}")

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        try:
            handshake_port = 8100
            path = make_zmq_path("tcp", self.host_ip, handshake_port)
            logger.info("Starting listening on path: %s", path)
            
            # 1. 🚀【核心修复】把 ZMQ 生产者死循环剥离到一个完全独立的后台守护线程中运行
            def zmq_listener_worker():
                try:
                    with zmq_ctx(zmq.ROUTER, path) as sock:  # type: ignore
                        # 这个死循环现在单独占用一个子线程，再也不会拖累下面的队列消费了
                        self.run_busy_loop(sock)
                except Exception as e:
                    logger.error("ZMQ background listener worker crashed: %s", e, exc_info=True)

            bg_zmq_thread = threading.Thread(target=zmq_listener_worker, name="D2RH-ZMQListener", daemon=True)
            bg_zmq_thread.start()
            logger.info("ZMQ background listener thread started successfully.")

        except Exception as e:
            logger.error("Failed to initialize ZMQ path in D2RHThread: %s", e, exc_info=True)
            return  # 初始化失败则无法工作，直接退出

        # 2. 🛒【消费者循环】当前主线程留在这里，无干扰、全速地“一直 get 队列”
        logger.info("D2RHThread queue consumer loop started.")
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    # 遵循你的意图：即使收到空信号也绝对不 break，记录警告并继续坚守岗位
                    logger.warning("Received an unexpected None request in worker loop! Skipping...")
                    continue
                
                # 消费队列任务
                self._handle_request(request_data)
                
            except Exception as e:
                # 💡【异常防护隔离】通过把 try-except 放在 while 循环内部，
                # 即使某一笔 KV Cache 传输抛出任何不可预知的业务异常（例如底层 RPC 报错、KeyError 等），
                # 异常被捕获记录后，while True 绝不会中断，线程会立刻进入下一次 `get()`，保障服务高可用。
                logger.error("Error occurred while processing request in D2RHThread loop: %s", e, exc_info=True)
            finally:
                self.request_queue.task_done()
    
    def run_busy_loop(self, sock: zmq.Socket):  # type: ignore
        # encoder = msgspec.msgpack.Encoder()
        # encoded_data = encoder.encode(self.metadata)
        # size_in_bytes = len(encoded_data)
        # logger.debug("Size of encoded MooncakeAgentMetadata: %s bytes", str(size_in_bytes))

        decoder = msgspec.msgpack.Decoder(type=tuple)
        while True:
            try:
                frames = sock.recv_multipart()
                if len(frames) < 2:
                    logger.error("Invalid message format: %s", frames)
                    continue

                identity = frames[0]
                payload = [f for f in frames[1:] if f != b""]
                if len(payload) != 1:
                    logger.error("Invalid message format: %s", frames)
                    continue

                msg = decoder.decode(payload[0])
                logger.info(f"[===] we get msg[0] {msg[0]=}")
                if msg[0] == START_PULL:
                    logger.info("Got START_PULL for request %s", msg[1])
                    request_id = msg[1]
                    params = msg[2]
                    remote_host = params['remote_host']
                    remote_port = params['remote_port']
                    remote_engine_id = params['remote_engine_id']
                    remote_block_ids = params['remote_block_ids']
                    grouped_remote_block_ids = remote_block_ids
                    grouped_local_block_ids = self.cpu_kvcache_manager.alloc_blocks(len(remote_block_ids))
                    global BLOCK_DICT
                    BLOCK_DICT[request_id] = grouped_local_block_ids
                    logger.info(f"[===] we after alloc blocks{grouped_local_block_ids=}{grouped_remote_block_ids=}")
                    if grouped_local_block_ids is not None:
                        self.add_request(
                                    request_id = request_id,
                                    remote_host = remote_host,
                                    remote_port = remote_port,
                                    remote_engine_id = remote_engine_id,
                                    grouped_remote_block_ids = grouped_remote_block_ids,
                                    grouped_local_block_ids = grouped_local_block_ids)

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
                    logger.error("Connection listener got unexpected message %s", msg)
            except Exception as e:
                logger.error("Connection listener got exception %s: %s", type(e), e)

    def _handle_request(self, req_meta: dict[str, Any]):
        remote_host = req_meta["remote_host"]
        remote_port = req_meta["remote_port"]
        remote_engine_id = req_meta["remote_engine_id"]
        self._transfer_kv_cache(req_meta)

    def _transfer_kv_cache(self, req_meta: dict[str, Any]):
        request_id =  req_meta["request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_port"]
        remote_engine_id = req_meta["remote_engine_id"]
        grouped_remote_block_ids = [req_meta["grouped_remote_block_ids"]]
        grouped_local_block_ids = [req_meta["grouped_local_block_ids"]]
        

        if (
            remote_engine_id not in self.kv_caches_base_addr
            or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]
        ):
            self._get_remote_metadata(remote_host, remote_handshake_port)
        remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
        te_rpc_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
        session_id = f"{remote_host}:{te_rpc_port}"
        inner_block_len = 262144
        block_len = 262144
        src_list, dst_list, length_list = [], [], []
        logger.info(f"[===] P side we need to pull is {remote_kv_caches_base_addrs=}")
        for src_layer_base_addr in remote_kv_caches_base_addrs:
            for remote_block_id, local_block_id in zip(grouped_remote_block_ids, grouped_local_block_ids):
                src = src_layer_base_addr + local_block_id[0] * block_len
                length = inner_block_len * len(local_block_id)
                src_list.append(src)
                length_list.append(length)
        logger.info(f"[===] D2RH thread do transfer {self.ptrs=}{src_list=}{session_id=}{length_list=}{grouped_remote_block_ids=}{grouped_local_block_ids=}")
        ret = self.engine.batch_transfer_sync_read(session_id, self.ptrs, src_list, length_list)

        self.send_pull_done(request_id)
        if ret < 0:
            logger.error(f"trans_failed{session_id}")

    def send_pull_done(self,request_id):
        sock: zmq.Socket | None = None  # type: ignore
        try:
            port = 8200
            sock = self._get_remote_socket(self.host_ip, port)
            data_bytes = self.encoder.encode((READY_SCHEDULER, request_id))
            ensure_zmq_send(sock, data_bytes, f"{self.host_ip}:{port}")
            resp = ensure_zmq_recv(
                sock, self.remote_poller, f"{self.host_ip}:{port}", timeout=self.timeout
            )
            logger.debug(f"Received response for request {request_id}: {resp.decode('utf-8')}")
            if resp != b"ACK":
                logger.error(
                    "Failed to receive ACK for request %s from %s:%d", request_id, self.host_ip, port
                )
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning(f"Unexpected error occurred in socket, {e}, closing the original channel")
        finally:
            if sock is not None:
                self._return_remote_socket(sock, self.host_ip, port)
                logger.debug("Returned socket to pool for %s:%d", self.host_ip, port)

    def _get_remote_metadata(self, remote_host: str, remote_handshake_port: int) -> None:
        """Get the metadata from the remote host."""
        sock: zmq.Socket | None = None  # type: ignore
        try:
            sock = self._get_remote_socket(remote_host, remote_handshake_port)
            ensure_zmq_send(sock, self.encoder.encode((GET_META_MSG, "")), f"{remote_host}:{remote_handshake_port}")
            metadata_bytes = ensure_zmq_recv(sock, self.remote_poller, f"{remote_host}:{remote_handshake_port}")
            agent_meta = self.decoder.decode(metadata_bytes)
            engine_id = agent_meta.engine_id
            # assert engine_id != self.local_engine_id, (
            #     f"Conflict engine id {engine_id} with local engine id {self.local_engine_id}."
            # )
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
            logger.info(f"[===] base_add te_rpc_port {agent_meta.kv_caches_base_addr} {agent_meta.te_rpc_port}")
            logger.info(f"[===] _get_remote_metadata {self.kv_caches_base_addr} {self.remote_te_port}")
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
        local_kv_caches_base_addr: list[int],
        block_len: list[int],
        ready_event: threading.Event,
        vllm_config: VllmConfig,
        kv_caches: dict[str, Any],
        te_rpc_port,
        cpu_ptr,
        cpu_kvcache_manager,
        prefill_pp_layer_partition: str | None = None,
    ):
        super().__init__(daemon=True, name="KVCacheRecvingThread")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._prefill_pp_size = _prefill_pp_size
        self.local_engine_id = local_engine_id
        self.local_handshake_port = local_handshake_port
        self.side_channel_port = side_channel_port
        self.engine = engine
        self.ready_event = ready_event
        self.te_rpc_port = te_rpc_port
        self.cpu_ptrs = cpu_ptr
        self.cpu_kvcache_manager = cpu_kvcache_manager

        self.kv_caches = kv_caches
        self.kv_caches_base_addr: dict[str, dict[int, list[int]]] = SizedDict()
        self.kv_caches_base_addr[local_engine_id][local_handshake_port] = local_kv_caches_base_addr
        self.remote_te_port: dict[str, dict[int, int]] = SizedDict()
        self.block_len = block_len
        # TODO(jianzs): find a better way to detect MLA.
        self.use_mla = len(block_len) == 2

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

        self.vllm_config = vllm_config
        self.model_config = self.vllm_config.model_config
        self.block_size = self.vllm_config.cache_config.block_size
        self.num_layers = self.model_config.hf_text_config.num_hidden_layers
        self.pp_layer_indices = {
            rank: get_prefill_pp_indices(self.num_layers, rank, self._prefill_pp_size, prefill_pp_layer_partition)
            for rank in range(self._prefill_pp_size)
        }
        if not is_vl_model(vllm_config):
            if self.use_mla:
                self.k_head_dim = self.model_config.hf_text_config.kv_lora_rank
                self.v_head_dim = self.model_config.hf_text_config.qk_rope_head_dim
                self.num_kv_heads = 1
            else:
                self.k_head_dim = self.model_config.hf_text_config.head_dim
                self.v_head_dim = self.model_config.hf_text_config.head_dim
                self.num_kv_heads = max(self.model_config.hf_text_config.num_key_value_heads // self.tp_size, 1)
        self.proc_not_transfer_request: dict[str, bool] = {}

    def add_request(
        self,
        request_id: str,
        remote_request_id: str,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_engine_id: str,
        remote_host: str,
        remote_handshake_port: int,
        offset: int,
        tp_num_need_pulls: int,
        remote_port_send_num: dict[int, RemotePortInfo] | None = None,
        all_task_done: bool = False,
    ):
        """Add a new request to the queue for processing."""
        if remote_port_send_num is None:
            remote_port_send_num = {}
        logger.debug(f"Adding request {request_id} to the queue.")
        self.request_queue.put(
            {
                "request_id": request_id,
                "local_block_ids": local_block_ids,
                "remote_block_ids": remote_block_ids,
                "remote_engine_id": remote_engine_id,
                "remote_request_id": remote_request_id,
                "remote_host": remote_host,
                "remote_handshake_port": remote_handshake_port,
                "offset": offset,
                "tp_num_need_pulls": tp_num_need_pulls,
                "remote_port_send_num": remote_port_send_num,
                "all_task_done": all_task_done,
            }
        )

    def get_and_clear_finished_requests(self) -> set[str]:
        """
        Get and clear the requests that have been completed.
        Returns:
            A set of request IDs that have been completed.
        """
        return self.task_tracker.get_and_clear_finished_requests()

    def run(self):
        """Run the thread to handle KV cache transfer requests."""
        self.ready_event.set()
        while True:
            try:
                request_data = self.request_queue.get()
                if request_data is None:
                    logger.warning("Received a None request!")
                    self.request_queue.task_done()
                    continue
                self._handle_request(request_data)
            except Exception as e:
                logger.error(f"Error in KVCacheTransferThread: {e}")

    def _handle_request(self, req_meta: dict[str, Any]):
        request_id = req_meta["request_id"]
        remote_request_id = req_meta["remote_request_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        remote_port_send_num = req_meta["remote_port_send_num"]
        all_task_done = req_meta["all_task_done"]

        try:
            logger.debug(f"Starting to transfer KV cache for request {remote_request_id}.")
            self._transfer_kv_cache(req_meta)
            logger.debug(f"Finished transferring KV cache for request {remote_request_id}.")
        except Exception as e:
            logger.error(f"Failed to transfer KV cache for request {remote_request_id}: {e}", exc_info=True)
        finally:
            self._send_done_signal_to_free_remote_port(remote_request_id, remote_host, remote_port_send_num)
            if all_task_done:
                self.task_tracker.update_done_task_count(request_id)
                if request_id in self.proc_not_transfer_request:
                    del self.proc_not_transfer_request[request_id]
            self.request_queue.task_done()
            # Always send the done signal to the remote host to ensure proper
            # resource cleanup. Failing to do so may cause a memory leak on the
            # remote host.
            self._send_done_recv_signal(remote_request_id, remote_host, remote_handshake_port, remote_port_send_num)

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

    def _transfer_kv_cache(self, req_meta: dict[str, Any]):
        """Handle a KV cache transfer request."""
        remote_request_id = req_meta["remote_request_id"]
        remote_block_ids = req_meta["remote_block_ids"]
        local_block_ids = req_meta["local_block_ids"]
        remote_engine_id = req_meta["remote_engine_id"]
        remote_host = req_meta["remote_host"]
        remote_handshake_port = req_meta["remote_handshake_port"]
        offset = req_meta["offset"]
        tp_num_need_pulls = req_meta["tp_num_need_pulls"]

        # Full prefix cache hit: do not need to read remote blocks, just notify
        # P worker that we have the blocks we need.
        num_local_blocks = len(local_block_ids)
        if num_local_blocks == 0:
            return

        num_remote_blocks = len(remote_block_ids)
        assert num_local_blocks <= num_remote_blocks
        if num_local_blocks < num_remote_blocks:
            remote_block_ids = remote_block_ids[-num_local_blocks:]

        # Check if we have the remote metadata cached.
        if (
            remote_engine_id not in self.kv_caches_base_addr
            or remote_handshake_port not in self.kv_caches_base_addr[remote_engine_id]
        ):
            self._get_remote_metadata(remote_host, remote_handshake_port)

        if tp_num_need_pulls == 1:
            grouped_remote_block_ids, grouped_local_block_ids = group_concurrent_contiguous(
                remote_block_ids, local_block_ids
            )
            logger.info(f"[===]{grouped_remote_block_ids=}{grouped_local_block_ids=}{remote_block_ids=}{local_block_ids=}")
        else:
            remote_block_ids = list(map(lambda x: [x], remote_block_ids))
            local_block_ids = list(map(lambda x: [x], local_block_ids))
            grouped_remote_block_ids, grouped_local_block_ids = remote_block_ids, local_block_ids
        num_transfer_groups = len(grouped_remote_block_ids)
        # tp_num_need_pulls: number of KV caches each Decode node needs to pull from each PP stage
        # Due to GQA, different KV heads are distributed across different ranks, so there are offsets
        # indicating which KV head to pull
        global_offset = offset  # Global offset of request across all ranks
        prefill_pp_rank = offset // tp_num_need_pulls  # PP rank where current request resides
        inner_offset = offset % tp_num_need_pulls  # Offset within each PP stage
        logger.info(f"[===] import thing {inner_offset=}")

        remote_kv_caches_base_addrs = self.kv_caches_base_addr[remote_engine_id][remote_handshake_port]
        first_layer_index, end_layer_index = self.pp_layer_indices[prefill_pp_rank]
        # support MTP layer kv transfer
        if self.vllm_config.speculative_config is not None:
            # all MTP layer use the same kv cache layer, so only need to transfer once
            if prefill_pp_rank == self._prefill_pp_size - 1:
                end_layer_index = end_layer_index + 1
        num_cache_per_layer = len(list(self.kv_caches.values())[0])  # Number of KV caches per layer
        local_kv_caches_base_addrs = self.kv_caches_base_addr[self.local_engine_id][self.local_handshake_port][
            first_layer_index * num_cache_per_layer : end_layer_index * num_cache_per_layer
        ]
        logger.debug(f"transfer kv cache first_layer_index:{first_layer_index} , end_layer_index:{end_layer_index}")
        remote_transfer_port = self.remote_te_port[remote_engine_id][remote_handshake_port]
        num_blocks = len(local_block_ids)
        session_id = f"{remote_host}:{remote_transfer_port}"
        local_host = get_ip()
        session_id = f"{local_host}:{self.te_rpc_port}"


        req_start_time = time.perf_counter()
        src_list, dst_list, length_list = [], [], []
        block_length = len(self.block_len)
        for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(
            zip(local_kv_caches_base_addrs, remote_kv_caches_base_addrs)
        ):
            block_len = self.block_len[k % block_length]
            inner_block_len = block_len // tp_num_need_pulls
            for remote_block_id, local_block_id in zip(grouped_remote_block_ids, grouped_local_block_ids):
                src = src_layer_base_addr + local_block_id[0] * block_len + inner_offset * inner_block_len
                dst = dst_layer_base_addr + remote_block_id[0] * inner_block_len
                length = inner_block_len * len(local_block_id)
                src_list.append(src)
                dst_list.append(dst)
                length_list.append(length)
        logger.info(f"[===] Receive thread {session_id=}{src_list=}{dst_list=}{length_list=}{grouped_local_block_ids=}{grouped_remote_block_ids}")
        ret = self.engine.batch_transfer_sync_read(session_id, self.cpu_ptrs, dst_list, length_list)
        self.cpu_kvcache_manager.free_blocks(grouped_local_block_ids)
        if ret < 0:
            logger.error("Mooncake transfer failed for request %s", req_meta["remote_request_id"])
            raise RuntimeError(f"Mooncake transfer failed, ret: {ret}")

        req_end_time = time.perf_counter()
        req_transfer_elapsed = (req_end_time - req_start_time) * 1000
        logger.info(
            "KV cache transfer for request %s took %.2f ms (%d groups,"
            " %d blocks). local_ip %s local_device_id %s remote_session_id %s",
            remote_request_id,
            req_transfer_elapsed,
            num_transfer_groups,
            num_blocks,
            get_ip(),
            self.tp_rank,
            session_id,
        )

        # Determine if the current position is the offset position at the end of
        # the KV transmission.
        is_kv_transfer_end = global_offset == tp_num_need_pulls * self._prefill_pp_size - 1
        need_cat_cache = tp_num_need_pulls > 1 and is_kv_transfer_end
        need_nz_cache = get_ascend_config().enable_kv_nz and is_kv_transfer_end
        use_fused_op = ascend_envs.VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK
        if need_nz_cache or need_cat_cache:
            # use fused op to reformat kv cache, we keep original implementation to provide ability to disable it.
            if use_fused_op and enable_custom_op():
                if need_cat_cache:
                    # the fused op only support cat GQA/MHA kv cache by head
                    self.reformat_kv_cache_with_fused_op(grouped_local_block_ids, tp_num_need_pulls)
                if need_nz_cache:
                    # maybe use fused op to reformat kv nz too in the future.
                    self.reformat_kv_cache(grouped_local_block_ids, tp_num_need_pulls, False, need_nz_cache)
            else:
                self.reformat_kv_cache(grouped_local_block_ids, tp_num_need_pulls, need_cat_cache, need_nz_cache)

    def reformat_kv_cache_with_fused_op(self, block_ids: list[list[int]], tp_num_need_pulls: int):
        # Get necessary parameters
        k_cache = list(self.kv_caches.values())[0][0]
        device = k_cache.device
        head_dim = self.model_config.hf_text_config.head_dim
        block_size = self.vllm_config.cache_config.block_size
        num_kv_head = max(self.model_config.hf_text_config.num_key_value_heads // self.tp_size, 1)
        layers = len(self.kv_caches)
        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int64, device=device)

        k_caches = []
        v_caches = []
        for _, (k_cache_layer, v_cache_layer) in self.kv_caches.items():
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
    ):
        # Get necessary parameters
        k_cache = list(self.kv_caches.values())[0][0]
        dtype = k_cache.dtype
        device = k_cache.device

        flat_block_ids = [item for sublist in block_ids for item in sublist]
        block_ids_tensor = torch.tensor(flat_block_ids, dtype=torch.int32, device=device)
        num_blocks = len(flat_block_ids)
        num_tokens = num_blocks * self.block_size

        # Create device tensors for copy operations
        block_table = block_ids_tensor.view(1, -1)
        block_len_tensor = torch.tensor([num_tokens], dtype=torch.int32, device=device)
        seq_start_tensor = torch.tensor([0], dtype=torch.int32, device=device)

        # Initialize buffers
        k_buffer = torch.empty((num_tokens, self.num_kv_heads, self.k_head_dim), dtype=dtype, device=device)
        v_buffer = torch.empty((num_tokens, self.num_kv_heads, self.v_head_dim), dtype=dtype, device=device)

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
        for _, (k_cache_layer, v_cache_layer) in self.kv_caches.items():
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
                )
            if need_nz_cache:
                self._nz_kv_cache(k_cache_layer, v_cache_layer, k_buffer, v_buffer, slot_mapping)
        # Clean up buffers
        del k_buffer, v_buffer

    def _cat_kv_cache(
        self, k_cache_layer, v_cache_layer, k_buffer, v_buffer, tp_num_need_pulls, num_blocks, num_tokens, slot_mapping
    ):
        def _transpose_kv_cache_between_head(buffer: torch.Tensor) -> torch.Tensor:
            buffer = buffer.view(num_blocks, tp_num_need_pulls, self.block_size, -1)
            buffer.transpose_(1, 2)
            return buffer.contiguous().view(num_tokens, self.num_kv_heads, -1)

        # Transpose KV cache
        k_buffer = _transpose_kv_cache_between_head(k_buffer)
        v_buffer = _transpose_kv_cache_between_head(v_buffer)

        # Reshape and cache the processed buffers
        torch_npu._npu_reshape_and_cache(
            key=k_buffer, value=v_buffer, key_cache=k_cache_layer, value_cache=v_cache_layer, slot_indices=slot_mapping
        )

    def _nz_kv_cache(self, k_cache_layer, v_cache_layer, k_buffer, v_buffer, slot_mapping):
        nz_fmt_last_dim = 16
        k_cache_layer = k_cache_layer.view(
            -1, self.k_head_dim * self.num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
        )
        v_cache_layer = v_cache_layer.view(
            -1, self.v_head_dim * self.num_kv_heads // nz_fmt_last_dim, self.block_size, nz_fmt_last_dim
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
            self.kv_caches_base_addr[engine_id][remote_handshake_port] = agent_meta.kv_caches_base_addr
            self.remote_te_port[engine_id][remote_handshake_port] = agent_meta.te_rpc_port
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
            logger.debug(f"Received response for request {request_id}: {resp.decode('utf-8')}")
            if resp != b"ACK":
                logger.error(
                    "Failed to receive ACK for request %s from %s:%d", request_id, remote_host, remote_handshake_port
                )
                raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
        except RuntimeError as e:
            if isinstance(sock, zmq.Socket):  # type: ignore
                sock.close()
                sock = None
                logger.warning(f"Unexpected error occurred in socket, {e}, closing the original channel")
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
        local_block_ids: list[int],
        num_external_tokens: int,
        kv_transfer_params: dict[str, Any],
    ):
        self.requests[request_id] = ReqMeta(
            local_block_ids=local_block_ids,
            num_external_tokens=num_external_tokens,
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


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        assert vllm_config.kv_transfer_config is not None
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self._connector_metadata = MooncakeConnectorMetadata()

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = MooncakeConnectorScheduler(
                vllm_config, str(self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, str(self.engine_id))

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

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
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
        self._reqs_need_recv: dict[str, tuple[Request, list[int], int]] = {}
        self._reqs_need_send: dict[str, float] = {}
        self._reqs_in_batch: set[str] = set()

        self.kv_role = vllm_config.kv_transfer_config.kv_role
        if self.kv_role == "kv_consumer":
            self.listeningthread = HostListeningThread()
            self.listeningthread.start()
            self.all_requests = set()

        # master-slave meta information for cross-nodes
        self.multi_nodes_meta_mapping: dict[str, dict[str, Any]] = {}
        self.encoder = msgspec.msgpack.Encoder()
        self.decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)
        self.remote_sockets: dict[  # type: ignore
            str, deque[zmq.Socket]
        ] = defaultdict(  # type: ignore
            deque
        )
        self.remote_poller = zmq.Poller()  # type: ignore
        self.timeout = 1.0  # seconds
        self.remote_sockets_lock = threading.Lock()
        self.local_host = get_ip()
        self.local_port = 8100


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
            if request.request_id not in self.all_requests:
                sock: zmq.Socket | None = None  # type: ignore
                try:
                    sock = self._get_remote_socket(self.local_host, self.local_port)
                    data_bytes = self.encoder.encode((START_PULL, request.request_id, params))
                    ensure_zmq_send(sock, data_bytes, f"{self.local_host}:{self.local_port}")
                    resp = ensure_zmq_recv(
                        sock, self.remote_poller, f"{self.local_host}:{self.local_port}", timeout=self.timeout
                    )
                    logger.debug(f"Received response for request {request.request_id}: {resp.decode('utf-8')}")
                    if resp != b"ACK":
                        logger.error(
                            "Failed to receive ACK for request %s from %s:%d", request.request_id, self.local_host, self.local_port
                        )
                        raise RuntimeError(f"Failed to receive ACK, resp: {resp.decode('utf-8')}")
                except RuntimeError as e:
                    if isinstance(sock, zmq.Socket):  # type: ignore
                        sock.close()
                        sock = None
                        logger.warning(f"Unexpected error occurred in socket, {e}, closing the original channel")
                finally:
                    if sock is not None:
                        self._return_remote_socket(sock, self.local_host, self.local_port)
                        logger.debug("Returned socket to pool for %s:%d", self.local_host, self.local_port)
                self.all_requests.add(request.request_id)
            with self.listeningthread.ready_lock:
                if request.request_id in self.listeningthread.ready_request:
                    self.all_requests.remove(request.request_id)
                    assert num_computed_tokens % self.block_size == 0
                    # Note: We use the full token count as transmit data here.
                    count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
                    logger.info(f"[===]success scheduler")
                    return count, count > 0
                else:
                    return None, False

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
                    local_block_ids = blocks.get_unhashed_block_ids() if num_external_tokens > 0 else []
                    # Get unhashed blocks to pull from remote.
                    self._reqs_need_recv[request.request_id] = (request, local_block_ids, num_external_tokens)
                else:
                    logger.warning("Got invalid KVTransferParams: %s. This request will not utilize KVTransfer", params)
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
        block_ids: list[int],
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
        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", len(computed_block_ids), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)

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



