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

        num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)
        computed_block_ids = tuple(
            block_ids[:num_prompt_blocks] if not self._is_mamba_group[i] else block_ids
            for i, block_ids in enumerate(computed_block_ids)
        )

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
        self.total_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
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
        self._is_hma_required = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager and any(
            not isinstance(g.kv_cache_spec, FullAttentionSpec) for g in kv_cache_config.kv_cache_groups
        )
        self._layer_specs = {
            layer: group.kv_cache_spec for group in kv_cache_config.kv_cache_groups for layer in group.layer_names
        }
        self._layer_group_ids = {
            layer: group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            for layer in group.layer_names
        }
        self.hma_group_size = len(kv_cache_config.kv_cache_groups)

        # Mamba metadata
        self._is_mamba_group = [isinstance(group.kv_cache_spec, MambaSpec) for group in kv_cache_config.kv_cache_groups]
        print(f"[_layer_group_ids] {self._layer_group_ids}")
        print(f"[_is_mamba_group] {self._is_mamba_group}")
        mamba_ssm_size = (0, 0)
        mamba_spec = None
        self._has_mamba = any(self._is_mamba_group)
        if self._has_mamba:
            assert self._is_hma_required
            assert self.pcp_size * self.dcp_size == 1
            mamba_spec = next(spec for spec in self._layer_specs.values() if isinstance(spec, MambaSpec))
            conv_nbytes, ssm_nbytes = (
                torch.tensor([], dtype=mamba_spec.dtypes[0]).element_size(),  # type: ignore[misc]
                torch.tensor([], dtype=mamba_spec.dtypes[1]).element_size(),  # type: ignore[misc]
            )
            conv_shape, ssm_shape = (
                torch.Size(mamba_spec.shapes[0]),
                torch.Size(mamba_spec.shapes[1]),
            )
            mamba_ssm_size = (
                conv_shape.numel() * conv_nbytes,
                ssm_shape.numel() * ssm_nbytes,
            )
        self._mamba_ssm_size = mamba_ssm_size
        self._mamba_spec = mamba_spec

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
        self._prefill_pp_layer_partition = prefill_parallel_config.get("pp_layer_partition")

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data."""
        self.use_mla = self.vllm_config.model_config.is_deepseek_mla
        self.use_sparse = hasattr(self.vllm_config.model_config.hf_text_config, "index_topk")

        self.num_blocks = self.kv_cache_config.num_blocks
        logger.info("num_blocks: %s", self.num_blocks)
        self.block_len_per_addr = []
        self.kv_caches = kv_caches
        kv_caches_base_addr = []
        group_kv_cache_indices = [set() for _ in range(self.hma_group_size)]  # group_idx to kv_idx
        block_size_scale = [[] for _ in range(self.total_layers)]
        
        ptrs = []
        lengths = []
        if not self._is_hma_required:
            for layer_name, kv_cache_tuple in kv_caches.items():
                layer_spec = self._layer_specs[layer_name]
                if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                    layer_spec = layer_spec.kv_cache_specs[layer_name]
                if isinstance(kv_cache_tuple, (list, tuple)) is False:
                    kv_cache_tuple = [kv_cache_tuple]
                for single_kv_cache in kv_cache_tuple:
                    tensor_num_blocks = single_kv_cache.shape[0]
                    block_size_scale = tensor_num_blocks // self.num_blocks
                    block_shape = single_kv_cache.shape[1:]
                    self.block_len_per_addr.append(
                        single_kv_cache.element_size() * math.prod(block_shape) * block_size_scale
                    )
                    kv_caches_base_addr.append(single_kv_cache.data_ptr())
                    ptrs.append(single_kv_cache.data_ptr())
                    lengths.append(single_kv_cache.element_size() * math.prod(single_kv_cache.shape))
        elif self._has_mamba:
            kv_caches_base_addr = [[] for _ in range(self.total_layers)]
            kv_cache_shape = [[] for _ in range(self.total_layers)]
            self.block_size_scale = [[] for _ in range(self.total_layers)]
            self.block_len_per_addr = [[] for _ in range(self.total_layers)]
            conv_padding = 0
            for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
                share_tensor_addr = []
                has_mtp = False
                for layer_name in kv_cache_tensor.shared_by:
                    from vllm.v1.worker.utils import extract_layer_index
                    num_attn_module = 2 if self.vllm_config.model_config.hf_text_config.model_type == "longcat_flash" else 1
                    layer_idx = extract_layer_index(layer_name, num_attn_module)
                    
                    if "mtp" in layer_name:
                        has_mtp = True
                    layer_spec = self._layer_specs[layer_name]
                    if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                        layer_spec = layer_spec.kv_cache_specs[layer_name]
                    kv_cache_tuple = kv_caches[layer_name]
                    if isinstance(kv_cache_tuple, (list, tuple)) is False:
                        kv_cache_tuple = [kv_cache_tuple]
                    for single_kv_cache in kv_cache_tuple:
                        kv_cache_ptr = single_kv_cache.data_ptr()
                        tensor_num_blocks = single_kv_cache.shape[0]
                        block_shape = single_kv_cache.shape[1:]
                        self.block_len_per_addr[layer_idx].append(single_kv_cache.element_size() * math.prod(block_shape))
                        kv_caches_base_addr[layer_idx].append(kv_cache_ptr)
                        share_tensor_addr.append(kv_cache_ptr)
                        group_kv_cache_indices[self._layer_group_ids[layer_name]].add(layer_idx)
                        self.block_size_scale[layer_idx].append(tensor_num_blocks // self.num_blocks)
                        kv_cache_shape[layer_idx].append(single_kv_cache.shape)
                    if isinstance(layer_spec, MambaSpec) and len(self._mamba_ssm_size) == 2:
                        conv_padding = self.num_blocks * self._mamba_ssm_size[0]
                if share_tensor_addr:
                    if not has_mtp:
                        ptrs.append(min(share_tensor_addr))
                        lengths.append(kv_cache_tensor.size)
                    else:
                        ptrs.append(min(share_tensor_addr) - conv_padding)
                        lengths.append(kv_cache_tensor.size)

        else:
            raise TypeError("Mooncake connector does not support this type kv_cache now.")

        global_te.register_buffer(ptrs, lengths)
        # After KV Caches registered, start the sending or receiving thread.
        print(f"[loacl_kv_caches_base_addr] {kv_caches_base_addr=} {group_kv_cache_indices=} {kv_cache_shape=} {self.block_len_per_addr=} {self.num_blocks=}")
        metadata = MooncakeAgentMetadata(
            engine_id=self.engine_id,
            te_rpc_port=self.te_rpc_port,
            block_size=self.block_size,
            kv_caches_base_addr=kv_caches_base_addr,
            block_size_scale = self.block_size_scale,
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_addr,
            ssm_sizes=self._mamba_ssm_size,
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
            self.kv_recv_thread = KVCacheRecvingThread(
                self.tp_rank,
                self.tp_size,
                self._prefill_pp_size,
                self.engine,
                self.engine_id,
                self.handshake_port,
                self.side_channel_port,
                kv_caches_base_addr,
                self.block_len_per_addr,
                self._mamba_ssm_size,
                self._is_hma_required,
                self._has_mamba,
                self.hma_group_size,
                self._is_mamba_group,
                ready_event,
                self.vllm_config,
                self.kv_caches,
                self._prefill_pp_layer_partition,
                self._mamba_spec,
                [group.layer_names for group in self.kv_cache_config.kv_cache_groups],
                group_kv_cache_indices,
                self.block_size_scale
            )
            self.kv_recv_thread.start()
        print(f"[group_kv_cache_indices] {group_kv_cache_indices}")
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

    def _get_kv_split_metadata(
        self,
        req_id: str,
        meta: ReqMeta,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        """
        In cp/dcp scenario, kv_cache may be split, so we need to pull multiple blocks from multiple remote P node.
        Use this function to calculate remote port and remote block number of each remote P node that we need to pull.
        """
        prefill_tp_size = meta.remote_ptp_size if getattr(meta, "remote_ptp_size", None) else self._prefill_tp_size
        if meta.remote_pcp_size * meta.remote_dcp_size * self.pcp_size * self.dcp_size == 1:
            chosen_rank_list = self._get_remote_rank(req_id, prefill_tp_size)
            remote_handshake_port_list = [[x + meta.remote_port for x in chosen_rank_list]]
            local_block_ids_list, remote_block_ids_list = [meta.local_block_ids], [meta.remote_block_ids]
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

        assert math.ceil(num_external_blocks / (self.pcp_size * self.dcp_size)) == len(meta.local_block_ids), (
            f"num_external_blocks({num_external_blocks}), cp_size({self.pcp_size * self.dcp_size}), "
            f"local_block_ids_len ({len(meta.local_block_ids)})"
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
            remote_block_ids_list.append(meta.remote_block_ids[:num_blocks_to_pull])
            local_block_ids_list.append(
                meta.local_block_ids[local_block_offset : local_block_offset + num_blocks_to_pull]
            )
            local_block_offset += num_blocks_to_pull

        tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
        assert tp_num_need_pulls == len(remote_handshake_port_list[0]), (
            f"tp_num_need_pulls: {tp_num_need_pulls}, remote_handshake_port_list: {remote_handshake_port_list}"
        )

        return remote_handshake_port_list, local_block_ids_list, remote_block_ids_list

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        """Start loading KV blocks from remote engine."""
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

            prefill_tp_size = meta.remote_ptp_size if getattr(meta, "remote_ptp_size", None) else self._prefill_tp_size
            tp_num_need_pulls = self._get_tp_num_need_pulls(prefill_tp_size)
            remote_req_id = meta.remote_request_id

            if meta.remote_pcp_size * meta.remote_dcp_size > 1 and (not self._has_mamba):
                remote_handshake_port_list, local_block_ids_list, remote_block_ids_list = self._get_kv_split_metadata(
                    req_id, meta
                )

                for pcp_dcp_rank in range(len(remote_handshake_port_list)):
                    for i in range(tp_num_need_pulls):
                        assert self.kv_recv_thread is not None
                        remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                            meta.remote_port,
                            remote_handshake_port_list[pcp_dcp_rank][i],
                            meta.remote_host,
                            meta.remote_engine_id,
                            meta.remote_multi_nodes_meta_mapping,
                        )
                        self.kv_recv_thread.add_request(
                            request_id=req_id,
                            remote_request_id=remote_req_id,
                            local_block_ids=local_block_ids_list[pcp_dcp_rank],
                            remote_block_ids=remote_block_ids_list[pcp_dcp_rank],
                            remote_engine_id=remote_engine_id,
                            remote_host=remote_host,
                            remote_handshake_port=remote_handshake_port_list[pcp_dcp_rank][i],
                            offset=i,
                            tp_num_need_pulls=tp_num_need_pulls,
                            remote_port_send_num=self.remote_port_send_num[meta.remote_engine_id],
                            all_task_done=(
                                pcp_dcp_rank == len(remote_handshake_port_list) - 1 and i == tp_num_need_pulls - 1
                            ),
                        )
            else:  # TODO: support prefill context parallel and pipeline parallel open at the same time
                chosen_rank_list = self._get_remote_rank(remote_req_id, prefill_tp_size)
                remote_handshake_port_list = [[x + meta.remote_port] for x in chosen_rank_list]
                for i in range(tp_num_need_pulls * self._prefill_pp_size):
                    assert self.kv_recv_thread is not None
                    remote_host, remote_engine_id = self._get_remote_host_info_by_port(
                        meta.remote_port,
                        remote_handshake_port_list[i][0],
                        meta.remote_host,
                        meta.remote_engine_id,
                        meta.remote_multi_nodes_meta_mapping,
                    )
                    self.kv_recv_thread.add_request(
                        request_id=req_id,
                        remote_request_id=remote_req_id,
                        local_block_ids=meta.local_block_ids,
                        remote_block_ids=meta.remote_block_ids,
                        remote_engine_id=remote_engine_id,
                        remote_host=remote_host,
                        remote_handshake_port=remote_handshake_port_list[i][0],
                        offset=i,
                        tp_num_need_pulls=tp_num_need_pulls,
                        all_task_done=(i == tp_num_need_pulls * self._prefill_pp_size - 1),
                    )

        for req_id in metadata.reqs_in_batch:
            if self.kv_send_thread is not None:
                self.kv_send_thread.task_tracker.add_req_to_process(req_id)
            if self.kv_recv_thread is not None:
                self.kv_recv_thread.task_tracker.add_req_to_process(req_id)

        if self.kv_send_thread is not None and self.pcp_size * self.dcp_size == 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                if self.tp_rank in self._prefill_get_remote_rank(req_id):
                    self.kv_send_thread.add_delayed_request(req_id, delay_start_time)
                else:
                    self.kv_send_thread.add_not_transfer_request(req_id)

        if self.kv_send_thread is not None and self.pcp_size * self.dcp_size > 1:
            for req_id, delay_start_time in metadata.requests_to_send.items():
                self.kv_send_thread.add_delayed_request(req_id, delay_start_time)

    def _get_tp_num_need_pulls(self, prefill_tp_size: int) -> int:
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
        ori_data = ori_data.reshape(self._prefill_pp_size, -1)
        num_groups = max(
            1, len(ori_data[0]) // num_kv_head
        )  # The number of redundant copies for each KV head within the PP stage
        rand_group_index = rand.sample(
            range(num_groups), (max(self._decode_tp_size // num_kv_head, 1))
        )  # random choose a group
        all_results = [
            self._get_remote_tp_ranks(ori_data[pp_index], rand_group_index, num_groups, prefill_tp_size)
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
    src: list[int], dst: list[int]
) -> tuple[list[npt.NDArray[np.int64]], list[npt.NDArray[np.int64]]]:
    """Vectorised NumPy implementation."""
    src_indices: npt.NDArray[np.int64] = np.array(src, dtype=np.int64)
    dst_indices: npt.NDArray[np.int64] = np.array(dst, dtype=np.int64)

    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


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
                logger.warning(f"Send failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Send failed after all retries: {e}")
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
                logger.warning(f"Receive failed: {e}, retrying... ({retries_left} attempts left)")
                time.sleep(0.1)
            else:
                logger.error(f"Receive failed from {path} after all retries: {e}")
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
