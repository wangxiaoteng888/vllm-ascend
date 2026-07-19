        # Reformat metadata keyed by request_id then CP shard index. Populated by the
        # last TP-offset pull task for each shard; applied once all pull tasks finish.
        self.pending_reformat: defaultdict[str, dict[int, list[tuple[int, list[list[int]], int, list[int]]]]] = (
            defaultdict(dict)
        )
        self.pending_reformat_lock = threading.Lock()


shard_idx: int = 0,

"shard_idx": shard_idx,


            all_tasks_done = self._mark_request_task_done(request_id, all_task_done)
            if all_tasks_done:
                if transfer_failed or self._is_failed_recv_request(request_id):
                    with self.pending_reformat_lock:
                        self.pending_reformat.pop(request_id, None)
                else:
                    try:
                        self._reformat_pending_kv_caches(request_id)
                    except Exception as e:
                        transfer_failed = True
                        self._mark_failed_recv_request(request_id, req_meta["local_block_ids"])
                        with self.pending_reformat_lock:
                            self.pending_reformat.pop(request_id, None)
                        logger.exception(
                            "Failed to reformat KV cache after all pulls for request %s: %s",
                            remote_request_id,
                            e,
                        )







        if ready_attention_group_reformat_block_ids:
            shard_idx = int(req_meta.get("shard_idx", 0))
            self._stash_pending_reformat(
                req_meta["request_id"],
                shard_idx,
                ready_attention_group_reformat_block_ids,
            )

    def _stash_pending_reformat(
        self,
        request_id: str,
        shard_idx: int,
        ready_attention_group_reformat_block_ids: list[tuple[int, list[list[int]], int, list[int]]],
    ) -> None:
        with self.pending_reformat_lock:
            self.pending_reformat[request_id][shard_idx] = ready_attention_group_reformat_block_ids

    def _reformat_pending_kv_caches(self, request_id: str) -> None:
        with self.pending_reformat_lock:
            shard_reformats = self.pending_reformat.pop(request_id, {})
        for shard_idx in sorted(shard_reformats):
            logger.debug(
                "Reformatting KV cache after all pulls completed. request_id=%s shard_idx=%s",
                request_id,
                shard_idx,
            )
            self._apply_kv_cache_reformat(shard_reformats[shard_idx])

    def _apply_kv_cache_reformat(
        self,
        ready_attention_group_reformat_block_ids: list[tuple[int, list[list[int]], int, list[int]]],
    ) -> None:




                        shard_idx=pcp_dcp_rank,
