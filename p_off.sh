vllm serve /home/weight/Qwen3-30B-A3B-Instruct-2507 \
  --host 0.0.0.0 \
  --port 30060 \
  --enable-expert-parallel \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --data-parallel-start-rank 0 \
  --api-server-count 1 \
  --data-parallel-address 90.90.97.28 \
  --data-parallel-rpc-port 5964  \
  --tensor-parallel-size 8 \
  --seed 1024 \
  --served-model-name deepseek \
  --distributed-executor-backend mp \
  --max-model-len 40000 \
  --max-num-batched-tokens 512 \
  --trust-remote-code \
  --max-num_seqs 16 \
  --gpu-memory-utilization 0.95 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY","cudagraph_capture_sizes":[1, 8, 16]}' \
  --kv-transfer-config \
  '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "kv_connector_extra_config": {
      "connectors": [
        {
          "kv_connector": "MooncakeLayerwiseConnector",
          "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector",
          "kv_buffer_device": "npu",
          "kv_role": "kv_consumer",
          "kv_port": "23010",
          "engine_id": "1",
          "kv_connector_extra_config": {
            "prefill": { "dp_size": 1, "tp_size": 8 },
            "decode":  { "dp_size": 1, "tp_size": 8 }
          }
        },
        {
          "kv_connector": "CPUOffloadingConnector",
          "kv_connector_module_path": "vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload.cpu_offload_connector",
          "kv_role": "kv_both",
          "kv_connector_extra_config": {
            "cpu_swap_space_gb": 200,
            "swap_in_threshold": 0
          }
        }
      ]
    }
  }'
