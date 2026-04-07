
unset ftp_proxy
unset https_proxy
unset http_proxy
export VLLM_NIXL_ABORT_REQUEST_TIMEOUT=30000
source /home/liziyu/CONFIG
export HCCL_EXEC_TIMEOUT=60
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=$IP_ADDRESS
export GLOO_SOCKET_IFNAME=$NETWORK_CARD_NAME
export TP_SOCKET_IFNAME=$NETWORK_CARD_NAME
export HCCL_SOCKET_IFNAME=$NETWORK_CARD_NAME
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=2048
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export DISAGGREGATED_PREFILL_RANK_TABLE_PATH=/home/liziyu/b061/vllm-ascend/examples/disaggregated_prefill_v1/ranktable.json
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_DETERMINISTIC=true
export VLLM_ASCEND_LLMDD_RPC_PORT=6557
export TASK_QUEUE_ENABLE=1
export exportVLLM_LOGGING_LEVEL="info"
# export ASCEND_AGGREGATE_ENABLE=1
# export ASCEND_TRANSPORT_PRINT=0
# export ACL_OP_INIT_MODE=1
# export ASCEND_A3_ENABLE=1

export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# vllm serve /mnt/weight/deepseekv3-lite-base-latest-w8a8-dynamic \
# vllm serve /mnt/weight/Qwen3-30B-A3B-Instruct-2507 \

#vllm serve /mnt/weight/Qwen3-235B-A22B-W8A8 \
#  --enforce-eager \
#vllm serve /mnt/weight/Qwen3-8B \
#vllm serve /mnt/weight/deepseek_diff/deepseek_r1_w8a8_vllm \
#  --no-enable-prefix-caching \

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
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_consumer",
  "kv_port": "23010",
  "engine_id": "1",
  "kv_connector_extra_config": {
            "prefill": {
                    "dp_size": 1,
                    "tp_size": 8
             },
             "decode": {
                    "dp_size": 1,
                    "tp_size": 8
             }
      }
  }'


#export VLLM_USE_V2_MODEL_RUNNER=1
