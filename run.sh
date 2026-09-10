#!/bin/bash
set -e

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-29500}
export WORLD_SIZE=${WORLD_SIZE:-$(nvidia-smi -L | grep ^GPU | wc -l)}

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_NET_GDR_LEVEL="LOC"
export NCCL_SOCKET_FAMILY=AF_INET

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--config CONFIG_PATH] [other args...]"
    echo "Defaults to configs/oscllm_10m.json"
    exit 1
fi

NUMA_BIND=${NUMA_BIND:-true}
if [ "$NUMA_BIND" = true ] && command -v numactl &> /dev/null; then
    NCPUS=$(nproc)
    NNUMA=$(numaclt --hardware | grep "available:" | awk '{print $2}')
    if [ "$NNUMA" -gt 0 ]; then
        PER_NODE=$((WORLD_SIZE / NNUMA))
        for i in $(seq 0 $((NCPUS - 1))); do
            NUMACTL_ARGS="$NUMACTL_ARGS --cpunodebind=$((i % NNUMA)) --membind=$((i % NNUMA))"
        done
    fi
fi

if [ "$WORLD_SIZE" -gt 1 ]; then
    DEEPSPEED_ARGS="--num_nodes 1 --num_gpus $WORLD_SIZE"
else
    DEEPSPEED_ARGS=""
fi

CMD="deepspeed $DEEPSPEED_ARGS train/train_deepspeed.py $@"
if [ "$NUMA_BIND" = true ] && command -v numactl &> /dev/null; then
    CMD="numactl $NUMACTL_ARGS $CMD"
fi

echo "Running: $CMD"
eval $CMD
