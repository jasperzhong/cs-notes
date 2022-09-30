#!/bin/bash

INTERFACE="eth2"

NNODES=2
NPROC_PER_NODE=1
MASTER_IP=10.28.1.16

CURRENT_NODE_IP=$(ip -4 a show dev ${INTERFACE} | grep inet | cut -d " " -f6 | cut -d "/" -f1)
if [ $CURRENT_NODE_IP = $HOST_NODE_ADDR ]; then
    IS_HOST=true
else
    IS_HOST=false
fi

export NCCL_SOCKET_IFNAME=${INTERFACE}
export GLOO_SOCKET_IFNAME=${INTERFACE}
export TP_SOCKET_IFNAME=${INTERFACE}

cmd="python3 -m torch.distributed.run \
    --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=1234 --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP \
    --rdzv_conf is_host=$IS_HOST \
    main.py"

LOGLEVEL=DEBUG NCCL_DEBUG=INFO exec $cmd
