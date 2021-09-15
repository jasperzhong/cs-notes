#!/bin/bash

NNODES=2
NPROC_PER_NODE=1
MASTER_IP=172.30.2.12

python3 -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=static \
	--rdzv_endpoint=$MASTER_IP \
	--node_rank 0 \
	main.py
