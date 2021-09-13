#!/bin/bash

NNODES=1
NPROC_PER_NODE=2
MASTER_IP=localhost
MASTER_PORT=2021


python -m torch.distributed.run \
	--nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE \
	--rdzv_id=1234 --rdzv_backend=c10d \
	--rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
	main.py
