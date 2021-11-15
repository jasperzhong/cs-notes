#!/bin/bash
# `make` before running on both machines
# run this script on net-g1

export LD_LIBRARY_PATH=$LD_LDBRARY_PATH:/home/yczhong/repos/nccl/build/lib
export NCCL_SOCKET_IFNAME=eth2
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# use absolute path
mpirun -np 4 --hosts 10.28.1.16,10.28.1.17 /home/yczhong/repos/cs-notes/NCCL/01/test -x LD_LIBRARY_PATH -x NCCL_DEBUG -x NCCL_SOCKET_IFNAME -x NCCL_IB_DISABLE
