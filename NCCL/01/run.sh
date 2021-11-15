#!/bin/bash
# `make` before running on both machines
# run this script on net-g1

export NCCL_DEBUG=INFO

eval `ssh-agent`
ssh-add ~/yczhong.pem

# use absolute path
mpirun -npernode 1 --host 172.30.2.12,172.30.2.79 /home/ubuntu/repos/cs-notes/NCCL/01/test -x NCCL_DEBUG 
