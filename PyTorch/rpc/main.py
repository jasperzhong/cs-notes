import os
from datetime import timedelta

import torch
import torch.distributed.rpc as rpc


def main():
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    torch.distributed.init_process_group(
        'gloo', world_size=world_size,
        rank=rank, timeout=timedelta(seconds=10)
    )

    name = "worker{}".format(rank)
    print(name)
    rpc.init_rpc(name, rank=rank, world_size=world_size)
    torch.distributed.barrier()

    my_count = rank

    def my_add():
        my_count += 1

    if rank == 0:
        rpc.rpc_sync("worker1", my_add)

    torch.distributed.barrier()
    print(my_count)


if __name__ == "__main__":
    main()
