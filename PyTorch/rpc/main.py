import os

import torch
import torch.distributed
import torch.distributed.rpc as rpc


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl', init_method="env://",
    )


    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    rpc.init_rpc(
        name='worker{}'.format(rank), rank=rank, world_size=world_size)

    if rank == 0:
        ret = rpc.rpc_sync(
            "worker{}".format((rank + 1) % world_size),
            torch.add, args=(torch.tensor(1),
                             torch.tensor(2)))
        print("worker{}".format(rank))
        print(ret)

    rpc.shutdown()


if __name__ == '__main__':
    main()
