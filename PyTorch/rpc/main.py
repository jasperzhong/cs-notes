import os

import torch
import torch.distributed
import torch.distributed.rpc as rpc


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    a = a.cuda()
    b = b.cuda()
    c = a + b
    return c.cpu()


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl')

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    rpc.init_rpc(
        name='worker{}'.format(rank), rank=rank, world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=8,
            rpc_timeout=10
        )
    )

    ret = rpc.rpc_sync(
        "worker{}".format((rank + 1) % world_size),
        add,
        args=(torch.ones(1), torch.ones(1))
    )


    ret = ret.cuda()
    torch.distributed.all_reduce(ret)

    print("worker{}".format(rank))
    print(ret)

    torch.distributed.barrier()
    rpc.shutdown()


if __name__ == '__main__':
    main()
