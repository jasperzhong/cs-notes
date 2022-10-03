import os

import torch
import torch.distributed
import torch.distributed.rpc as rpc


class CustomClass:
    def __init__(self):
        self.a: torch.Tensor = None
        self.b: torch.Tensor = None
        self.c: int = 0

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


globals()['CustomClass'] = CustomClass


def add(a: torch.Tensor, b: torch.Tensor) -> CustomClass:
    ret = CustomClass()
    ret.a = torch.randn_like(a)
    ret.b = torch.randn_like(b)
    ret.c = int(torch.randn(1).item())
    return ret


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl')

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    rpc.init_rpc(
        name='worker{}'.format(rank), rank=rank, world_size=world_size
    )

    fut = rpc.rpc_async(
        "worker{}".format((rank + 1) % world_size),
        add,
        args=(torch.ones(1), torch.ones(1))
    )

    ret = fut.wait().cuda()

    print("worker{}".format(rank))
    print(ret)

    torch.distributed.barrier()
    rpc.shutdown()


if __name__ == '__main__':
    main()
