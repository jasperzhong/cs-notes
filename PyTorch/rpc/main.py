import os

import torch
import torch.distributed
import torch.distributed.rpc as rpc

from torchvision.models import resnet18


class SamplingResultTorch:
    def __init__(self):
        self.row: torch.Tensor = None
        self.col: torch.Tensor = None
        self.num_src_nodes: int = None
        self.num_dst_nodes: int = None
        self.all_nodes: torch.Tensor = None
        self.all_timestamps: torch.Tensor = None
        self.delta_timestamps: torch.Tensor = None
        self.eids: torch.Tensor = None

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


# let pickle know how to serialize the SamplingResultType
globals()['SamplingResultTorch'] = SamplingResultTorch


def add(a: torch.Tensor, b: torch.Tensor) -> SamplingResultTorch:
    # some cuda operations
    a = a.cuda()
    b = b.cuda()
    c = a + b

    ret = SamplingResultTorch()
    ret.row = torch.tensor([1, 2, 3])
    ret.col = torch.tensor([4, 5, 6])
    ret.num_src_nodes = 3
    ret.num_dst_nodes = 3
    ret.all_nodes = torch.tensor([1, 2, 3, 4, 5, 6])
    ret.all_timestamps = torch.tensor([1, 2, 3, 4, 5, 6])
    ret.delta_timestamps = torch.tensor([1, 2, 3, 4, 5, 6])
    ret.eids = torch.tensor([1, 2, 3, 4, 5, 6])
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

    futures = []
    for _ in range(10):
        futures.append(rpc.rpc_async(
            "worker{}".format((rank + 1) % world_size),
            add,
            args=(torch.ones(1), torch.ones(1))
        ))

    for future in futures:
        future.wait()

    torch.distributed.barrier()

    model = resnet18()
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )

    fut = rpc.rpc_async(
        "worker{}".format((rank + 1) % world_size),
        add,
        args=(torch.ones(1), torch.ones(1))
    )
    ret = fut.wait()
    print("worker{}".format(rank))
    print(ret)

    rpc.shutdown()


if __name__ == '__main__':
    main()
