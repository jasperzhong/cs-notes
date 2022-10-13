import os

import torch
import torch.distributed
import torch.distributed.rpc as rpc

from torchvision.models import resnet18


def add(a: torch.Tensor, b: torch.Tensor):
    # some cuda operations
    a = a.cuda()
    b = b.cuda()
    c = a + b


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('gloo')

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    rpc.init_rpc(
        name='worker{}'.format(rank), rank=rank, world_size=world_size
    )

    futures = []
    for i in range(10):
        a = torch.rand(100, 100)
        b = torch.rand(100, 100)
        add(a, b)
        futures.append(rpc.rpc_async(
            "worker{}".format((rank + 1) % world_size),
            add, args=(a, b)))

    for future in futures:
        future.wait()

    torch.distributed.barrier()

    model = resnet18()
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank]
    )

    a = torch.rand(100, 100)
    b = torch.rand(100, 100)
    add(a, b)

    fut = rpc.rpc_async(
        "worker{}".format((rank + 1) % world_size),
        add, args=(a, b))
    fut.wait()
    print("worker{} done".format(rank))

    rpc.shutdown()


if __name__ == '__main__':
    main()
