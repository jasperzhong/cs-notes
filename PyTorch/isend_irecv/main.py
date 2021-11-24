import os
from datetime import timedelta

import torch


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl', init_method="env://",
        timeout=timedelta(seconds=5)
    )

    rank = torch.distributed.get_rank()
    size = (1000000, )
    if rank == 0:
        x = torch.zeros(size=size)
        req = torch.distributed.irecv(x, 1)
        req.wait()
        x += x
        print(x[0])
    else:
        x = torch.ones(size=size)
        torch.distributed.send(x, 0)


if __name__ == '__main__':
    main()
