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
    for _ in range(100):
        if rank == 0:
            x = torch.zeros(size=size).cuda()
            req = torch.distributed.irecv(x, 1)
            req.wait()
            x += x
            assert x[0].item() == 2, "wrong"
        else:
            x = torch.ones(size=size).cuda()
            torch.distributed.send(x, 0)
    
    print("right")


if __name__ == '__main__':
    main()
