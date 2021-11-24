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
        dst_rank = (rank + 1) % 2
        x = torch.ones(size=size, requires_grad=True).cuda()
        y = torch.zeros(size=size, requires_grad=True).cuda()
        send_op = torch.distributed.P2POp(torch.distributed.isend, x,
                                          dst_rank)
        recv_op = torch.distributed.P2POp(torch.distributed.irecv, y,
                                          dst_rank)
        reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

        z = x + y
        assert z[0].item() == 2, "wrong"

    print("right")


if __name__ == '__main__':
    main()
