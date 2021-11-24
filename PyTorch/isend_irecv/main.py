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
            x = torch.ones(size=size, requires_grad=True)
            y = torch.zeros(size=size, requires_grad=True)
            send_op = torch.distributed.P2POp(torch.distributed.isend, x,
                                              1)
            recv_op = torch.distributed.P2POp(torch.distributed.irecv, y,
                                              1)
            reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()

            z = x + y
            assert z[0].item() == 2, "wrong"
        else:
            y = torch.ones(size=size, requires_grad=True)
            x = torch.zeros(size=size, requires_grad=True)
            send_op = torch.distributed.P2POp(torch.distributed.isend, x,
                                              0)
            recv_op = torch.distributed.P2POp(torch.distributed.irecv, y,
                                              0)
            reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
            for req in reqs:
                req.wait()

            z = x + y
            assert z[0].item() == 2, "wrong"

    print("right")


if __name__ == '__main__':
    main()
