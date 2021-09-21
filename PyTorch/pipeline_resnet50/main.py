import argparse
import os
from datetime import timedelta

import torch

parser = argparse.ArgumentParser(
    description='minimal code to reproduce the bug')
parser.add_argument('--master_ip', default=None, type=str,
                    help='master ip for c10d')
parser.add_argument('--master_port', default=None, type=int,
                    help='master port for c10d')


def main():
    args = parser.parse_args()

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    init_method = "tcp://{}:{}".format(args.master_ip, args.master_port)
    torch.distributed.init_process_group(
        'nccl', init_method=init_method,
        world_size=args.world_size, rank=args.rank,
        timeout=timedelta(seconds=10)
    )

    x = torch.randn((1000)).cuda()

    if torch.distributed.get_rank() == 0:
        torch.distributed.send(x, 1)
    elif torch.distributed.get_rank() == 1:
        torch.distributed.recv(x, 0)

    print(x)


if __name__ == '__main__':
    main()
