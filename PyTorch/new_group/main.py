import os

import torch


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl'
    )

    rank = torch.distributed.get_rank()
    if rank in [0, 2]:
        new_group = torch.distributed.new_group([0, 2])
    else:
        new_group = torch.distributed.new_group([1, 3])

    new_rank = torch.distributed.get_rank(new_group)
    print(f"{rank} -> {new_rank}")

    x = torch.randn(100)
    torch.distributed.all_reduce(x, group=new_group)
    print(f"{rank} {torch.sum(x)}")



if __name__ == '__main__':
    main()
