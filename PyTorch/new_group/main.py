import os
import time

import torch


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        'nccl'
    )

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    if rank % 2 == 0:
        new_group = torch.distributed.new_group([r for r in range(world_size) if r % 2 == 0])
    else:
        time.sleep(1)
        new_group = torch.distributed.new_group([r for r in range(world_size) if r % 2 == 1])

    group_size = torch.distributed.get_world_size(new_group)
    group_rank = torch.distributed.get_rank(new_group)
    print(f"{rank}/{world_size} -> {group_rank}/{group_size}")

    x = torch.randn(100).cuda()
    torch.distributed.all_reduce(x)
    print(f"{rank}/{world_size} {torch.sum(x)}")

    torch.distributed.all_reduce(x, group=new_group)
    print(f"{group_rank}/{group_size} {torch.sum(x)}")


if __name__ == '__main__':
    main()
