import argparse
import os
from datetime import timedelta

import torch

parser = argparse.ArgumentParser(
    description='Pipeline Parallel ResNet50 Arguments')
parser.add_argument('--pipeline-model-parallel-size', type=int,
                    default=1, help='Degree of pipeline model parallelism')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='num of classes in vision classification task')
parser.add_argument('--micro-batch-size', type=int, default=None,
                    help='Batch size per model instance (local batch size).')
parser.add_argument('--global-batch-size', type=int,
                    default=None, help='Training batch size.')


def main():
    args = parser.parse_args()

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl', world_size=args.world_size, rank=args.rank,
        timeout=timedelta(seconds=10)
    )


if __name__ == '__main__':
    main()
