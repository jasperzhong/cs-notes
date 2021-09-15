import argparse
import os
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from model import PipelineParallelResNet50
from schedule import pipedream_flush_schedule, initialize_global_args

parser = argparse.ArgumentParser(
    description='Pipeline Parallel ResNet50 Arguments')
parser.add_argument('--pipeline-model-parallel-size', type=int, default=1, help='Degree of pipeline model parallelism')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='num of classes in vision classification task')
parser.add_argument('--micro-batch-size', type=int, default=None,
                    help='Batch size per model instance (local batch size).')
parser.add_argument('--global-batch-size', type=int,
                    default=None, help='Training batch size.')

def get_data_iterator(args):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_kwargs = {'batch_size': args.micro_batch_size, 'num_workers': 1,
                    'pin_memory': True, 'shuffle': True}
    train_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
    data_iterator = iter(train_loader)
    return data_iterator

def train(data_iterator, model, optimizer, loss_func):
    optimizer.zero_grad()
    pipedream_flush_schedule(data_iterator, model, loss_func)
    optimizer.step()


def main():
    args = parser.parse_args()
    initialize_global_args(args)

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl', world_size=args.world_size, rank=args.rank,
        timeout=timedelta(seconds=10)
    )

    data_iterator = get_data_iterator(args)
    model = PipelineParallelResNet50(balance=[6, 5])
    model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()

    train(data_iterator, model, optimizer, loss_func)


if __name__ == '__main__':
    main()
