# compare the performance of the map with the built-in dict
import time
import psutil
import argparse

import torch

from kvstore import KVStore


parser = argparse.ArgumentParser()
parser.add_argument('-type', type=str, default='map', help='map or dict')
parser.add_argument('-N', type=int, default=1000000)
parser.add_argument('-ingestion_batch_size', type=int, default=10000)
parser.add_argument('-lookup_batch_size', type=int, default=10000)
args = parser.parse_args()


MiB = 1024 * 1024


def get_memory_usage():
    return psutil.Process().memory_info().rss / MiB


def benchmark_map(X: torch.Tensor):
    """
    :param X: array of shape 
    """
    m1 = get_memory_usage()
    start = time.time()
    kv = KVStore()
    # in batches
    for i in range(0, len(X), args.ingestion_batch_size):
        keys = list(range(i, i+args.ingestion_batch_size))
        kv.set(keys, X[i:i+args.ingestion_batch_size])
    end = time.time()
    insertion_time = end - start
    m2 = get_memory_usage()
    # print('memory usage after insertion: {} MiB'.format(kv.memory_usage()/MiB))
    start = time.time()
    # in batches
    for i in range(0, len(X), args.lookup_batch_size):
        keys = list(range(i, i+args.lookup_batch_size))
        torch.stack(kv.get(keys))
    end = time.time()
    lookup_time = end - start
    return insertion_time, lookup_time, m2 - m1


def benchmark_dict(X: torch.Tensor):
    """
    :param X: array of shape 
    """
    m1 = get_memory_usage()
    start = time.time()
    d = {}
    # in batches
    for i in range(0, len(X), args.ingestion_batch_size):
        keys = list(range(i, i+args.ingestion_batch_size))
        for k, v in zip(keys, X[i:i+args.ingestion_batch_size]):
            d[k] = v
    end = time.time()
    insertion_time = end - start
    m2 = get_memory_usage()
    start = time.time()
    # in batches
    for i in range(0, len(X), args.lookup_batch_size):
        torch.stack([d[k]
                     for k in range(i, i+args.lookup_batch_size)])
    end = time.time()
    lookup_time = end - start
    return insertion_time, lookup_time, m2 - m1


if __name__ == "__main__":
    X = torch.ones(args.N, 186, dtype=torch.bool)
    if args.type == 'map':
        insertion_time, lookup_time, memory = benchmark_map(X)
    elif args.type == 'dict':
        insertion_time, lookup_time, memory = benchmark_dict(X)
    else:
        raise ValueError('type must be map or dict')

    print(
        f"{args.type}: insertion_time={insertion_time:.2f}, lookup_time={lookup_time:.2f}, memory={memory:.2f} MiB")
