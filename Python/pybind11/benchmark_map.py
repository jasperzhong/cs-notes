# compare the performance of the map with the built-in dict
import time
import psutil
import argparse

import torch

from kvstore import KVStore


parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, default=1000000)
parser.add_argument('-batch_size', type=int, default=1000)
args = parser.parse_args()


MiB = 1024 * 1024


def get_memory_usage():
    return psutil.Process().memory_info().rss / MiB


def benchmark_map(X: torch.Tensor):
    """
    :param X: array of shape (N, 128)
    """
    m1 = get_memory_usage()
    start = time.time()
    kv = KVStore(num_threads=8)
    # in batches
    for i in range(0, len(X), args.batch_size):
        keys = list(range(i, i+args.batch_size))
        kv.set(keys, X[i:i+args.batch_size])
    end = time.time()
    insertion_time = end - start
    m2 = get_memory_usage()
    print('memory usage after insertion: {} MiB'.format(kv.memory_usage()/MiB))
    start = time.time()
    # in batches
    out = []
    for i in range(0, len(X), args.batch_size):
        keys = list(range(i, i+args.batch_size))
        out.append(kv.get(keys))
    out = torch.cat(out)
    end = time.time()
    lookup_time = end - start
    return out.sum(), insertion_time, lookup_time, m2 - m1


def benchmark_dict(X: torch.Tensor):
    """
    :param X: array of shape (N, 128)
    """
    m1 = get_memory_usage()
    start = time.time()
    d = {}
    # in batches
    for i in range(0, len(X), args.batch_size):
        keys = list(range(i, i+args.batch_size))
        for k, v in zip(keys, X[i:i+args.batch_size]):
            d[k] = v
    end = time.time()
    insertion_time = end - start
    m2 = get_memory_usage()
    start = time.time()
    # in batches
    out = []
    for i in range(0, len(X), args.batch_size):
        out.append(torch.stack([d[k] for k in range(i, i+args.batch_size)]))
    out = torch.cat(out)
    end = time.time()
    lookup_time = end - start
    return out.sum(), insertion_time, lookup_time, m2 - m1


if __name__ == "__main__":
    X = torch.ones(args.N, 186, dtype=torch.bool)
    # warmup
    benchmark_map(X)
    # benchmark
    sum1, insertion_time, lookup_time, memory = benchmark_map(X)
    print(
        f"cpp unordered_map: insertion_time={insertion_time:.2f}, lookup_time={lookup_time:.2f}, memory={memory:.2f} MiB")
    sum2, insertion_time, lookup_time, memory = benchmark_dict(X)
    print(
        f"python dict: insertion_time={insertion_time:.2f}, lookup_time={lookup_time:.2f}, memory={memory:.2f} MiB")
    assert torch.allclose(sum1, sum2)
