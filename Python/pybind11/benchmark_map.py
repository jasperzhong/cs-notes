# compare the performance of the map with the built-in dict
import time
import psutil

import torch

from kvstore import KVStore

N = 10 ** 6

MiB = 1024 * 1024


def get_memory_usage():
    return psutil.Process().memory_info().rss / MiB


def benchmark_map(X: torch.Tensor):
    """
    :param X: array of shape (N, 128)
    """
    m1 = get_memory_usage()
    start = time.time()
    kv = KVStore()
    keys = list(range(len(X)))
    kv.set(keys, X)
    end = time.time()
    insertion_time = end - start
    m2 = get_memory_usage()
    start = time.time()
    out = kv.get(keys)
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
    for i in range(len(X)):
        d[i] = X[i]
    end = time.time()
    insertion_time = end - start
    m2 = get_memory_usage()
    start = time.time()
    out = torch.stack([d[i] for i in range(len(X))])
    end = time.time()
    lookup_time = end - start
    return out.sum(), insertion_time, lookup_time, m2 - m1


if __name__ == "__main__":
    X = torch.randn(N, 128, dtype=torch.float32)
    # warmup
    benchmark_map(X)
    # benchmark
    sum1, insertion_time, lookup_time, memory = benchmark_map(X)
    print("N={}".format(N))
    print(f"cpp unordered_map: insertion_time={insertion_time:.2f}, lookup_time={lookup_time:.2f}, memory={memory:.2f} MiB")
    sum2, insertion_time, lookup_time, memory = benchmark_dict(X)
    print(
        f"python dict: insertion_time={insertion_time:.2f}, lookup_time={lookup_time:.2f}, memory={memory:.2f} MiB")
    assert torch.allclose(sum1, sum2)
