# compare the performance of the map with the built-in dict
import argparse
import time
import multiprocessing as mp

import numpy as np
import psutil
import torch

from kvstore import KVStore, AbslKVStore

parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int, default=1000000)
parser.add_argument('-ingestion_batch_size', type=int, default=10000)
parser.add_argument('-lookup_batch_size', type=int, default=1000)
args = parser.parse_args()


MiB = 1024 * 1024
NUM_BENHMARCKS = 5


def get_memory_usage():
    return psutil.Process().memory_info().rss / MiB


def benchmark_absl_flat_hash_map(X: torch.Tensor, query_keys: torch.Tensor):
    """
    :param X: array of shape
    """
    avg_insertion_time = 0
    avg_query_time = 0
    avg_memory_usage = 0

    def _benchmark():
        start = time.time()
        kv = AbslKVStore()
        m1 = get_memory_usage()
        # in batches
        for i in range(0, len(X), args.ingestion_batch_size):
            keys = list(range(i, i+args.ingestion_batch_size))
            kv.set(keys, X[i:i+args.ingestion_batch_size])
        end = time.time()
        insertion_time = end - start
        m2 = get_memory_usage()
        memory_usage = m2 - m1
        # print("KVStore size: {:.2f}MiB".format(kv.memory_usage() / MiB))

        start = time.time()
        for i in range(0, len(query_keys), args.lookup_batch_size):
            kv.get(query_keys[i:i+args.lookup_batch_size].tolist())
        end = time.time()
        query_time = end - start
        return insertion_time, query_time, memory_usage

    _benchmark()
    for _ in range(NUM_BENHMARCKS):
        insertion_time, query_time, memory_usage = _benchmark()
        avg_insertion_time += insertion_time
        avg_query_time += query_time
        avg_memory_usage += memory_usage
    print("absl::flat_hash_map insertion time: {:.2f}s, query time: {:.2f}s, memory usage: {:.2f}MiB".format(
        avg_insertion_time / NUM_BENHMARCKS,
        avg_query_time / NUM_BENHMARCKS,
        avg_memory_usage / NUM_BENHMARCKS))


def benchmark_unordered_map(X: torch.Tensor, query_keys: torch.Tensor):
    """
    :param X: array of shape
    """
    avg_insertion_time = 0
    avg_query_time = 0
    avg_memory_usage = 0

    def _benchmark():
        start = time.time()
        kv = KVStore()
        m1 = get_memory_usage()
        # in batches
        for i in range(0, len(X), args.ingestion_batch_size):
            keys = list(range(i, i+args.ingestion_batch_size))
            kv.set(keys, X[i:i+args.ingestion_batch_size])
        end = time.time()
        insertion_time = end - start
        m2 = get_memory_usage()
        memory_usage = m2 - m1
        # print("KVStore size: {:.2f}MiB".format(kv.memory_usage() / MiB))

        start = time.time()
        for i in range(0, len(query_keys), args.lookup_batch_size):
            kv.get(query_keys[i:i+args.lookup_batch_size].tolist())
        end = time.time()
        query_time = end - start
        return insertion_time, query_time, memory_usage

    _benchmark()
    for _ in range(NUM_BENHMARCKS):
        insertion_time, query_time, memory_usage = _benchmark()
        avg_insertion_time += insertion_time
        avg_query_time += query_time
        avg_memory_usage += memory_usage
    print("std::flat_hash_map insertion time: {:.2f}s, query time: {:.2f}s, memory usage: {:.2f}MiB".format(
        avg_insertion_time / NUM_BENHMARCKS,
        avg_query_time / NUM_BENHMARCKS,
        avg_memory_usage / NUM_BENHMARCKS))


def benchmark_dict_id2tensor(X: torch.Tensor, query_keys: torch.Tensor):
    """
    :param X: array of shape
    """
    avg_insertion_time = 0
    avg_query_time = 0
    avg_memory_usage = 0

    def _benchmark():
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
        memory_usage = m2 - m1

        start = time.time()
        for i in range(0, len(query_keys), args.lookup_batch_size):
            torch.stack([d[k]
                        for k in query_keys[i:i+args.lookup_batch_size].tolist()])
        end = time.time()
        query_time = end - start
        return insertion_time, query_time, memory_usage

    _benchmark()
    for _ in range(NUM_BENHMARCKS):
        insertion_time, query_time, memory_usage = _benchmark()
        avg_insertion_time += insertion_time
        avg_query_time += query_time
        avg_memory_usage += memory_usage
    print("dict(id -> tensor) insertion time: {:.2f}s, query time: {:.2f}s, memory usage: {:.2f}MiB".format(
        avg_insertion_time / NUM_BENHMARCKS,
        avg_query_time / NUM_BENHMARCKS,
        avg_memory_usage / NUM_BENHMARCKS))


def benchmark_dict_id2numpy(X: torch.Tensor, query_keys: torch.Tensor):
    """
    :param X: array of shape
    """
    avg_insertion_time = 0
    avg_query_time = 0
    avg_memory_usage = 0

    def _benchmark():
        m1 = get_memory_usage()
        start = time.time()
        d = {}
        # in batches
        for i in range(0, len(X), args.ingestion_batch_size):
            keys = list(range(i, i+args.ingestion_batch_size))
            for k, v in zip(keys, X[i:i+args.ingestion_batch_size]):
                d[k] = v.numpy()
        end = time.time()
        insertion_time = end - start
        m2 = get_memory_usage()
        memory_usage = m2 - m1

        start = time.time()
        for i in range(0, len(query_keys), args.lookup_batch_size):
            torch.from_numpy(np.stack([d[k]
                                       for k in query_keys[i:i+args.lookup_batch_size].tolist()]))

        end = time.time()
        query_time = end - start
        return insertion_time, query_time, memory_usage

    _benchmark()
    for _ in range(NUM_BENHMARCKS):
        insertion_time, query_time, memory_usage = _benchmark()
        avg_insertion_time += insertion_time
        avg_query_time += query_time
        avg_memory_usage += memory_usage
    print("dict(id -> numpy) insertion time: {:.2f}s, query time: {:.2f}s, memory usage: {:.2f}MiB".format(
        avg_insertion_time / NUM_BENHMARCKS,
        avg_query_time / NUM_BENHMARCKS,
        avg_memory_usage / NUM_BENHMARCKS))


def benchmark_dict_id2idx(X: torch.Tensor, query_keys: torch.Tensor):
    """
    :param X: array of shape
    """
    avg_insertion_time = 0
    avg_query_time = 0
    avg_memory_usage = 0

    def _benchmark():
        m1 = get_memory_usage()
        start = time.time()
        d = {}
        # in batches
        out = None
        for i in range(0, len(X), args.ingestion_batch_size):
            keys = list(range(i, i+args.ingestion_batch_size))
            if out is None:
                out = X[i:i+args.ingestion_batch_size]
            else:
                out = torch.cat([out, X[i:i+args.ingestion_batch_size]])
            for k, v in zip(keys, X[i:i+args.ingestion_batch_size]):
                d[k] = k
        end = time.time()
        insertion_time = end - start
        m2 = get_memory_usage()
        memory_usage = m2 - m1

        start = time.time()
        for i in range(0, len(query_keys), args.lookup_batch_size):
            out[query_keys[i:i+args.lookup_batch_size].tolist()]
        end = time.time()
        query_time = end - start
        return insertion_time, query_time, memory_usage

    _benchmark()
    for _ in range(NUM_BENHMARCKS):
        insertion_time, query_time, memory_usage = _benchmark()
        avg_insertion_time += insertion_time
        avg_query_time += query_time
        avg_memory_usage += memory_usage
    print("dict(id -> idx) insertion time: {:.2f}s, query time: {:.2f}s, memory usage: {:.2f}MiB".format(
        avg_insertion_time / NUM_BENHMARCKS,
        avg_query_time / NUM_BENHMARCKS,
        avg_memory_usage / NUM_BENHMARCKS))


def benchmark_no_dict(X: torch.Tensor, query_keys: torch.Tensor):
    """
    :param X: array of shape
    """
    avg_insertion_time = 0
    avg_query_time = 0
    avg_memory_usage = 0

    def _benchmark():
        m1 = get_memory_usage()
        start = time.time()
        out = None
        ids = None
        for i in range(0, len(X), args.ingestion_batch_size):
            if out is None:
                out = X[i:i+args.ingestion_batch_size]
            else:
                out = torch.cat([out, X[i:i+args.ingestion_batch_size]])
            if ids is None:
                ids = torch.arange(i, i+args.ingestion_batch_size)
            else:
                ids = torch.cat(
                    [ids, torch.arange(i, i+args.ingestion_batch_size, dtype=torch.int64)])
        end = time.time()
        insertion_time = end - start
        m2 = get_memory_usage()
        memory_usage = m2 - m1

        start = time.time()
        for i in range(0, len(query_keys), args.lookup_batch_size):
            # find key's index in sorted ids
            keys = query_keys[i:i+args.lookup_batch_size]
            idx = torch.searchsorted(ids, keys)
            out.index_select(0, idx)

        end = time.time()
        query_time = end - start
        return insertion_time, query_time, memory_usage

    _benchmark()
    for _ in range(NUM_BENHMARCKS):
        insertion_time, query_time, memory_usage = _benchmark()
        avg_insertion_time += insertion_time
        avg_query_time += query_time
        avg_memory_usage += memory_usage
    print("no dict insertion time: {:.2f}s, query time: {:.2f}s, memory usage: {:.2f}MiB".format(
        avg_insertion_time / NUM_BENHMARCKS,
        avg_query_time / NUM_BENHMARCKS,
        avg_memory_usage / NUM_BENHMARCKS))


if __name__ == "__main__":
    X = torch.ones(args.N, 186, dtype=torch.bool)
    print("X size: {:.2f}MiB".format(X.element_size() * X.nelement() / MiB))
    query_keys = torch.randint(0, args.N, (args.N,))
    query_keys = torch.unique(query_keys)

    benchmark_funcs = [
        benchmark_dict_id2tensor,
        benchmark_dict_id2numpy,
        benchmark_absl_flat_hash_map,
        benchmark_unordered_map,
        benchmark_dict_id2idx,
        benchmark_no_dict,
    ]
    # launch a process for each benchmark
    processes = []
    for benchmark_func in benchmark_funcs:
        p = mp.Process(target=benchmark_func, args=(X, query_keys))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
