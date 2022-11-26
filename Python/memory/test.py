import sys
import argparse

import psutil
import torch

MiB = 1024 * 1024

parser = argparse.ArgumentParser()
parser.add_argument("-N", type=int, default=1000000)
args = parser.parse_args()


def get_memory_usage():
    return psutil.Process().memory_info().rss / MiB


def get_memory_usage_of_dict(d):
    size = sys.getsizeof(d)
    for k, v in d.items():
        size += sys.getsizeof(k) + sys.getsizeof(v)
    return size / MiB


def test(size: int):
    x = torch.randn(size, 128)
    size_of_tensor = x.element_size() * x.nelement() / MiB

    m2 = get_memory_usage()
    d = {}
    for i in range(len(x)):
        d[i] = x[i]

    m3 = get_memory_usage()
    size_of_dict = get_memory_usage_of_dict(d)

    print("N = {}, Memory usage: {:.1f} MiB, Size of tensor: {:.1f} MiB, Size of dict: {:.1f} MiB, Diff: {:.1f} MiB".format(
        len(x), m3 - m2, size_of_tensor, size_of_dict, m3 - m2 - size_of_tensor - size_of_dict))


if __name__ == "__main__":
    test(args.N)
