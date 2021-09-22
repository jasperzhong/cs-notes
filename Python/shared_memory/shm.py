import time
from multiprocessing import shared_memory

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

nitems = 10
dtype = np.int32
itemsize = np.dtype(dtype).itemsize
size = nitems * itemsize

if rank == 0:
    shm = shared_memory.SharedMemory(name='mpi_shm', create=True, size=size)
comm.Barrier()
if rank != 0:
    shm = shared_memory.SharedMemory(name='mpi_shm', create=False, size=size)

arr = np.ndarray(nitems, dtype=dtype, buffer=shm.buf)

if rank == 0:
    arr[0] = 100


comm.Barrier()
print(arr)

if rank != 0:
    shm.close()

comm.Barrier()
time.sleep(1)
if rank == 0:
    shm.close()
    shm.unlink()
