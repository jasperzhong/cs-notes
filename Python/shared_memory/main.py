from multiprocessing import Array
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

arr = Array('i', 10)

arr[rank] = rank

print(arr[:])

