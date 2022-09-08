#include <cuda_runtime.h>
#include <fcntl.h> /* For O_* constants */
#include <math.h>
#include <mpich/mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>

void launch_kernel(int N, const float *x, float target, int *idx);

int main(void) {
  srand(time(NULL));
  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cudaSetDevice(rank);

  int N = 1 << 20;

  int fd;
  if (rank == 0) {
    fd = shm_open("/shm", O_RDWR | O_CREAT, 0666);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank != 0) {
    fd = shm_open("/shm", O_RDWR, 0666);
  }

  if (fd == -1) {
    printf("rank %d: shm_open failed\n", rank);
    return 1;
  }

  if (rank == 0) {
    int size = N * sizeof(float);
    ftruncate(fd, size);
  }

  float *x = (float *)mmap(NULL, N * sizeof(float), PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);

  if (x == MAP_FAILED) {
    printf("mmap failed\n");
    return 1;
  }

  float target = 50.0f;
  int *idx;
  cudaHostAlloc(&idx, sizeof(int), 0);
  if (rank == 0) {
    for (int i = 0; i < N; ++i) {
      x[i] = rand() % 100;
    }

    std::sort(x, x + N);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  cudaHostRegister(x, N * sizeof(float), cudaHostRegisterDefault);

  launch_kernel(N, x, target, idx);

  printf("Rank %d: %d\n", rank, *idx);
  printf("lower bound: %ld\n", std::lower_bound(x, x + N, target) - x);

  cudaHostUnregister(x);
  munmap(x, N * sizeof(float));

  shm_unlink("/shm");

  MPI_Finalize();
  return 0;
}
