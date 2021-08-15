#ifndef PARALLEL_RANK_H_
#define PARALLEL_RANK_H_

#include <mpich/mpi.h>

extern int TMPI_Rank(float* send_data, int* recv_data, MPI_Comm comm);

#endif
