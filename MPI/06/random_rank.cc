#include "tmpi_rank.h"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpich/mpi.h>

int main(int argc, char* argv[])
{
	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	srand(time(NULL) * world_rank);

	float rand_num = rand() / (float)RAND_MAX;
	int rank;
	TMPI_Rank(&rand_num, &rank, MPI_COMM_WORLD);
	std::cout << "Rank for " << rand_num << " on process " << world_rank << " is " << rank << std::endl;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
