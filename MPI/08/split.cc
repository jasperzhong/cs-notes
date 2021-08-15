#include <iostream>
#include <mpich/mpi.h>

int main(int argc, char* argv[])
{
	MPI_Init(NULL, NULL);

	int world_rank, world_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int color = world_rank / 4;

	MPI_Comm row_comm;
	MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &row_comm);

	int row_rank, row_size;
	MPI_Comm_rank(row_comm, &row_rank);
	MPI_Comm_size(row_comm, &row_size);

	std::cout << "WORLD RANK/SIZE: " << world_rank << "/" << world_size << " --- "
		  << "ROW RANK/SIZE: " << row_rank << "/" << row_size << std::endl;

	MPI_Comm_free(&row_comm);

	MPI_Finalize();
}
