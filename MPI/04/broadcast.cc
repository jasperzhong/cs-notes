#include <iostream>
#include <mpich/mpi.h>

void my_bcast(void* data, int count, MPI_Datatype datatype, int root, MPI_Comm communicator)
{
	int world_rank;
	MPI_Comm_rank(communicator, &world_rank);

	int wolrd_size;
	MPI_Comm_size(communicator, &wolrd_size);

	if (world_rank == root) {
		for (int i = 0; i < wolrd_size; ++i) {
			if (i != world_rank) {
				MPI_Send(data, count, datatype, i, 0, communicator);
			}
		}
	} else {
		MPI_Recv(data, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
	}
}

int main(int argc, char* argv[])
{
	if (argc != 3) {
		std::cerr << "Usage: broadcast num_elements num_trials" << std::endl;
		exit(1);
	}

	int num_elements = atoi(argv[1]);
	int num_trials = atoi(argv[2]);

	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	double total_my_bcast_time = 0.0;
	double total_mpi_bcast_time = 0.0;

	int* data = new int[num_elements];

	for (int i = 0; i < num_trials; ++i) {
		MPI_Barrier(MPI_COMM_WORLD);
		total_my_bcast_time -= MPI_Wtime();
		my_bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		total_my_bcast_time += MPI_Wtime();

		MPI_Barrier(MPI_COMM_WORLD);
		total_mpi_bcast_time -= MPI_Wtime();
		MPI_Bcast(data, num_elements, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		total_mpi_bcast_time += MPI_Wtime();
	}

	if (world_rank == 0) {
		std::cout << "Data size = " << num_elements * sizeof(int) << ", Trails = " << num_trials << std::endl;
		std::cout << "Avg my_bcast time = " << total_my_bcast_time / num_trials << std::endl;
		std::cout << "Avg MPI_Bcast time = " << total_mpi_bcast_time / num_trials << std::endl;
	}

	delete[] data;
	MPI_Finalize();
}
