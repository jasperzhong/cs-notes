#include <algorithm>
#include <mpich/mpi.h>
#include <vector>

float* gather_numbers_to_root(float* number, MPI_Comm comm)
{
	int comm_rank, comm_size;
	MPI_Comm_rank(comm, &comm_rank);
	MPI_Comm_size(comm, &comm_size);

	float* gathered_numbers = NULL;
	if (comm_rank == 0) {
		gathered_numbers = new float[comm_size];
	}

	MPI_Gather(number, 1, MPI_FLOAT, gathered_numbers, 1, MPI_FLOAT, 0, comm);

	return gathered_numbers;
}

int* get_ranks(float* gathered_numbers, int gathered_number_count)
{
	std::vector<std::pair<float, int>> temp(gathered_number_count);
	for (int i = 0; i < gathered_number_count; ++i) {
		temp[i] = std::make_pair(gathered_numbers[i], i);
	}

	std::sort(temp.begin(), temp.end());

	int* ranks = new int[gathered_number_count];

	for (int i = 0; i < temp.size(); ++i) {
		ranks[temp[i].second] = i;
	}
	return ranks;
}

int TMPI_Rank(float* send_data, int* recv_data, MPI_Comm comm)
{
	int comm_rank, comm_size;
	MPI_Comm_rank(comm, &comm_rank);
	MPI_Comm_size(comm, &comm_size);

	float* gathered_numbers = gather_numbers_to_root(send_data, comm);

	int* ranks = NULL;
	if (comm_rank == 0) {
		ranks = get_ranks(gathered_numbers, comm_size);
	}

	MPI_Scatter(ranks, 1, MPI_INT, recv_data, 1, MPI_INT, 0, comm);

	if (comm_rank == 0) {
		delete[] gathered_numbers;
		delete[] ranks;
	}

	return 0;
}
