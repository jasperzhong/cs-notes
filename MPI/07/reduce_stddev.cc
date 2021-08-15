#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpich/mpi.h>

float* create_rand_nums(int num_elements)
{
	float* rand_nums = new float[num_elements];
	for (int i = 0; i < num_elements; ++i) {
		rand_nums[i] = rand() / (float)RAND_MAX;
	}

	return rand_nums;
}

int main(int argc, char* argv[])
{
	if (argc != 2) {
		std::cerr << "Usage: reduce_avg num_elements_per_proc" << std::endl;
		exit(1);
	}

	int num_elements_per_proc = atoi(argv[1]);

	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	srand(time(NULL) * world_rank);

	float* rand_nums = create_rand_nums(num_elements_per_proc);

	double local_sum = 0;
	for (int i = 0; i < num_elements_per_proc; ++i) {
		local_sum += rand_nums[i];
	}

	double global_sum;
	MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	float mean = global_sum / (num_elements_per_proc * world_size);

	double local_sq_diff = 0;
	for (int i = 0; i < num_elements_per_proc; ++i) {
		local_sq_diff += (rand_nums[i] - mean) * (rand_nums[i] - mean);
	}

	double global_sq_diff;
	MPI_Reduce(&local_sq_diff, &global_sq_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (world_rank == 0) {
		float stddev = std::sqrt(global_sq_diff / (num_elements_per_proc * world_size));
		std::cout << "mean = " << mean << " std = " << stddev << std::endl;
	}

	delete[] rand_nums;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
