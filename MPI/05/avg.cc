#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mpich/mpi.h>

float* create_rand_nums(int num_elements)
{
	float* rand_nums = new float[num_elements];
	for (int i = 0; i < num_elements; ++i) {
		rand_nums[i] = (rand() / (float)RAND_MAX);
	}
	return rand_nums;
}

float compute_avg(float* array, int num_elements)
{
	double sum = 0.0;
	for (int i = 0; i < num_elements; ++i) {
		sum += array[i];
	}
	return sum / num_elements;
}

int main(int argc, char* argv[])
{
	if (argc != 2) {
		std::cerr << "Usage: avg num_elements_per_proc" << std::endl;
		exit(1);
	}

	int num_elements_per_proc = atoi(argv[1]);

	srand(time(NULL));

	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	float* rand_nums = NULL;
	if (world_rank == 0) {
		rand_nums = create_rand_nums(num_elements_per_proc * world_size);
	}

	float* sub_rand_nums = new float[num_elements_per_proc];

	MPI_Scatter(rand_nums, num_elements_per_proc, MPI_FLOAT, sub_rand_nums, num_elements_per_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);

	float sub_avg = compute_avg(sub_rand_nums, num_elements_per_proc);

	float* sub_avgs = NULL;
	if (world_rank == 0) {
		sub_avgs = new float[world_size];
	}

	MPI_Gather(&sub_avg, 1, MPI_FLOAT, sub_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	if (world_rank == 0) {
		float avg = compute_avg(sub_avgs, world_size);
		std::cout << "Avg of all elements is " << avg << std::endl;
		float original_data_avg = compute_avg(rand_nums, num_elements_per_proc * world_size);
		std::cout << "Avg computed across original data is " << original_data_avg << std::endl;
	}

	if (world_rank == 0) {
		delete[] rand_nums;
		delete[] sub_avgs;
	}
	delete[] sub_rand_nums;

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}
