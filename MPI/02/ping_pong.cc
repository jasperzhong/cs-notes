#include <chrono>
#include <iostream>
#include <mpich/mpi.h>
#include <thread>

int main(int argc, char* argv[])
{
	constexpr int PING_PONG_COUNT = 10;
	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	if (world_size != 2) {
		std::cerr << "World size must be 2" << std::endl;
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int ping_pong_count = 0;
	int partner_rank = (world_rank + 1) % 2;
	while (ping_pong_count < PING_PONG_COUNT) {
		if (world_rank == ping_pong_count % 2) {
			ping_pong_count++;
			MPI_Send(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
			std::cout << world_rank << " sent and incremented ping_pong_count " << ping_pong_count << " to " << partner_rank << std::endl;
		} else {
			MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			std::cout << world_rank << " received ping_pong_count " << ping_pong_count << " from " << partner_rank << std::endl;
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
	MPI_Finalize();
}
