#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <ctime>

struct TemporalBlock {
    long* target_vertices;
    long* edges;
    float* timestamps;
    long num_edges;
    TemporalBlock* next;
};

__global__ void get_neighbors(long* target_vertices, float* timestamps, long num_sample, TemporalBlock* vertex_table, long num_vertices, long* indices)
{
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sample) {
        long vertex = target_vertices[idx];
        float timestamp = timestamps[idx];
        TemporalBlock block = vertex_table[vertex];
        long left = 0, right = block.num_edges;
        // binary search; find the first edge with timestamp >= timestamp
        while (left < right) {
            long mid = (left + right) / 2;
            if (block.timestamps[mid] < timestamp) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        indices[idx] = left;
    }
}

int main()
{
    int num_vertices = 10;
    thrust::device_vector<TemporalBlock> vertex_table(num_vertices);
    thrust::host_vector<TemporalBlock> vertex_table_host(num_vertices);

    std::srand(std::time(nullptr));

    // add some edges
    for (int i = 0; i < num_vertices; i++) {
        int num_edges = std::rand() % 100;

        long* h_target_vertices = new long[num_edges];
        long* h_edges = new long[num_edges];
        float* h_timestamps = new float[num_edges];
        for (int j = 0; j < num_edges; j++) {
            h_target_vertices[j] = std::rand() % num_vertices;
            h_edges[j] = std::rand() % num_vertices; h_timestamps[j] = std::rand() % 100;
        }
        std::sort(h_timestamps, h_timestamps + num_edges);

        long *d_target_vertices, *d_edges;
        float* d_timestamps;
        cudaMalloc(&d_target_vertices, num_edges * sizeof(long));
        cudaMalloc(&d_edges, num_edges * sizeof(long));
        cudaMalloc(&d_timestamps, num_edges * sizeof(float));

        // copy to device
        cudaMemcpy(d_target_vertices, h_target_vertices, num_edges * sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_edges, h_edges, num_edges * sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_timestamps, h_timestamps, num_edges * sizeof(float), cudaMemcpyHostToDevice);

        // copy "meta" data to device
        vertex_table[i] = TemporalBlock { d_target_vertices, d_edges, d_timestamps, num_edges, nullptr };
        vertex_table_host[i] = TemporalBlock { h_target_vertices, h_edges, h_timestamps, num_edges, nullptr };
    }

    long num_sample = std::rand() % num_vertices;
    thrust::device_vector<long> target_vertices(num_sample);
    thrust::device_vector<float> timestamps(num_sample);
    thrust::device_vector<long> indices(num_sample);

    for (int i = 0; i < num_sample; i++) {
        target_vertices[i] = std::rand() % num_vertices;
        timestamps[i] = std::rand() % 100;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_sample + threadsPerBlock - 1) / threadsPerBlock;
    long* d_target_vertices = thrust::raw_pointer_cast(target_vertices.data());
    float* d_timestamps = thrust::raw_pointer_cast(timestamps.data());
    long* d_indices = thrust::raw_pointer_cast(indices.data());
    TemporalBlock* d_vertex_table = thrust::raw_pointer_cast(vertex_table.data());

    get_neighbors<<<blocksPerGrid, threadsPerBlock>>>(d_target_vertices, d_timestamps, num_sample, d_vertex_table, num_vertices, d_indices);

    cudaDeviceSynchronize();

    // copy back to host
    long* h_indices = new long[num_sample];
    cudaMemcpy(h_indices, d_indices, num_sample* sizeof(long), cudaMemcpyDeviceToHost);

    // print results
    for (int i = 0; i < num_sample; i++) {
        long vertex = target_vertices[i];
        float timestamp = timestamps[i];
        std::cout << vertex << " (given timestamp: " << timestamp << "): ";
        for (int j = 0; j < h_indices[i]; j++) {
            std::cout << vertex_table_host[vertex].timestamps[j] << " ";
        }
        std::cout << "(" << timestamps[i] << ") ";
        for (int j = h_indices[i]; j < vertex_table_host[vertex].num_edges; j++) {
            std::cout << vertex_table_host[vertex].timestamps[j] << " ";
        }
        std::cout << std::endl;
    }
}
