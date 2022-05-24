#include <thrust/device_vector.h>

struct TemporalBlock {
    long* target_vertices;
    long* edges;
    float* timestamps;
    long num_neighbors;
    TemporalBlock* next;
};


int main() {
    thrust::device_vector<TemporalBlock> blocks;
}
