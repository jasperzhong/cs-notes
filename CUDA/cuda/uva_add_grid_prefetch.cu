__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) 
        y[i] += x[i];
}

int main(void) {
    int N = 1 << 20;
    float *x, *y;

    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // prefetch the data to the GPU
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(x, N*sizeof(float), device, NULL);
    cudaMemPrefetchAsync(y, N*sizeof(float), device, NULL);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    for (int i = 0; i < 3; ++i) {
        add<<<numBlocks, blockSize>>>(N, x, y);
    }

    cudaDeviceSynchronize();
}
