__global__ void add(int n, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride) 
        y[i] += x[i];
}

int main(void) {
    int N = 1 << 20;
    float *x, *y;

    cudaHostAlloc(&x, N*sizeof(float), 0);
    cudaHostAlloc(&y, N*sizeof(float), 0);

    for (int i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    for (int i = 0; i < 3; ++i) {
        add<<<numBlocks, blockSize>>>(N, x, y);
    }

    cudaDeviceSynchronize();
}
