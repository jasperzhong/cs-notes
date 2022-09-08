__global__ void lower_bound(int N, const float *x, float target, int *idx) {
  int left = 0, right = N;
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (x[mid] < target) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  *idx = left;
}

void launch_kernel(int N, const float *x, float target, int *idx) {
  lower_bound<<<1, 1>>>(N, x, target, idx);
  cudaDeviceSynchronize();
}
