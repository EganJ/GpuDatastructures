#include <cstdio>
#include <cuda_runtime.h>

// A simple vector addition kernel
__global__ void vectorAddKernel(const float *a, const float *b, float *c,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// Host-callable wrapper
void launchVectorAdd(const float *d_a, const float *d_b, float *d_c, int n) {
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  vectorAddKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
  cudaDeviceSynchronize();
}
