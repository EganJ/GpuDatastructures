#include "datastructures.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

using namespace gpuds;

// Declaration of device launcher from .cu file
void launchVectorAdd(const float *d_a, const float *d_b, float *d_c, int n);

void gpuds::vectorAdd(const float *h_a, const float *h_b, float *h_c, int n) {
  size_t size = n * sizeof(float);

  float *d_a, *d_b, *d_c;
  if (cudaMalloc(&d_a, size) != cudaSuccess ||
      cudaMalloc(&d_b, size) != cudaSuccess ||
      cudaMalloc(&d_c, size) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed");
  }

  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

  // Call kernel launcher
  launchVectorAdd(d_a, d_b, d_c, n);

  cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
