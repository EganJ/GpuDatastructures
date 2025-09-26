#include "datastructures.h"
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;

  const int N = 1024;
  float a[N], b[N], c[N];

  // Initialize vectors
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(i);
  }

  // Call GPU vector addition
  gpuds::vectorAdd(a, b, c, N);

  // Print a few results
  for (int i = 0; i < 10; ++i) {
    std::cout << "c[" << i << "] = " << c[i] << std::endl;
  }

  return 0;
}