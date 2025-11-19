#ifndef GPUDS_KERNELS_UTILS_CUH
#define GPUDS_KERNELS_UTILS_CUH

#include <cuda_runtime.h>
#include "../rules.h"


__device__ bool operator==(const FuncNode &a, const FuncNode &b);

#endif
