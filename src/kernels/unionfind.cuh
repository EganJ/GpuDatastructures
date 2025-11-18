#ifndef GPUDS_KERNELS_UNIONFIND_CU
#define GPUDS_KERNELS_UNIONFIND_CU

namespace gpuds::unionfind
{

    __device__ int get_class_readonly(const int *classes, int i);
    __device__ int get_class(int *classes, int i);
    __device__ void union_classes(int *classes, int a, int b);

    __global__ void kernel_mass_merge(int *classes, const int *a, const int *b, int n);
    __global__ void kernel_flatten(int *classes, int n);
} // namespace gpuds::unionfind



#endif