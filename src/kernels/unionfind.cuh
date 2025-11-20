#ifndef GPUDS_KERNELS_UNIONFIND_CU
#define GPUDS_KERNELS_UNIONFIND_CU

namespace gpuds::unionfind
{

    __device__ int get_class_readonly(const int *classes, int i);
    __device__ int get_class(int *classes, int i);
    __device__ void merge(int *classes, int a, int b);

    /**
     * Merges classes containing a and b, returning the old root of either a or b that is atomically
     * no longer a root, as well as the new root (which may be stale). Don't care if the root used to
     * belong to a or b, simply that it was a non-stale root before the merge and will never be a root again.
     *
     * Returns -1 if a and b were already in the same class, and therefore finding
     * an old root is impossible. In this case old_root is not modified.
     */
    __device__ int atomic_merge_and_get_old_root(int *classes, int a, int b, int &old_root);

    __global__ void kernel_mass_merge(int *classes, const int *a, const int *b, int n);
    __global__ void kernel_flatten(int *classes, int n);
    // Host functions to launch the kernels
    void mass_merge(int *classes, const int *a, const int *b, int n);
    void flatten(int *classes, int n);
} // namespace gpuds::unionfind

#endif