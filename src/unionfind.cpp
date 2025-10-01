#include "datastructures.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Declaration of unionfind device functions from .cu file
namespace gpuds::unionfind
{
    void mass_merge(int *classes, const int *a, const int *b, int n);

    UnionFind::UnionFind(int n) : num_elements(n)
    {
        if (n <= 0)
        {
            throw std::invalid_argument("UnionFind size must be positive");
        }
        int arr_size = (n + 1) * sizeof(int); // index 0 is unused for simplicity.
        if (cudaMalloc(&cuda_class_array, arr_size) != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed");
        }
        // Inititalize class array to 0 to indicate each element is its own class
        cudaMemset(cuda_class_array, 0, arr_size);
    }

    UnionFind::~UnionFind()
    {
        cudaFree(cuda_class_array);
    }

    int *cuda_malloc_vector(std::vector<int> &vec)
    {
        int *d_vec;
        size_t arr_size = vec.size() * sizeof(int);
        if (cudaMalloc(&d_vec, arr_size) != cudaSuccess)
        {
            throw std::runtime_error("cudaMalloc failed");
        }
        cudaMemcpy(d_vec, vec.data(), arr_size, cudaMemcpyHostToDevice);
        return d_vec;
    }

    void UnionFind::massMerge(std::vector<int> &a, std::vector<int> &b)
    {
        if (a.size() != b.size())
        {
            throw std::invalid_argument("UnionFind::massMerge input vectors must be the same size");
        }

        int *d_a = cuda_malloc_vector(a);
        int *d_b = cuda_malloc_vector(b);

        // TODO made up kernel launch parameters.
        gpuds::unionfind::mass_merge(cuda_class_array, d_a, d_b, a.size());
        cudaDeviceSynchronize();

        cudaFree(d_a);
        cudaFree(d_b);
    }

    std::vector<int> UnionFind::getClasses()
    {
        std::vector<int> out_classes(num_elements);
        size_t arr_size = out_classes.size() * sizeof(int);
        cudaMemcpy(out_classes.data(), cuda_class_array + 1, arr_size, cudaMemcpyDeviceToHost);
        return out_classes;
    }
}