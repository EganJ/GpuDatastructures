#include <cuda_runtime.h>
#include <stdio.h>

__global__ void allocate1MB() {
    // Allocate once (by a single thread) to avoid excessive heap usage
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        size_t bytes = 1024 * 1024; // 1 MB
        void* ptr = malloc(bytes);
        if (ptr == nullptr) {
            printf("Device malloc failed.\n");
            return;
        }

        // Example: touch the memory to ensure it's usable
        unsigned char* p = static_cast<unsigned char*>(ptr);
        for (size_t i = 0; i < bytes; i += 4096) { // stride to reduce runtime
            p[i] = 0xAB;
        }

        // Free the allocation
        free(ptr);
        printf("Allocated and freed 1 MB on device.\n");
    }
}

int main() {
    // Increase device heap to allow device-side malloc
    size_t heapSize = 1024 * 1024 * 1024; // 8 MB (adjust as needed)
    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSetLimit failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch a small grid; only one thread will allocate
    allocate1MB<<<1, 32>>>();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Done.\n");
    return 0;
}
