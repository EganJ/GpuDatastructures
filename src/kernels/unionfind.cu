/**
 * A union-find data structure implementation for CUDA.
 *
 * Represented as a (n+1) length array of integers representing n classes
 * (labeled in the range [1, n]) with the 0th element unused. classes[i] is
 * either 0, indicating that i is a root class, or some 0 < j < i pointing further
 * up the chain to the root class.
 *
 * The requirement that j < i ensures that there are no cycles and helps prevent
 * some concurrency issues.
 *
 * Leverages the fact that concurrent writes from a warp are well-defined
 * in that only one of them will succeed, and values will not be mangled:
 * "If a non-atomic instruction executed by a warp writes to the same location
 * in global memory for more than one of the threads of the warp, only one
 * thread performs a write and which thread does it is undefined."  -
 * https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_C_Programming_Guide.pdf
 * section F.4.2.
 * However, this is somewhat contradicted by
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-consistency:
 * "The two threads read and write from the same memory locations X and Y
 * simultaneously. Any data-race is undefined behavior, and has no defined
 * semantics. The resulting values for A and B can be anything."
 */

#include <cstdio>
#include <cuda_runtime.h>

namespace gpuds::unionfind
{
  namespace impl
  {
    __device__ void swap(int &a, int &b)
    {
      int t = a;
      a = b;
      b = t;
    }

    __device__ int ceildiv(int a, int b)
    {
      return (a + b - 1) / b;
    }
  }

  /**
   * A read-only single-scalar version to get the class of an element.
   * In the case that another thread is in the middle of modifying the
   * classes array, the value may returned may be stale or not a root, but
   * will be on a path to the true root. Safe to call in parallel.
   */
  __device__ int get_class_readonly(const int *classes, int i)
  {
    while (classes[i] != 0)
    {
      // Classes should never return to being zero! Otherwise we have a
      // data race here. In this case we may return a stale value but
      // it will be on the path to the root.
      i = classes[i];
    }
    return i;
  }

  /*
   * A read-write single-scalar version to get the class of an element, with path
   * compression. This is safe to call in parallel! Even if multiple threads
   * go to compress the same path and overwrite each other's writes, the end
   * result will still be correct. (Modulo memory tearing / value mangling as noted
   * above. We will need to revisit this if the undefined behavior is indeed the case.)
   * This value may be immediately stale.
   *
   * Case to consider: two threads fetch the readonly root. In between, the
   * root is modified by another thread so that on thread has a stale value.
   * Then both threads try to path-compress. We may possibly have a mix of
   * stale and new values along the path, but pointing to the stale value
   * is still correct as the stale value should also point to the new root (eventually).
   *
   * When no other threads are modifying the classes array, this function will return
   * the true root and compress the path fully (as any compression conflicts will also
   * be writing the same root value).
   */
  __device__ int get_class(int *classes, int i)
  {
    int root = get_class_readonly(classes, i);
    // Due to race conditions, the only thing we can say at this point is that
    // root <= i, and that root eventually points to the same final class as i (though
    // not necessarily on the same chain).
    int old;
    do
    {
      old = atomicMin(classes + i, root); // can we safely un-atomic this? are there performance benefits?
      i = old;
    } while (old > root);

    return root;
  }

  /**
   * Merges max(a, b) into min(a, b).
   *
   * Threads calling merge(a, b) and merge(b, a) is safe since we
   * always merge the higher into the lower.
   *
   * Problem: let a < b < c be root classes.
   * Then merge(a, c) and merge(b, c) is difficult to resolve:
   * classes[c] will try to point to both a and b. In sequence this
   * resolves by c --> b --> a.
   */
  __device__ void merge(int *classes, int a, int b)
  {
    a = get_class(classes, a);
    do
    {
      b = get_class(classes, b);
      if (a == b)
      {
        return;
      }
      else if (a > b)
      {
        impl::swap(a, b);
      } // Ensure a < b
      // Now we want to merge b into a, but only if b is still a root.
      // If another thread has already merged b into something else, we need to
      // to try again with the new root- which may now be lower than a, in which
      // case to maintain the invariant we need to swap them. Then a may be
      // a stale root, so we need to loop. Notice that the class merging into
      // can safely be stale, and will be resolved the next time we try to get the class.
      // Safety can be seen as follows: a < b given locally, and b remains a root
      // from atomicity, so this mantains all our invariants. Liveness is a little more
      // dubious.
    } while (0 != atomicCAS(classes + b, 0, a));
  }

  /*
   * A kernel that takes a union-find class array and two arrays of equal length
   * containing the indices to merge. If there are k threads across all blocks,
   * then each does n/k merges.
   */
  __global__ void kernel_mass_merge(int *classes, const int *a, const int *b, int n)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int work = impl::ceildiv(n, gridDim.x * blockDim.x);
    int start = idx * work;
    int end = min(n, start + work);
    for (int i = start; i < end; i++)
    {
      merge(classes, a[i], b[i]);
    }
  }

  void mass_merge(int *classes, const int *a, const int *b, int n)
  {
    int blockSize = 32;
    // int numBlocks = (n + blockSize - 1) / blockSize;
    int numBlocks = 256;
    kernel_mass_merge<<<numBlocks, blockSize>>>(classes, a, b, n);
    cudaDeviceSynchronize();
  }

  // After all merges are done, call flatten to make all elements point directly
  // to their root class. (Helps with checking correctness and may be useful for
  // optimizing runtime after all merges are done.)
  __global__ void kernel_flatten(int *classes, int n)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int work = impl::ceildiv(n, gridDim.x * blockDim.x);
    int start = 1 + idx * work;
    int end = min(1 + n, start + work);
    for (int i = start; i < end; i++)
    {
      int val = get_class_readonly(classes, i);
      classes[i] = (val == i) ? 0 : val;
    }
  }

  void flatten(int *classes, int n)
  {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    kernel_flatten<<<numBlocks, blockSize>>>(classes, n);
    cudaDeviceSynchronize();
  }

} // namespace gpuds::unionfind