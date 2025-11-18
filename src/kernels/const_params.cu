#include "const_params.cuh"

__constant__ char func_arg_counts[sizeof(func_operand_count) / sizeof(func_operand_count[0])];

__device__ int getFuncArgCount(const FuncName &fname)
{
    return (int)func_arg_counts[(int)fname];
}

__host__ void setArgCounts()
{
    cudaMemcpyToSymbol(func_arg_counts, func_operand_count,
                       sizeof(func_operand_count));
}
