#include "utils.cuh"
#include "const_params.cuh"

__device__ bool operator==(const FuncNode &a, const FuncNode &b)
{
    if (a.name != b.name)
        return false;
    int argc = getFuncArgCount(a.name);
    for (int i = 0; i < argc; i++)
    {
        if (a.args[i] != b.args[i])
            return false;
    }
    return true;
}