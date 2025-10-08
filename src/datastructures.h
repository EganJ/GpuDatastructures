#ifndef GPU_DATASTRUCTURES_H
#define GPU_DATASTRUCTURES_H

#include <vector>
namespace gpuds
{

    void vectorAdd(const float *a, const float *b, float *c, int n);

    namespace unionfind
    {

        class UnionFind
        {
            int *cuda_class_array;
            int num_elements;

        public:
            UnionFind(int n);
            ~UnionFind();
            void massMerge(std::vector<int> &a, std::vector<int> &b);
            std::vector<int> getClasses();
        };
    }
}

#endif // GPU_DATASTRUCTURES_H