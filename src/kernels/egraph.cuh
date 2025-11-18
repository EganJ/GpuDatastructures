#ifndef GPUDS_KERNELS_EGRAPH_CUH
#define GPUDS_KERNELS_EGRAPH_CUH

#include <cuda_runtime.h>
#include <vector>
#include <map>

#include "../rules.h"

#include "unionfind.cuh"
#include "linkedlist.cuh"
#include "const_params.cuh"


struct EGraph
{
    int num_classes = 0; // Counter for used class slots
    int num_nodes = 0;   // Counter for used node slots

    // Nodes storage only for nodes in the e-graph
    FuncNode node_space[MAX_NODES];
    int node_to_class[MAX_NODES];

    // Union-find structure to manage classes
    int class_ids[MAX_CLASSES + 1]; // +1 because 0 is discarded by unionfind

    BlockedList class_to_nodes[MAX_CLASSES + 1];
    BlockedList class_to_parents[MAX_CLASSES + 1];

    // Keep these two contiguous! list_space must immediatly
    // follow list_space_cursor in memory.
    BlockedListBuffer list_space_cursor;
    char list_space[MAX_LIST_SPACE];

    // TODO hashconsing structure

    __device__ FuncNode &getNode(int node_id)
    {
        return node_space[node_id];
    }

    __device__ int getNumNodes()
    {
        return num_nodes;
    }

    __device__ int resolveClassReadOnly(int class_id)
    {
        return gpuds::unionfind::get_class_readonly(class_ids, class_id);
    }

    __device__ int resolveClass(int class_id)
    {
        // TODO revisit concurrency correctness vs get_class_readonly.
        return gpuds::unionfind::get_class(class_ids, class_id);
    }

    __device__ int getClassOfNode(int node_id)
    {
        return resolveClassReadOnly(node_to_class[node_id]);
    }
};

__host__ void initialize_egraph(EGraph *egraph, const std::vector<FuncNode> &host_nodes,
                                const std::vector<int> &roots,
                                std::vector<int> &compressed_roots);

#endif
