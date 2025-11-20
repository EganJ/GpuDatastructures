#ifndef GPUDS_KERNELS_EGRAPH_CUH
#define GPUDS_KERNELS_EGRAPH_CUH

#include <cuda_runtime.h>
#include <vector>
#include <map>

#include "../rules.h"

#include "unionfind.cuh"
#include "linkedlist.cuh"
#include "const_params.cuh"
#include "linearprobing.h"

struct ClassesToMerge
{
    int firstClassID;
    int secondClassID;
};

struct EGraph
{
    int num_classes = 0; // Counter for used class slots
    int num_nodes = 0;   // Counter for used node slots

    // Nodes storage only for nodes in the e-graph
    FuncNode node_space[MAX_NODES];
    int node_to_class[MAX_NODES];

    // Union-find structure to manage classes
    int class_ids[MAX_CLASSES + 1]; // +1 because 0 is discarded by unionfind
    ClassesToMerge classes_to_merge[MAX_MERGE_LIST_SIZE];
    int classes_to_merge_count;

    BlockedList class_to_nodes[MAX_CLASSES + 1];
    BlockedList class_to_parents[MAX_CLASSES + 1];  

    // Keep these two contiguous! list_space must immediatly
    // follow list_space_cursor in memory.
    BlockedListBuffer list_space_cursor;
    char list_space[MAX_LIST_SPACE];

    HashTable hashcons;

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

    // Bound-checked version of resolveClassReadOnly
    __device__ int resolveClassReadOnlySafe(int class_id)
    {
        if (class_id <= 0 || class_id > num_classes)
        {
            return -1; // invalid class
        }
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

    /**
     * Inserts the given node into the e-graph. If an equivalent node already exists (or 
     * is concurrently inserted), output its node and class IDs instead, otherwise output
     * the newly inserted node and class IDs. Returns whether a new node was inserted.
     * @param node The node to insert.
     * @param class_id The class ID to insert the node into (only relevant for non-vars/consts).
     * @param out_node_id Output parameter for the node ID of the inserted/found node.
     * @param out_class_id Output parameter for the class ID of the inserted/found node.
     * 
     * Warning: frequent calls to this with duplicates may exhaust the nodespace. Should be filtered
     * by a prior lookup, even if that lookup may be stale.
     */
    __device__ bool insertNode(const FuncNode &node, int class_id, int &out_node_id);

    __device__ void stageMergeClasses(int class1, int class2);
};

__host__ void initialize_egraph(EGraph *egraph, const std::vector<FuncNode> &host_nodes,
                                const std::vector<int> &roots,
                                std::vector<int> &compressed_roots);

#endif
