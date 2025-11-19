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

// TODO replace this with a real implementation.
struct MockHashTable
{
    // Concurrent with deletes and inserts,
    // but may of course return a stale result.
    // Returns the node_id of the found node, or -1 if not found.
    __device__ int lookup(const FuncNode &node) { return 0; }

    /**
     * Psuedocode for a lookup that commutes only with insert that doesn't need atomics:
     */

    // Inserts the node. Concurrent with lookups but not with deletes.
    // Concurrent with other inserts: may not insert duplicates, and if
    // a duplicate is found, returns false without inserting. In such
    // a case, out_node_id is set to the existing node's ID.
    __device__ bool insert_lookup(const FuncNode &node, int node_id, int &out_node_id) { return false; }

    // Deletes the node. Concurrent with lookups but not with inserts.
    __device__ void remove(const FuncNode &node) {}
};

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

    BlockedList class_to_nodes[MAX_CLASSES + 1];
    BlockedList class_to_parents[MAX_CLASSES + 1];  
    BlockedList staged_merges {&list_space_cursor, (unsigned) -1, (unsigned) -1};

    // Keep these two contiguous! list_space must immediatly
    // follow list_space_cursor in memory.
    BlockedListBuffer list_space_cursor;
    char list_space[MAX_LIST_SPACE];

    // TODO hashconsing structure
    HashTable hashcons; // TODO swap out mock_hashcons with this

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

    __device__ void stageMergeClasses(int class1, int class2)
    {
        ListNode *ln = list_space_cursor.allocateBlock(sizeof(int) * 2);
        ClassesToMerge* m = (ClassesToMerge*) &ln->data[0];

        int lesser;
        int greater;

        if (class1 < class2)
        {
            lesser = class1;
            greater = class2;
        }
        else
        {
            lesser = class2;
            greater = class1;
        }

        m->firstClassID = lesser;
        m->secondClassID = greater;
        addToList(&staged_merges, ln);
    }
};

__host__ void initialize_egraph(EGraph *egraph, const std::vector<FuncNode> &host_nodes,
                                const std::vector<int> &roots,
                                std::vector<int> &compressed_roots);

#endif
