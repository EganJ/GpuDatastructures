#include "egraph.cuh"
#include "const_params.cuh"
#include <stdio.h>
#include <iostream>

#include "../parser.h" // for print expression, delete as soon as possible

__global__ void initialize_empty_lists(EGraph *egraph)
{
    int n_threads = blockDim.x * gridDim.x;
    int total_classes = MAX_CLASSES + 1; // including class 0 which is unused
    int classes_per_thread = (total_classes + n_threads - 1) / n_threads;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int start_class = tid * classes_per_thread;
    int end_class = min(total_classes, start_class + classes_per_thread);
    for (int class_id = start_class; class_id < end_class; class_id++)
    {
        egraph->class_to_nodes[class_id].initalize(&egraph->list_space_cursor);
        egraph->class_to_parents[class_id].initalize(&egraph->list_space_cursor);
    }
    if (tid == 0)
    {
        egraph->list_space_cursor.buffer_size = MAX_LIST_SPACE;
        egraph->list_space_cursor.buffer_allocated = 0;
    }
}

__global__ void initialize_class_to_nodes_list_values(EGraph *egraph)
{
    const auto single_int_block = sizeof(int) + sizeof(ListNode);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid == 0)
    {
        // Reserve space for all nodes for class_to_nodes
        assert(egraph->list_space_cursor.buffer_allocated == 0);
        egraph->list_space_cursor.buffer_allocated += single_int_block * (egraph->num_nodes + 1);
    }

    // Handle class to nodes mapping
    int blocks_per_thread = (egraph->num_classes + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int start_class = tid * blocks_per_thread;
    int end_class = min(egraph->num_classes, start_class + blocks_per_thread);
    for (int class_id = start_class + 1; class_id < end_class + 1; class_id++)
    {
        int idx_to_listnode = (class_id - 1) * single_int_block;
        ListNode *link = (ListNode *)(&egraph->list_space[idx_to_listnode]);
        link->block_size = (unsigned)(sizeof(int));
        link->next_node = NULL_ID;
        ((int *)link->data)[0] = class_id - 1; // Nodes start at 0, classes at 1.
        addToList(&egraph->class_to_nodes[class_id], link);
    }
}

__global__ void initialize_class_to_parents_list_values(EGraph *egraph)
{
    // For each node (currently synonymous with class), add outgoing edges.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blocks_per_thread = (egraph->num_nodes + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int start_node = tid * blocks_per_thread;
    int end_node = min(egraph->num_nodes, start_node + blocks_per_thread);
    for (int node_id = start_node; node_id < end_node; node_id++)
    {
        int parent_class_id = egraph->getClassOfNode(node_id);
        FuncNode &node = egraph->getNode(node_id);
        int argc = getFuncArgCount(node.name);
        if (node.name != FuncName::Var && node.name != FuncName::Const)
        {
            for (int i = 0; i < argc; i++)
            {
                int child_node_id = node.args[i];
                int child_class_id = egraph->getClassOfNode(child_node_id);
                // promote the node to point to the eclass
                node.args[i] = child_class_id;
                // add parent link from child class to this node's class
                ListNode *block = egraph->list_space_cursor.allocateBlock(sizeof(int));
                ((int *)block->data)[0] = parent_class_id;
                addToList(&egraph->class_to_parents[child_class_id], block);
            }
        }
    }
}

__global__ void set_num_classes_and_nodes(EGraph *egraph, int n_initial_nodes)
{
    egraph->num_nodes = n_initial_nodes;
    egraph->num_classes = n_initial_nodes; // initially each node in its own class
    // While we're here, initialize the hashcons table
    egraph->hashcons.parent_egraph = egraph;
}

__global__ void insert_base_graph_to_hashcons(EGraph *egraph, int n_initial_nodes)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    assert(n_initial_nodes <= blockDim.x * gridDim.x); // ensure enough threadss
    if (tid >= n_initial_nodes)
        return;
    int added_val;
    egraph->hashcons.insert(egraph->node_space[tid], tid, added_val);
    assert(added_val == tid);
}

// Steps that must be performed on the device
__host__ void kernel_initialize_egraph(EGraph *egraph, int n_initial_nodes)
{
    set_num_classes_and_nodes<<<1, 1>>>(egraph, n_initial_nodes);
    initialize_empty_lists<<<512, 32>>>(egraph);
    initialize_class_to_nodes_list_values<<<512, 32>>>(egraph);
    initialize_class_to_parents_list_values<<<512, 32>>>(egraph);
    insert_base_graph_to_hashcons<<<512, 32>>>(egraph, n_initial_nodes);
}

__host__ void initialize_egraph(EGraph *egraph, const std::vector<FuncNode> &host_nodes,
                                const std::vector<int> &roots, std::vector<int> &compressed_roots)
{
    std::vector<FuncNode> compressed_nodes;
    compress_nodespace(host_nodes, roots, compressed_nodes, compressed_roots);

    printf("Expressions, compressed:\n");
    for (int r : compressed_roots)
    {
        printf("%s\n", printExpression(compressed_nodes, r).c_str());
    }

    std::vector<int> class_ids(compressed_nodes.size());
    for (int i = 0; i < class_ids.size(); i++)
    {
        class_ids[i] = i + 1; // each node in its own class, and class ID's skip 0.
    }

    cudaMemcpy(&egraph->node_space[0], compressed_nodes.data(),
               sizeof(FuncNode) * compressed_nodes.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(&egraph->node_to_class[0], class_ids.data(),
               sizeof(int) * class_ids.size(),
               cudaMemcpyHostToDevice);
    cudaMemset(&egraph->class_ids[0], 0, sizeof(int) * (MAX_CLASSES + 1));
    cudaMemset(&egraph->list_space[0], 0, sizeof(char) * MAX_LIST_SPACE);

    initialize_hashcons_memory(&egraph->hashcons);

    // Launch kernel to initialize egraph structures that we can't from host.
    kernel_initialize_egraph(egraph, compressed_nodes.size());
    cudaDeviceSynchronize();
}

/**
 * Should not be done concurrently with anything that accesses the member lists of the eclasses!
 */
__device__ bool EGraph::insertNode(const FuncNode &node, int class_id, int &out_node_id)
{
    // Allocate new nodespace.
    int node_id = atomicAdd(&num_nodes, 1);
    node_space[node_id] = node;

    if (class_id == -1)
    {
        // Create new class
        class_id = atomicAdd(&num_classes, 1) + 1; // +1 because class 0 is unused
    }
    else
    {
        class_id = resolveClass(class_id);
    }
    node_to_class[node_id] = class_id;

    // Attempt to insert. If failed because of duplicate, mark changes as discarded.
    int existing_node_id = -1;
    bool inserted = hashcons.insert(node, node_id, existing_node_id);

    if (!inserted)
    {
        // Discard changes
        // Cannot decrement num_nodes safely, so just leave the node unused.
        node_to_class[node_id] = -1; // mark as deleted
        node_space[node_id].name = FuncName::Unset;
        out_node_id = existing_node_id;
        return false;
    } 
    out_node_id = node_id;

    // Add to class_to_nodes list. This can be done after we know the insert succeeded,
    // means that other classes cannot access the class_to_nodes lists concurrently or they may
    // not see nodes they should.
    ListNode *block = list_space_cursor.allocateBlock(sizeof(int));
    ((int *)block->data)[0] = node_id;
    addToList(&class_to_nodes[class_id], block);

    // Look at how many of the new node's operands are themselves nodes (not constants or vars)
    int op_count = 0;
    FuncName name = node.name;

    if (name != FuncName::Unset && name != FuncName::Var && name != FuncName::Const)
    {
        op_count = getFuncArgCount(name);
    }
    // For all of the node's node operands, add this as one of their parents
    for (int i = 0; i < op_count; i++)
    {
        int resolved_class = resolveClass(node.args[i]);
        ListNode *ln = list_space_cursor.allocateBlock(sizeof(int));
        *((int *)ln->data) = class_id;
        addToList(&this->class_to_parents[resolved_class], ln);
    }

    return true;
}

__device__ void EGraph::stageMergeClasses(int class1, int class2)
{
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

    int spot = atomicAdd(&(this->classes_to_merge_count), 1);
    ClassesToMerge *m = (ClassesToMerge *)(&this->classes_to_merge[spot]);
    m->firstClassID = lesser;
    m->secondClassID = greater;
}