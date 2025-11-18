#include "eqsat.cuh"
#include "const_params.cuh"

#include "../rules.h"

using namespace gpuds;
using namespace gpuds::eqsat;

/**
 * Host-side code to initialize memory structures needed for eqsat.
 */
void initialize_eqsat_memory()
{
    // func_operand_counts (device) to func_arg_counts (host)
    cudaMemcpyToSymbol(
        func_arg_counts,
        func_operand_count,
        sizeof(func_operand_count));
}


// class RuleMatchIterator {
//     int current_index[MAX_RULE_TERMS];
//     int current_eclass_ids[MAX_RULE_TERMS];
//     int current_var_bindings
// };

/**
 * Since a rule may match multiple times depending on choice of bindings,
 * or initial choice of bindings may determine whether a rule can match
 * at all, we allow for a fixed number of potential bindings.
 *
 * Inputs:
 *  - egraph: egraph context
 *  - pattern: funcnode from the global ruleset representing the pattern to match
 *  - enode_idx: index of the enode to attempt to match against
 *  - out_bindings: array to write extended bindings to. Size MULTIMATCH_LIMIT.
 * Returns:
 * - number of entries written to out_bindings
 */
// __device__ unsigned count_matches_enode(EGraph &egraph, const FuncNode &pattern, unsigned enode_idx,
//                              RuleMatch *out_bindings)
// {
//     FuncNode enode = egraph.getNode(enode_idx);

//     if (enode.name != pattern.name)
//     {
//         return 0;
//     }

//     if (enode.name == FuncName::Const)
//     {
//         // For constants, we require exact match of value.
//         if (enode.args[0] != pattern.args[0])
//         {
//             return 0;
//         }
//         return copy_available_to_space<RuleMatch>(in_bindings, in_bindings_count, out_bindings, out_bindings_space);
//     }

//     RuleMatch binds[MAX_FUNC_ARGS][MULTIMATCH_LIMIT];
//     unsigned bind_counts[MAX_FUNC_ARGS];

//     // Project structurally down, collecting possible bindings to merge later.
//     int argc = func_arg_counts[static_cast<int>(pattern.name)];
//     for (int i = 0; i < argc; i++)
//     {
//         bind_counts[i] = count_matches_eclass(egraph, pattern, enode.args[i], binds[i]);
//     }

//     // Now we have a (subset of the) possible bindings for each structural child.
//     // Need to see if any combinations of these are compatible.
// }

__global__ void gpuds::eqsat::kernel_eqsat_match_rules(EqSatSolver *solver)
{
    RuleMatch local_matches[N_LOCAL_MATCH_BUFF];

    // Each node is owned by exactly one block.
    int n_nodes = solver->egraph.getNumNodes();
    int nodes_per_block = (n_nodes + gridDim.x - 1) / gridDim.x;
    int start_node = blockIdx.x * nodes_per_block;
    int end_node = min(n_nodes, start_node + nodes_per_block);

    // For each node, each thread covers one rule.
    int my_thread = threadIdx.x;
    if (my_thread >= N_RULES)
        return;

    FuncNode my_rule = global_ruleset.rule_nodes[my_thread];
    for (int node_idx = start_node; node_idx < end_node; node_idx++)
    {
        // TODO we can retrieve our slice of nodes into local beforehand,
        // to avoid fetching per rule.
        const FuncNode node = solver->egraph.getNode(node_idx);
        RuleMatch match;
        // if (matches_pattern(solver->egraph, node, my_rule, match))
        // {
        //     // TODO
        // }
    }
}

void gpuds::eqsat::launch_eqsat_match_rules(EqSatSolver *solver)
{
    int blockSize = 32 * ((N_RULES / 32) + 1); // At least 1 thread per rule and as tight as possible.
    int numBlocks = 512;                      // TODO tune. Can this be num_nodes or something?
    kernel_eqsat_match_rules<<<numBlocks, blockSize>>>(solver);
    cudaDeviceSynchronize();
}