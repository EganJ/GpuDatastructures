#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

#include "eqsat.cuh"
#include "const_params.cuh"

#include "../rules.h"

using namespace gpuds;
using namespace gpuds::eqsat;
namespace gpuds::eqsat
{
    __constant__ Ruleset global_ruleset;

    /**
     * Host-side code to initialize memory structures needed for eqsat.
     */
    __host__ void initialize_ruleset_on_device(std::vector<FuncNode> &rule_nodes_host, std::vector<Rule> &rules_host)
    {
        assert(rule_nodes_host.size() <= MAX_RULESET_TERMS);
        assert(rules_host.size() <= N_RULES);

        // rule_nodes_host to global_ruleset.rule_nodes (device)
        cudaMemcpyToSymbol(
            global_ruleset,
            rule_nodes_host.data(),
            rule_nodes_host.size() * sizeof(FuncNode),
            offsetof(Ruleset, rule_nodes),
            cudaMemcpyHostToDevice);

        // rules_host to global_ruleset.rules (device)
        cudaMemcpyToSymbol(
            global_ruleset,
            rules_host.data(),
            rules_host.size() * sizeof(Rule),
            offsetof(Ruleset, rules),
            cudaMemcpyHostToDevice);
    }

    __host__ void initialize_eqsat_memory()
    {
        setArgCounts();
    }

    __global__ void initialize_eqsat_solver(EqSatSolver *solver)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
            solver->n_rule_matches = 0;
        }
    }

    __host__ EqSatSolver *construct_eqsat_solver(const std::vector<FuncNode> node_space_host,
                                                 const std::vector<int> roots_host,
                                                 std::vector<int> &compressed_roots)
    {
        EqSatSolver *d_solver;
        cudaMalloc(&d_solver, sizeof(EqSatSolver));
        initialize_eqsat_solver<<<1, 1>>>(d_solver);
        cudaMemset(d_solver->rule_matches, 0, sizeof(RuleMatch) * MAX_RULE_MATCHES);
        initialize_egraph(&d_solver->egraph, node_space_host, roots_host, compressed_roots);

        return d_solver;
    }

    enum BindRelation
    {
        Equal,        // All bindings are equal.
        LHSWeaker,    // LHS bindings are a subset of RHS bindings.
        RHSWeaker,    // RHS bindings are a subset of LHS bindings.
        Compatible,   // Different set of non-conflicting bindings.
        Incompatible, // Conflicting bindings.
    };

    struct VarBind
    {
        bool bound;
        int bindings[MAX_RULE_VARS];

        __device__ VarBind()
        {
            for (int i = 0; i < MAX_RULE_VARS; i++)
            {
                bindings[i] = -1;
            }
            bound = false;
        }

        __device__ VarBind join_with(const VarBind &other, VarBind &result) const
        {
            for (int i = 0; i < MAX_RULE_VARS; i++)
            {
                if (min(bindings[i], other.bindings[i]) == -1 || bindings[i] == other.bindings[i])
                {
                    result.bound = true;
                    result.bindings[i] = max(bindings[i], other.bindings[i]);
                }
                else
                {
                    result.bindings[i] = -1;
                }
            }
            return result;
        }

        __device__ BindRelation compare(const VarBind &other) const
        {
            bool lhs_subset = true; // non-strict subsets. If both true, equal.
            bool rhs_subset = true;
            for (int i = 0; i < MAX_RULE_VARS; i++)
            {
                if (bindings[i] == -1)
                {
                    rhs_subset = rhs_subset && (other.bindings[i] == -1);
                }
                if (other.bindings[i] == -1)
                {
                    lhs_subset = lhs_subset && (bindings[i] == -1);
                }
                if (bindings[i] != -1 && other.bindings[i] != -1)
                {
                    if (bindings[i] != other.bindings[i])
                    {
                        return Incompatible;
                    }
                }
            }
            if (lhs_subset && rhs_subset)
            {
                return Equal;
            }
            if (lhs_subset)
            {
                return LHSWeaker;
            }
            if (rhs_subset)
            {
                return RHSWeaker;
            }
            return Compatible;
        }
    };

    struct MultiMatch
    {
        int n_matches;
        VarBind var_binds[MULTIMATCH_LIMIT];

        __device__ static inline MultiMatch nomatch()
        {
            MultiMatch mm;
            mm.n_matches = 0;
            return mm;
        }

        /**
         * Adds a new binding option if it wasn't already present in an equal or weaker form, and there
         * is space. Returns true if anything was added.
         */
        __device__ bool add_binding(const VarBind &new_bind)
        {
            if (n_matches >= MULTIMATCH_LIMIT)
            {
                return false;
            }
            for (int i = 0; i < n_matches; i++)
            {
                BindRelation rel = var_binds[i].compare(new_bind);
                if (rel == Equal || rel == LHSWeaker)
                {
                    // Already have this or a weaker binding.
                    return false;
                }
            }
            var_binds[n_matches] = new_bind;
            n_matches++;
            return true;
        }

        __device__ int add_bindings(const MultiMatch &other)
        {
            int n_added = 0;
            for (int i = 0; i < other.n_matches; i++)
            {
                if (add_binding(other.var_binds[i]))
                {
                    n_added++;
                }
            }
            return n_added;
        }
    };

    __device__ void find_matches_eclass(EGraph &graph, const FuncNode &pattern, int eclass_idx,
                                        const MultiMatch &match_in, MultiMatch &match_out);

    /**
     * Finds mathes from a root pattern to a given enode. Patterns may return multiple matches:
     * this is limited by the MULTIMATCH_LIMIT constant.
     *
     * Parameters:
     *  - graph: The egraph to search in.
     *  - pattern: The pattern to match, a funcnode in the global ruleset nodespace.
     *  - enode_idx: The index of the enode to match against.
     *  - match_in: Existing variable bindings to respect.
     *  - match_out: Will append found matches here.
     *
     * Returns: The number of new matches added to match_out.
     */
    __device__ int find_matches_enode(EGraph &graph, const FuncNode &pattern, int enode_idx,
                                      const MultiMatch &match_in, MultiMatch &match_out)
    {
        MultiMatch result = MultiMatch::nomatch();
        FuncNode enode = graph.getNode(enode_idx);

        // If pattern is a variable, we should be in find_matches_eclass instead.
        // If pattern is a constant, match only if enode is the same constant.
        // If pattern is a func, need to recursively check structural match and bindings.
        if (pattern.name == FuncName::Var)
        {
            // Hint: patterns should not be rooted with Var, so this should not be the first
            // visit, and subsequent visits should catch Var subterms before calling here.
            assert(false && "Should not reach here with Var pattern");
        }
        else if (pattern.name == FuncName::Const)
        {
            if (enode.name != FuncName::Const || pattern.args[0] != enode.args[0])
            {
                return 0;
            }
            else
            {
                return match_out.add_bindings(match_in);
            }
        }
        else if (pattern.name != enode.name)
        {
            return 0; // Different function names, no structural match.
        }

        // Have two op node with same name. Check arity.
        unsigned char n_args = getFuncArgCount(pattern.name);

        MultiMatch m1 = match_in;
        MultiMatch m2;
        bool rotate = false;
        for (int arg = 0; arg < n_args; arg++)
        {
            MultiMatch &m_current = (rotate ? m2 : m1);
            MultiMatch &m_next = (rotate ? m1 : m2);
            m_next.n_matches = 0;

            // Give current bindings to subterm and determine subsequent bindings.
            FuncNode pattern_arg = global_ruleset.rule_nodes[pattern.args[arg]];
            int eclass_id = graph.resolveClassReadOnly(enode.args[arg]);
            find_matches_eclass(graph, pattern_arg, eclass_id, m_current, m_next);
            rotate = !rotate;
        }
        int new_matches = 0;
        auto &subterm_matches = (rotate ? m2 : m1);
        for (int i = 0; i < subterm_matches.n_matches; i++)
        {
            if (match_out.add_binding(subterm_matches.var_binds[i]))
            {
                new_matches++;
            }
        }
        return new_matches;
    }

    /**
     * Finds matches from a root pattern to a given eclass. This is where the multimatch
     * possibilities stem from: different choices of nodes within the eclass l
lead to different
     * var bindings, with effects seen in other branches if vars repeat more than once.
     *
     * Parameters:
     *  - graph: The egraph to search in.
     *  - pattern: The pattern to match, a funcnode in the global ruleset nodespace.
     *  - eclass_idx: The index of the eclass to match against.
     *  - match_in: Existing variable binding possibliities.
     *  - match_out: Will append found matches here. Each found match will be equal or stronger than a
     *   match in match_in.
     */
    __device__ void find_matches_eclass(EGraph &graph, const FuncNode &pattern, int eclass_idx,
                                        const MultiMatch &match_in, MultiMatch &match_out)
    {
        // If pattern is a var, we can pass through any candidate bindings that are unbound or match.
        if (pattern.name == FuncName::Var)
        {
            int var_id = pattern.args[0];
            for (int i = 0; i < match_in.n_matches; i++)
            {
                VarBind candidate_bind = match_in.var_binds[i];
                if (candidate_bind.bindings[var_id] == -1 ||
                    candidate_bind.bindings[var_id] == eclass_idx)
                {
                    candidate_bind.bindings[var_id] = eclass_idx;
                    match_out.add_binding(candidate_bind);
                }
            }
            return;
        }

        // Otherwise, need to check each enode in the eclass.
        BlockedList *class_nodes = &graph.class_to_nodes[eclass_idx];
        int enode_idx;
        ListIterator<int> it(class_nodes);
        while (it.next(&enode_idx))
        {
            find_matches_enode(graph, pattern, enode_idx, match_in, match_out);
        }
    }

    __global__ void kernel_eqsat_match_rules(EqSatSolver *solver)
    {
        __shared__ int local_match_count;
        __shared__ int global_allocation_start;
        __shared__ RuleMatch local_matches[N_LOCAL_MATCH_BUFF];

        if (threadIdx.x == 0)
        {
            local_match_count = 0;
        }
        __syncthreads();

        // Each node is owned by exactly one block.
        int n_nodes = solver->egraph.getNumNodes();
        int nodes_per_block = (n_nodes + gridDim.x - 1) / gridDim.x;
        int start_node = blockIdx.x * nodes_per_block;
        int end_node = min(n_nodes, start_node + nodes_per_block);

        // For each node, each thread covers one rule.
        if (threadIdx.x >= N_RULES)
            return;

        const Rule my_rule = global_ruleset.rules[threadIdx.x];
        const FuncNode my_rule_node = global_ruleset.rule_nodes[my_rule.lhs];
        for (int node_idx = start_node; node_idx < end_node; node_idx++)
        {
            // TODO we can retrieve our slice of nodes into local beforehand,
            // to avoid fetching per rule.
            const FuncNode node = solver->egraph.getNode(node_idx);
            int eclass_idx = solver->egraph.node_to_class[node_idx];

            MultiMatch initial_match = MultiMatch::nomatch();
            initial_match.add_binding(VarBind()); // Start with empty binding.
            MultiMatch found_matches = MultiMatch::nomatch();
            if (my_rule_node.name == FuncName::Var)
            {
                // TODO for trivial LHS, do we want to randomly skip sometimes?
                VarBind vb;
                vb.bound = true;
                vb.bindings[my_rule_node.args[0]] = eclass_idx;
                found_matches.add_binding(vb);
            }
            else
            {
                find_matches_enode(solver->egraph, my_rule_node, node_idx, initial_match, found_matches);
            }

            // TODO store in local matches, then flush to global when full.
            int allocation_start = atomicAdd(&local_match_count, found_matches.n_matches);
            int allocation_end = allocation_start + found_matches.n_matches;
            allocation_end = min(allocation_end, N_LOCAL_MATCH_BUFF);
            for (int i = allocation_start; i < allocation_end; i++)
            {
                local_matches[i].lhs_class_id = eclass_idx;
                local_matches[i].rhs_root = my_rule.rhs;
                for (int j = 0; j < MAX_RULE_TERMS; j++)
                    local_matches[i].var_bindings[j] = found_matches.var_binds[i - allocation_start].bindings[j];
            }
        }

        __syncthreads();

        // Flush local matches to global. Leader thread should allocate into global buffer.
        if (threadIdx.x == 0)
        {
            int allocation_start = atomicAdd(&solver->n_rule_matches, local_match_count);
            global_allocation_start = allocation_start;
        }
        __syncthreads();

        int allocation_start = global_allocation_start;
        int allocation_end = min(MAX_RULE_MATCHES, allocation_start + local_match_count);
        int copies_per_thread = (allocation_end - allocation_start + blockDim.x - 1) / blockDim.x;

        int my_start = threadIdx.x * copies_per_thread;
        int my_end = min(local_match_count, my_start + copies_per_thread);
        my_end = min(my_end, allocation_end - allocation_start);
        for (int i = my_start; i < my_end; i++)
        {
            int alloc_idx = allocation_start + i;
            solver->rule_matches[alloc_idx] = local_matches[i];
        }
    }

    void launch_eqsat_match_rules(EqSatSolver *solver)
    {
        size_t stackSize;
        cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);

        size_t newSize = 8192;                           // example: 8 KB
        cudaDeviceSetLimit(cudaLimitStackSize, newSize); // TODO figure out good number, add to const_params.cuh

        int blockSize = N_RULES;
        int numBlocks = 512; // TODO tune. Can this be num_nodes or something?
        kernel_eqsat_match_rules<<<numBlocks, blockSize>>>(solver);
        cudaDeviceSynchronize();
    }
}

// ################################################################
// # Step 2: Apply matched rules to egraph
// ################################################################

/**
 * Looks up the eclass index for the node with the given idx in the rule nodespace.
 * Will recusively lookup or insert child nodes as needed, but will not insert the
 * parent node itself. Returns -1 if not found.
 *
 * FuncNode &node_out: Output parameter that will not necessarily be inserted in the graph
 * but will always point to children eclasses present in the egraph.
 */
__device__ int lookup_and_insert_children(EqSatSolver *solver, int rhs_node_idx, VarBind &var_bindings, FuncNode &node_out)
{
    node_out = global_ruleset.rule_nodes[rhs_node_idx];
    if (node_out.name == FuncName::Var)
    {
        return var_bindings.bindings[node_out.args[0]]; // Already bound by match.
    }
    if (node_out.name != FuncName::Const)
    {
        // rhs_node currently is a copy of the node in the rule nodespace, and
        // points to other nodes in the nodespace. Promote it to point to eclasses
        // in the egraph instead.
        int argc = getFuncArgCount(node_out.name);
        bool all_children_found = true;
        for (int i = 0; i < argc; i++)
        {
            FuncNode child_node;
            int child_eclass = lookup_and_insert_children(solver, node_out.args[i], var_bindings, child_node);
            if (child_eclass == -1)
            {
                all_children_found = false;
                // Insert child
                bool inserted = solver->egraph.insertNode(child_node, -1, child_eclass);
            }
            node_out.args[i] = child_eclass;
        }
        if (!all_children_found)
        {
            return -1; // Cannot possibly find this node if children were missing.
            // (Well, we can if another thread inserts it in the meantime, but that is handled during the
            // insertion step.)
        }
    }
    // At this point we have eliminated Vars and node_out should contain the right opcode and eclass-ids.
    // Try to look it up in the egraph.
    int found_node = solver->egraph.hashcons.lookup(node_out);
    if (found_node == -1)
    {
        return -1;
    }
    return solver->egraph.resolveClassReadOnly(solver->egraph.node_to_class[found_node]);
}

__device__ void apply_match(EqSatSolver *solver, const RuleMatch &match)
{
    VarBind var_bindings;
    var_bindings.bound = true;
    for (int i = 0; i < MAX_RULE_VARS; i++)
    {
        var_bindings.bindings[i] = match.var_bindings[i];
    }

    FuncNode rhs_root_node; // Root after promoting children to eclasses.
    int rhs_eclass = lookup_and_insert_children(solver, match.rhs_root, var_bindings, rhs_root_node);
    if (rhs_eclass == -1)
    {
        // Need to insert the root node into the matched LHS class.
        bool inserted = solver->egraph.insertNode(rhs_root_node, match.lhs_class_id, rhs_eclass);
        // TODO subsequent steps?
        // TODO if inserted = false, somebody else won the race and also need to go down the merge path.
        // (unless the one who won the race (rhs_eclass) is already the same as lhs_class_id)
    }
    else if (rhs_eclass != match.lhs_class_id)
    {
        // TODO merge the classes / mark them for merging later.
    }
}

__global__ void kernel_eqsat_apply_rules(EqSatSolver *solver)
{
    int n_matches = solver->n_rule_matches;
    int matches_per_thread = (n_matches + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);

    int start_match = (blockIdx.x * blockDim.x + threadIdx.x) * matches_per_thread;
    int end_match = min(n_matches, start_match + matches_per_thread);

    for (int match_idx = start_match; match_idx < end_match; match_idx++)
    {
        RuleMatch match = solver->rule_matches[match_idx];
        apply_match(solver, match);
    }
}