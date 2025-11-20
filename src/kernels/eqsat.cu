#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

#include "eqsat.cuh"
#include "const_params.cuh"

#include "../rules.h"
#include "../parser.h" // for print expression, delete as soon as possible

using namespace gpuds;
using namespace gpuds::eqsat;
namespace gpuds::eqsat
{
    __constant__ Ruleset global_ruleset;

    // Debugging
    __global__ void printgpustate(EqSatSolver *solver)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0)
            return;

        printf("Egraph num nodes: %d\n", solver->egraph.num_nodes);
        printf("Egraph num classes: %d\n", solver->egraph.num_classes);

        for (int i = 0; i < solver->egraph.num_nodes; i++)
        {
            FuncNode node = solver->egraph.getNode(i);
            int class_id = solver->egraph.getClassOfNode(i);
            int resolved_id = solver->egraph.resolveClassReadOnly(class_id);
            printf("Node %d: class %d (resolved %d), name %d, args:\n", i, class_id, resolved_id, node.name);
            unsigned char argc = getFuncArgCount(node.name);
            for (int j = 0; j < argc; j++)
            {
                printf("  Arg %d: %d\n", j, node.args[j]);
            }
        }

        printf("Membership by class:\n");
        for (int class_id = 1; class_id < solver->egraph.num_classes + 1; class_id++)
        {
            int resolved_id = solver->egraph.resolveClassReadOnly(class_id);
            printf("Class %d (resolved %d): ", class_id, resolved_id);
            BlockedList *members = &solver->egraph.class_to_nodes[class_id];
            ListIterator<int> it(members);
            int node_id;
            while (it.next(&node_id))
            {
                printf("%d ", node_id);
            }
            printf("\n");
        }

        printf("Parents by class:\n");
        for (int class_id = 1; class_id < solver->egraph.num_classes + 1; class_id++)
        {
            int resolved_id = solver->egraph.resolveClassReadOnly(class_id);
            printf("Class %d (resolved %d): ", class_id, resolved_id);
            BlockedList *parents = &solver->egraph.class_to_parents[class_id];
            ListIterator<int> it(parents);
            int parent_id;
            while (it.next(&parent_id))
            {
                printf("%d(%d) ", parent_id, solver->egraph.resolveClassReadOnly(parent_id));
            }
            printf("\n");
        }
    }

    __global__ void printgpustate_forcomputer(EqSatSolver *solver)
    {
        if (threadIdx.x != 0 || blockIdx.x != 0)
            return;

        printf("Egraph num nodes: %d\n", solver->egraph.num_nodes);
        printf("Egraph num classes: %d\n", solver->egraph.num_classes);

        for (int i = 0; i < solver->egraph.num_nodes; i++)
        {
            FuncNode node = solver->egraph.getNode(i);
            int class_id = solver->egraph.getClassOfNode(i);
            int resolved_id = solver->egraph.resolveClassReadOnly(class_id);
            printf("%d,%d,%d,", i, class_id, node.name);
            unsigned char argc = getFuncArgCount(node.name);
            for (int j = 0; j < argc; j++)
            {
                printf("%d,", node.args[j]);
            }
            printf("\n");
        }
    }

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
            solver->egraph.hashcons.parent_egraph = &solver->egraph;
        }
    }

    __host__ EqSatSolver *construct_eqsat_solver(const std::vector<FuncNode> node_space_host,
                                                 const std::vector<int> roots_host,
                                                 std::vector<int> &compressed_roots)
    {
        printf("Expressions, uncompressed:\n");
        for (int r : roots_host)
        {
            printf("%s\n", printExpression(node_space_host, r).c_str());
        }

        EqSatSolver *d_solver;
        cudaMalloc(&d_solver, sizeof(EqSatSolver));
        initialize_eqsat_solver<<<1, 1>>>(d_solver);
        cudaMemset(d_solver->rule_matches, 0, sizeof(RuleMatch) * MAX_RULE_MATCHES);
        initialize_egraph(&d_solver->egraph, node_space_host, roots_host, compressed_roots);

        cudaMemset(&d_solver->class_dirty_merged, 0, (MAX_CLASSES + 1) * sizeof(bool));
        cudaMemset(&d_solver->class_dirty_parent, 0, (MAX_CLASSES + 1) * sizeof(bool));

        // TODO remove after debugging
        cudaDeviceSynchronize();
        printf("Did it pass through correctly?\n");
        printgpustate_forcomputer<<<1, 1>>>(d_solver);
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
            int eclass_idx = solver->egraph.getClassOfNode(node_idx);

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
                // printf("Block %d Thread %d found match for rule %d on eclass %d\n",
                //        blockIdx.x, threadIdx.x, threadIdx.x, eclass_idx);
                local_matches[i].lhs_class_id = eclass_idx;
                local_matches[i].rhs_root = my_rule.rhs;
                for (int j = 0; j < MAX_RULE_VARS; j++)
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
        printf("Before launching match of the rules.\n");
        printgpustate<<<1, 1>>>(solver);
        cudaDeviceSynchronize();
        int blockSize = N_RULES;
        int numBlocks = 512; // TODO tune. Can this be num_nodes or something?
        kernel_eqsat_match_rules<<<numBlocks, blockSize>>>(solver);

        cudaDeviceSynchronize();
        printf("Rule Match has occurred!\n");
        printgpustate<<<1, 1>>>(solver);
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
                int found_node = -1;
                bool inserted = solver->egraph.insertNode(child_node, -1, found_node);
                child_eclass = solver->egraph.getClassOfNode(found_node);
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
    // printf("B %d T %d: lookup for rhs node %d found existing node %d\n",
    //        blockIdx.x, threadIdx.x, rhs_node_idx, found_node);
    if (found_node == -1)
    {
        return -1;
    }
    return solver->egraph.getClassOfNode(found_node);
}

__device__ void apply_match(EqSatSolver *solver, const RuleMatch &match)
{

    // Copy variable bindings from the match
    VarBind var_bindings;
    var_bindings.bound = true;
    for (int i = 0; i < MAX_RULE_VARS; i++)
    {
        var_bindings.bindings[i] = match.var_bindings[i];
    }

    FuncNode rhs_root_node; // Root after promoting children to eclasses.
    int rhs_eclass = lookup_and_insert_children(solver, match.rhs_root, var_bindings, rhs_root_node);

    const int NOT_FOUND = -1;
    if (rhs_eclass == NOT_FOUND)
    {
        // Need to insert the root node into the matched LHS class.
        int found_node = -1;
        bool inserted = solver->egraph.insertNode(rhs_root_node, match.lhs_class_id, found_node);
        rhs_eclass = solver->egraph.getClassOfNode(found_node);
        // printf("B %d T %d: Inserted RHS root node into egraph with success %d, got eclass %d\n",
        //        blockIdx.x, threadIdx.x, inserted, rhs_eclass);
        // TODO subsequent steps?
    }

    // If after inserting/looking up we find that our node is in a different class,
    if (rhs_eclass != match.lhs_class_id)
    {
        // TO DONE merge the classes / mark them for merging later.
        // printf("B %d T %d: Merging LHS class %d with RHS class %d\n",
        //        blockIdx.x, threadIdx.x, match.lhs_class_id, rhs_eclass);
        solver->egraph.stageMergeClasses(match.lhs_class_id, rhs_eclass);
    }
}

__global__ void kernel_eqsat_apply_rules(EqSatSolver *solver)
{
    int n_matches = solver->n_rule_matches;
    int matches_per_thread = (n_matches + (blockDim.x * gridDim.x) - 1) / (blockDim.x * gridDim.x);

    int start_match = (blockIdx.x * blockDim.x + threadIdx.x) * matches_per_thread;
    start_match = min(start_match, n_matches);
    int end_match = min(n_matches, start_match + matches_per_thread);
    for (int match_idx = start_match; match_idx < end_match; match_idx++)
    {
        RuleMatch match = solver->rule_matches[match_idx];
        apply_match(solver, match);
        // printf("Applied match %d: LHS class %d, RHS root %d\n", match_idx, match.lhs_class_id, match.rhs_root);
    }
}

__host__ void gpuds::eqsat::launch_eqsat_apply_rules(EqSatSolver *solver)
{
    printgpustate<<<1, 1>>>(solver);
    cudaDeviceSynchronize();
    int blockSize = 16;
    int numBlocks = 512; // TODO tune
    kernel_eqsat_apply_rules<<<numBlocks, blockSize>>>(solver);
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("There was an error \n");
        printf("%s \n", cudaGetErrorString(error));
        abort();
    }
    cudaDeviceSynchronize();
    printf("Launched the kernel apply rules.\n");
    printgpustate<<<1, 1>>>(solver);
    cudaDeviceSynchronize();
}

/**
 * @brief Step 1 of the repair phase. This kernel concatenates lists from the work lists,
 * and merges class IDs accordingly.
 */
__global__ void perform_merges(EqSatSolver *solver)
{
    int merges_per_thread = (solver->egraph.classes_to_merge_count + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_merge = tid * merges_per_thread;
    int end_merge = min(solver->egraph.classes_to_merge_count, start_merge + merges_per_thread);

    for (int i = start_merge; i < end_merge; i++)
    {
        ClassesToMerge m = solver->egraph.classes_to_merge[i];

        // Perform the merge. Rely on the fact that even if the union-find is
        // stale, the final result will be correct due to the implementation
        // of the list concatenation: it will always find the appropriate
        // list tail.

        int old_root = -1;
        int new_root = gpuds::unionfind::atomic_merge_and_get_old_root(solver->egraph.class_ids, m.firstClassID, m.secondClassID, old_root);

        if (new_root == -1)
        {
            // They were already merged.
            continue;
        }

        // Concatenate old_root into new_root. The converse can lead to branching lists, so only this
        // direction is allowed.
        concatLists(&solver->egraph.class_to_nodes[new_root], &solver->egraph.class_to_nodes[old_root]);
        concatLists(&solver->egraph.class_to_parents[new_root], &solver->egraph.class_to_parents[old_root]);
        solver->class_dirty_merged[new_root] = 1;
        // printf("Marked class %d as dirty merged\n", new_root);
    }
}

__device__ void dehash_parent(EqSatSolver *solver, int parent_id)
{
    // CAS is the only thing that works on 16 bits. Can't get away with 8 bits here.
    printf("B %d T %d: Trying to dehash parent class %d\n",
           blockIdx.x, threadIdx.x, parent_id);
    unsigned short int ownership = atomicCAS(&solver->class_dirty_parent[parent_id], (unsigned short)0, (unsigned short)1);
    printf("B %d T %d: Dehash ownership result for parent class %d: %d (sanity check: %d)\n",
           blockIdx.x, threadIdx.x, parent_id, ownership, solver->class_dirty_parent[parent_id]);
    if (ownership != 0)
        return;

    // printf("B %d T %d: Dehashing parent class %d\n",
    //        blockIdx.x, threadIdx.x, parent_id);

    BlockedList *member_list = &solver->egraph.class_to_nodes[parent_id];
    ListIterator<int> it(member_list);
    int node_id;
    while (it.next(&node_id))
    {
        // Remove from hashcons
        FuncNode node = solver->egraph.getNode(node_id);
        solver->egraph.hashcons.remove(node);
    }
}

/**
 * @brief Step 2 of the repair phase. This kernel, for all merged classes:
 * - deduplicates the parent list
 * - while iterating over parents, removes all their members from the hash table
 * -> we can check if we're the first thread doing this by checking if the first node has
 *    already been removed.
 * @param solver
 * @return
 */
__global__ void deduplicate_and_dehash_parents(EqSatSolver *solver)
{
    // TODO if space problems, change to shared.
    char have_seen_parent[MAX_CLASSES + 1];
    // Each thread grabs a class. If marked dirty, mark clean and process.
    int tid = blockIdx.x; // Probably don't want multiple threads on this warp since divergence will essentially serialize?
    if (threadIdx.x > 0)
        return;
    // printf("Dedup kernel launched with on TID %d\n", tid);
    int classes_per_thread = (solver->egraph.num_classes + gridDim.x - 1) / (gridDim.x);
    int start_class = tid * classes_per_thread;
    int end_class = min(solver->egraph.num_classes, start_class + classes_per_thread);

    assert(classes_per_thread <= 255); // else have_seen_parent won't fit.
    memset(have_seen_parent, 0, sizeof(char) * (solver->egraph.num_classes + 1));
    // printf("deduplicating: %d to %d\n", start_class, end_class);

    for (int class_id = start_class; class_id < end_class; class_id++)
    {
        printf("Dehash: Thread %d checking parents for class %d\n", tid, class_id);
        if (!solver->class_dirty_merged[class_id])
        {
            printf("Dehash: Thread %d skipping class %d due to not being dirty\n", tid, class_id);
            continue;
        }
        solver->class_dirty_merged[class_id] = 0; // Mark clean.

        printf("B %d T %d: Deduplicating and dehashing parents for class %d\n",
               blockIdx.x, threadIdx.x, class_id);

        char seen_marker = (char)(class_id - start_class + 1); // Avoid 0 marker.

        // We now have sole ownership of this class's parent list.
        BlockedList *parent_list = &solver->egraph.class_to_parents[class_id];
        // Deduplicate parent list. Hopefully not too many parents.
        int parent_id;
        ListIterator<int> it(parent_list);
        printf("Class id %d \n", class_id);
        for (; it.hasNext(); it.next(&parent_id))
        {
            it.peek(&parent_id);
            if (parent_id < 0)
                continue; // previously marked deleted.
            parent_id = solver->egraph.resolveClass(parent_id);
            printf("Class id %d has Parent Id %d \n", class_id, parent_id);
            if (have_seen_parent[parent_id] == seen_marker)
            {
                // Duplicate, remove from list by marking as -1.
                it.write(-1); // TODO oops, this points to the next item.
            }
            else
            {
                have_seen_parent[parent_id] = seen_marker;
                dehash_parent(solver, parent_id);
            }
        }
    }

    // Bonus: reset the worklist count.
    if (tid == 0)
    {
        solver->egraph.classes_to_merge_count = 0;
    }
}

/**
 * Step 3: deduplicate merged classes member nodes is uncessary!
 *
 * - In order to have a duplicate, it must have arisen from having stale child eclasses:
 *   otherwise, the hashcons would have prevented inserting the same node twice.
 * - Stale child eclasses can only arise if the class is a merge parent itself.
 * - However, if the class is a merge parent, its children will be re-resolved in the next step,
 *   so duplicates will be eliminated then.
 */
// TODO: a sanity check kernel to verify the above claim?

/**
 * @brief Phase 4 of the repair phase. This kernel, for all merged classes:
 * - Adds nodes back to the hash table with new hashes
 * - Already present nodes will add new merges to the worklist
 *
 * @param solver
 * @return
 */
__global__ void reinsert_parents_of_merged(EqSatSolver *solver)
{

    int classes_to_investigate_per_thread = (solver->egraph.num_classes + blockDim.x * gridDim.x - 1) / (blockDim.x * gridDim.x);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_parent = tid * classes_to_investigate_per_thread;
    int end_parent = min(solver->egraph.num_classes, start_parent + classes_to_investigate_per_thread);

    for (int class_id = start_parent + 1; class_id < end_parent + 1; class_id++)
    {
        printf("Thread %d investigating class %d\n", tid, class_id);
        if (!solver->class_dirty_parent[class_id])
        {
            printf("Thread %d skipping class %d due to not being dirty\n", tid, class_id);
            // Not dirty, skip.
            continue;
        }
        unsigned resolved_id = solver->egraph.resolveClass(class_id);
        // Not root, our work is owned by another thread.
        if (resolved_id != class_id)
        {
            printf("Thread %d skipping class %d due to not being root\n", tid, class_id);
            continue;
        }
        printf("Thread %d reinserting parents for class %d\n", tid, class_id);
        int next;
        for (ListIterator<int> i = ListIterator<int>(&(solver->egraph.class_to_nodes[class_id])); i.next(&next);)
        {
            int old_value;
            bool inserted = solver->egraph.hashcons.insert(solver->egraph.node_space[next], next, old_value);
            if (!inserted)
            {
                // Look at old_value's class. Add both that and this parent to a work list.
                // Then, mark this node as deleted (the other one remains).
                unsigned resolved_old_class_id = solver->egraph.resolveClass(old_value);

                solver->egraph.stageMergeClasses(resolved_old_class_id, class_id);
                // they are staged for the next round.
            }
        }
    }
}

__host__ void gpuds::eqsat::repair_egraph(EqSatSolver *solver)
{
    int num_merges_left;
    cudaMemcpy(&num_merges_left,
               (char *)solver +
                   offsetof(EqSatSolver, egraph) +
                   offsetof(EGraph, classes_to_merge_count),
               sizeof(int), cudaMemcpyDeviceToHost);
    while (num_merges_left > 0)
    {
        printf("Kernel 1 beginning...\n");
        perform_merges<<<128, 16>>>(solver);
        cudaDeviceSynchronize();
        printgpustate<<<1, 1>>>(solver);
        cudaDeviceSynchronize();
        printf("Kernel 2 beginning...\n");
        deduplicate_and_dehash_parents<<<512, 16>>>(solver);
        printgpustate<<<1, 1>>>(solver);
        // printgpustate<<<1, 1>>>(solver)
        cudaDeviceSynchronize();
        printf("Kernel 4 beginning...\n");

        cudaDeviceSynchronize();

        reinsert_parents_of_merged<<<512, 16>>>(solver);
        printf("After reinsertion\n");
        cudaDeviceSynchronize();
        cudaDeviceSynchronize();
        printgpustate<<<1, 1>>>(solver);
        printgpustate_forcomputer<<<1, 1>>>(solver);

        cudaMemcpy(&num_merges_left,
                   (char *)solver +
                       offsetof(EqSatSolver, egraph) +
                       offsetof(EGraph, classes_to_merge_count),
                   sizeof(int), cudaMemcpyDeviceToHost);

        printf("Kernels complete, and %d merges remain!\n", num_merges_left);
    }
}