#ifndef GPUDS_KERNELS_EQSAT_CUH
#define GPUDS_KERNELS_EQSAT_CUH

#include <cuda_runtime.h>

#include "../rules.h"

#include "egraph.cuh"
#include "const_params.cuh"

namespace gpuds::eqsat
{
    __constant__ char func_arg_counts[sizeof(func_operand_count)/sizeof(func_operand_count[0])];

    struct Ruleset
    {
        Rule rules[N_RULES];
        FuncNode rule_nodes[MAX_RULESET_TERMS];
    };
    // TODO copy over rules, terms from parser
    __constant__ Ruleset global_ruleset;

    // Contains the result of a single rule match.
    // Records which class had the LHS, the IDX of the
    // RHS root node inside of the Ruleset nodes space,
    // and the variable bindings to classIDs inside the
    // e-graph.
    struct RuleMatch
    {
        int lhs_class_id;
        int rhs_root;
        int var_bindings[MAX_RULE_VARS];
    };

    class EqSatSolver
    {
        public:
        EGraph egraph; // large.

        // Storage for matches found during rule matching.
        int n_rule_matches = 0;
        RuleMatch rule_matches[MAX_RULE_MATCHES]; // large.

    };

    __global__ void kernel_eqsat_match_rules(EqSatSolver *solver);
    void launch_eqsat_match_rules(EqSatSolver *solver);

    void initialize_eqsat_memory();

} // namespace gpuds::eqsat

#endif