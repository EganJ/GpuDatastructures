#ifndef GPUDS_KERNELS_CONST_PARAMS_CUH
#define GPUDS_KERNELS_CONST_PARAMS_CUH
#include <cuda.h>

#include "../rules.h"

// TODO currently hardcoded from the rules list, find more elegant way
// Number of rules in the rules list, exact.
const int N_RULES = 301;
// Maximum number of terms across all rules.
const int MAX_RULESET_TERMS = 700;
const int MAX_RULE_TERMS = 8;

// Made-up limits for now
// Maximum number of supported equivalence classes, counting
// ones that have been merged away.
const int MAX_CLASSES = 10000;
const int MAX_MERGE_LIST_SIZE = 1000;
// Maximum number of nodes in the e-graph.
const int MAX_NODES = 50000;
const int MAX_LIST_SPACE = 2 * MAX_NODES * sizeof(int) / sizeof(char);
const int MAX_HASH_CAPACITY = (int)(1.5 * MAX_NODES);

// Parameters for the LHS matching phase
const int N_LOCAL_MATCH_BUFF = 256; // Size of the shared-memory buffer for holding matches per block
const int MULTIMATCH_LIMIT = 4;     // Making this finite may miss possible matches.
const int MAX_RULE_MATCHES = 10000; // Tweak this.

__host__ void setArgCounts();

__device__ int getFuncArgCount(const FuncName& fname);
#endif