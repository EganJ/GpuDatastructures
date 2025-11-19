#pragma once

#include <vector>
#include "const_params.cuh"
#include "../rules.h"
#include <cuda_runtime.h>

typedef int Value;
typedef unsigned Hash;
struct HashValue
{
    Hash hash;
    Value value;

    __device__ bool operator==(const HashValue &other) const
    {
        return hash == other.hash && value == other.value;
    }
};
static_assert(sizeof(HashValue) == sizeof(unsigned long long));

// Forward declaration so we can use it in HashTable methods
class EGraph;

struct HashTable
{
    // Hashes should be in range [0, kHashTableCapacity), so these sentinels are safe.
    const static unsigned kSentinelUnused = 0xFFFFFFFF;
    const static unsigned kSentinelDeleted = kSentinelUnused - 1;
    const int kNotFound = -1;

    EGraph *parent_egraph;
    HashValue table[MAX_HASH_CAPACITY];

    /**
     * Looks up the key associated for the given value.
     * TODO: needs access to the egraph to make sense
     */
    __device__ inline int valToKeyId(int val) { return val; } // identity for now
    __device__ inline FuncNode valToKey(int val);

    __device__ int computeHash(const FuncNode &node);

    /**
     * Looks up a (potentially stale) value for the given key, and
     * returns it or -1 if not found. Concurrent with insertions and deletions.
     */
    __device__ int lookup(const FuncNode &node);

    /**
     *  Inserts the node. Concurrent with lookups but not with deletes.
     * Concurrent with other inserts: may not insert duplicates, and if
     * a duplicate is found, returns false without inserting. In such
     * a case, out_node_id is set to the existing node's ID.
     */
    __device__ bool insert(const FuncNode &node, int value, int &old_value);

    /**
     * Marks the given node as deleted. Concurrent with lookups.
     * Not concurrent with inserts.
     */
    __device__ void remove(const FuncNode &node);

    // Helper to get the next index in the table, wrapping around.
    __device__ static inline unsigned nextIdx(unsigned idx);
};

__host__ void initialize_hashcons_memory(HashTable *hashcons);