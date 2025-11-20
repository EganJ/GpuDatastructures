#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "linearprobing.h"
#include "const_params.cuh"
#include "utils.cuh"
#include "egraph.cuh"

typedef unsigned long long llu;

// Inspired by https://github.com/nosferalatu/SimpleGPUHashTable/blob/master/src/main.cpp.

// Hash a FuncNode to produce a hash value. Need to use a hash that accepts
// a sequence of ints. Go with the FNV hash.
const unsigned fnv_prime = 16777619;
const unsigned fnv_offset = 2166136261;
__device__ Hash hashNode(const FuncNode *node)
{
    int argc = getFuncArgCount(node->name);
    int bytes_to_hash = sizeof(FuncNode::name) + (argc) * sizeof(int);
    unsigned hash = fnv_offset;
    const unsigned char *p = (const unsigned char *)node;
    for (int i = 0; i < bytes_to_hash; i++)
    {
        hash ^= p[i];
        hash *= fnv_prime;
    }
    return hash % MAX_HASH_CAPACITY;
}

__device__ inline FuncNode HashTable::valToKey(int val)
{
    int key_id = valToKeyId(val);
    return parent_egraph->getNode(key_id);
}

__device__ inline unsigned HashTable::nextIdx(unsigned idx)
{
    idx++;
    return (idx < MAX_HASH_CAPACITY) ? idx : 0;
}

__device__ Value HashTable::lookup(const FuncNode &node)
{
    Hash h = hashNode(&node);
    unsigned idx = h;

    HashValue HV;
    while (true)
    {
        // It's ok if we miss some? TODO
        __nv_atomic_load(&table[idx], &HV, __NV_ATOMIC_RELAXED);
        if (HV.hash == h)
        {
            // Check if keys match.
            FuncNode other_key = valToKey(HV.value);
            if (other_key == node)
            {
                return HV.value;
            }
        }
        else if (HV.hash == kSentinelUnused)
        {
            return -1;
        }
        idx = nextIdx(idx);
    }
}

__device__ inline HashValue atomicCAS_HV(HashValue *address, HashValue compare, HashValue val)
{
    llu old = atomicCAS((llu *)address, *((llu *)&compare), *((llu *)&val));
    return *((HashValue *)&old);
}

// Concurrent with lookup and insert, but NOT delete if we want unique inserts.
// note: we will have to ensure table is aligned to 8 to be able to atomic on it.
__device__ bool HashTable::insert(const FuncNode &key, Value val, Value &member_value)
{
    uint32_t hash = hashNode(&key);

    unsigned idx = hash;
    HashValue HV;
    while (true)
    {
        // Can use atomic_relaxed here: we are gauranteed to see writes
        // to this location even if we miss them here, because the atomicCAS
        // has a stronger memory ordering.
        __nv_atomic_load(&table[idx], &HV, __NV_ATOMIC_RELAXED);
        if (HV.hash == kSentinelDeleted || HV.hash == kSentinelUnused)
        {
            // Try to write here, but someone else may write first!
            HashValue item_to_emplace = {hash, val};
            HashValue hv2 = atomicCAS_HV(&table[idx], HV, item_to_emplace);
            if (hv2 == HV)
            {
                // We won the race to insert
                member_value = val;
                return true;
            }
            // Otherwise, someone else wrote first; continuing the loop will
            // check what they wrote.
            continue;
        }
        else if (HV.hash == hash)
        {
            // Check if keys match.
            FuncNode other_key = valToKey(HV.value);
            if (other_key == key)
            {   
                printf("Structural equaltity found between nodes:\n");
                printf(" (1) name: %d args: ", key.name);
                int argc = getFuncArgCount(key.name);
                for (int i = 0; i < argc; i++) {
                    printf("%d ", key.args[i]);
                }
                printf("\n");
                printf(" (2) name: %d args: ", other_key.name);
                argc = getFuncArgCount(other_key.name);
                for (int i = 0; i < argc; i++) {
                    printf("%d ", other_key.args[i]);
                }
                printf("\n");
                member_value = HV.value;
                return false; // already present
            }
        }
        idx = nextIdx(idx);
    }
}

// NO CONCURRENT DELETES AND INSERTS.
__device__ void HashTable::remove(const FuncNode &key)
{

    uint32_t hash = hashNode(&key);
    uint32_t idx = hash;
    HashValue HV;
    while (true)
    {
        // Not concurrent with inserts, so can use relaxed here.
        __nv_atomic_load(&table[idx], &HV, __NV_ATOMIC_RELAXED);

        if (HV.hash == kSentinelUnused)
        {
            return; // Item is not in the hash table.
        }
        if (HV.hash == hash)
        {
            // Check if keys match.
            FuncNode other_key = valToKey(HV.value);
            if (other_key == key)
            {
                // Found it; mark as deleted. Don't need to check race because
                // if we lose that means someone else already deleted it.
                // This may need to be strenghted if we want to return
                // whether or not we actually deleted something.
                HashValue deleted_hv = {kSentinelDeleted, HV.value};
                __nv_atomic_store((HashValue *)&table[idx], &deleted_hv, __NV_ATOMIC_RELAXED);
                return;
            }
        }
        idx = nextIdx(idx);
    }
}

__host__ void initialize_hashcons_memory(HashTable *ht)
{
    cudaMemset(ht->table, HashTable::kSentinelUnused, sizeof(HashValue) * MAX_HASH_CAPACITY);
}