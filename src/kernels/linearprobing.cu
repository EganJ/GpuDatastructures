#include "stdio.h"
#include "stdint.h"
#include "vector"
#include "linearprobing.h"

typedef unsigned Value;
typedef unsigned Key;
typedef unsigned Hash;
// Apparently despite this being the same as long, atomic needs the extra "long" word.
typedef unsigned long long HashValue;


// 32 bit Murmur3 hash
__host__ __device__ uint32_t hash(uint32_t k)
{
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity-1);
}

__host__ __device__ uint32_t hashfn(uint32_t k)
{
    //TODO replace?
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k & (kHashTableCapacity-1);
}

// Create a hash table. For linear probing, this is just an array of KeyValues
KeyValue* create_hashtable() 
{
    // Allocate memory
    KeyValue* hashtable;
    cudaMalloc(&hashtable, sizeof(KeyValue) * kHashTableCapacity);

    // Initialize hash table to empty
    static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
    cudaMemset(hashtable, 0xff, sizeof(KeyValue) * kHashTableCapacity);

    return hashtable;
}

__device__ unsigned long long atomic_load_64(unsigned long long* v)
{
    return *v;
}

__device__ unsigned atomic_load_32(unsigned* v)
{
    return *v;
}

__device__ Hash getHash(HashValue hv)
{
    return ((KeyValue) hv).key;
}

__device__ Value getValue(HashValue hv)
{
    return ((KeyValue) hv).value;
}

__device__ HashValue createHashValue(Hash h, Value v)
{
    KeyValue kv = {h, v};
    return *(HashValue*)(&kv);
}

__device__ Value getKeyFromValue(Value v)
{
    return v; // TODO
}

__device__ Value lookup(KeyValue* table, Key key) {
    Hash h = hashfn(key);
    unsigned idx = h;
    while (true) {
        HashValue HV = atomic_load_64((HashValue*) &table[idx]);
        if (getHash(HV) == h) {
            Value V = getValue(HV);
            Key other_key = getKeyFromValue(V);
            if (other_key == key)
                return V;
        }
        idx++;
    }
}

// Concurrent with lookup and insert, but NOT delete if we want unique inserts.
// note: we will have to ensure table is aligned to 8 to be able to atomic on it.
__device__ bool insert(KeyValue* table, Key key, Value val) {
    Hash hash = hashfn(key);
    unsigned idx = hash;
    while (true) {
        HashValue HV = atomic_load_64((HashValue*) &table[idx]);
        Hash H = getHash(HV);
        Value V = getValue(HV);
        if (H == kWasPresentButDeleted || H == kEmpty) {
            Key other_key = getKeyFromValue(V);
            if (other_key != key){
                HashValue item_to_emplace = createHashValue(hash, val);
                HashValue hv2 = atomicCAS((HashValue*) &table[idx], HV, item_to_emplace); 
                if (hv2 == HV) {
                    return true;
                } else {
                    // Duplicates, either do nothing or overwrite? Do nothing 
                    // works for our application
                    return false;
                }
            }
        } else {
            idx++;
        }
    }
}


// NO CONCURRENT DELETES AND INSERTS.
__device__ void del(KeyValue* table, Key key){

    uint32_t hash = hashfn(key);
    uint32_t idx = hash;
    while (true) {
        HashValue HV = atomic_load_64((unsigned long long*) &table[idx]);
        Hash H = getHash(HV);
        Value V = getValue(HV);
        Key other_key = getKeyFromValue(V);
        if (other_key != key){
            idx++;
            idx = idx % kHashTableCapacity;
            continue;
        } else if (H == kEmpty){
            return; // Item is not in the hash table.
        }
        HashValue prev = atomicCAS((HashValue*) &table[idx], HV, kWasPresentButDeleted);
        return;
    } 
}



// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
__global__ void gpu_hashtable_delete(KeyValue* hashtable, const KeyValue* kvs, unsigned int numkvs)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < numkvs)
    {
        uint32_t key = kvs[threadid].key;
        uint32_t slot = hash(key);

        while (true)
        {
            if (hashtable[slot].key == key)
            {
                while(atomicCAS(&hashtable[slot].key, key, kWasPresentButDeleted) == key){
                }
                return;
            }
            if (hashtable[slot].key == kEmpty)
            {
                return;
            }
            slot = (slot + 1) & (kHashTableCapacity - 1);
        }
    }
}



// Iterate over every item in the hashtable; return non-empty key/values
__global__ void gpu_iterate_hashtable(KeyValue* pHashTable, KeyValue* kvs, uint32_t* kvs_size)
{
    unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kHashTableCapacity) 
    {
        if (pHashTable[threadid].key != kEmpty) 
        {
            uint32_t value = pHashTable[threadid].value;
            if (value != kEmpty)
            {
                uint32_t size = atomicAdd(kvs_size, 1);
                kvs[size] = pHashTable[threadid];
            }
        }
    }
}


void insert_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0,0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_insert<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    if (kDebug){
        KeyValue* my_hash_map = (KeyValue*) malloc(sizeof(KeyValue) * kHashTableCapacity);
        cudaError_t err = cudaMemcpy(my_hash_map, pHashTable, sizeof(KeyValue) * kHashTableCapacity, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        if (err != cudaSuccess){
            printf(" Failed to read the memory back.");
        }

    printf("num kvs: %d\n", num_kvs);
    for (int j = 0; j < num_kvs; j++) {
        printf("j: %d \t v: %d\n", j, my_hash_map[j].key);
    }


    bool all_present = true;
    int num_missing_keys = 0;
    for (int i = 0; i < num_kvs; i++){
        bool this_is_present = false;
        for (int j = 0; j < kHashTableCapacity; j++){
            if (my_hash_map[j].key == kvs[i].key){
                this_is_present = true;
            }
        }
        if (!this_is_present){
            all_present = false;
            num_missing_keys++;
            printf("Missing key: %d \n", kvs[i].key);
            // printf(" Expected at %d \n");
        }
    }
    printf(" All of the keys are present in the hash map after calling bulk insert? %d\n, We are missing %d out of %d keys\n", all_present, num_missing_keys, num_kvs);

    }



    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU inserted %d items in %f ms (%f million keys/second)\n", 
        num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}

void delete_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_delete<< <gridsize, threadblocksize >> > (pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    if (kDebug){
        KeyValue* my_hash_map = (KeyValue*) malloc(sizeof(KeyValue) * kHashTableCapacity);
        cudaError_t err = cudaMemcpy(my_hash_map, pHashTable, sizeof(KeyValue) * kHashTableCapacity, cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
        if (err != cudaSuccess){
            printf(" Failed to read the memory back.");
        }

        for (int i = 0; i < num_kvs; i++){
            bool found = false;

            for (int j = 0; j < kHashTableCapacity; j++){
                if (my_hash_map[j].key == kvs[i].key){
                    found = true;
                    printf(" Key: %d found at position %d in hash map \n", kvs[i].key, j);
                }
            }
            if (found){
                printf(" The key %d was not deleted.\n", kvs[i].key);
            }
        }
    }


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU delete %d items in %f ms (%f million keys/second)\n",
        num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}


std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable)
{
    uint32_t* device_num_kvs;
    cudaMalloc(&device_num_kvs, sizeof(uint32_t));
    cudaMemset(device_num_kvs, 0, sizeof(uint32_t));

    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * kNumKeyValues);

    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_iterate_hashtable, 0, 0);

    int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
    gpu_iterate_hashtable<<<gridsize, threadblocksize>>>(pHashTable, device_kvs, device_num_kvs);

    uint32_t num_kvs;
    cudaMemcpy(&num_kvs, device_num_kvs, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<KeyValue> kvs;
    kvs.resize(num_kvs);

    cudaMemcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyDeviceToHost);

    cudaFree(device_kvs);
    cudaFree(device_num_kvs);

    return kvs;
}

// Free the memory of the hashtable
void destroy_hashtable(KeyValue* pHashTable)
{
    cudaFree(pHashTable);
}


void lookup_hashtable(KeyValue* pHashTable, KeyValue* kvs, uint32_t num_kvs)
{
    // Copy the keyvalues to the GPU
    KeyValue* device_kvs;
    cudaMalloc(&device_kvs, sizeof(KeyValue) * num_kvs);
    cudaMemcpy(device_kvs, kvs, sizeof(KeyValue) * num_kvs, cudaMemcpyHostToDevice);

    // Have CUDA calculate the thread block size
    int mingridsize;
    int threadblocksize;
    cudaOccupancyMaxPotentialBlockSize(&mingridsize, &threadblocksize, gpu_hashtable_insert, 0, 0);

    // Create events for GPU timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Insert all the keys into the hash table
    int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
    gpu_hashtable_lookup << <gridsize, threadblocksize >> > (pHashTable, device_kvs, (uint32_t)num_kvs);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float seconds = milliseconds / 1000.0f;
    printf("    GPU lookup %d items in %f ms (%f million keys/second)\n",
        num_kvs, milliseconds, num_kvs / (double)seconds / 1000000.0f);

    cudaFree(device_kvs);
}