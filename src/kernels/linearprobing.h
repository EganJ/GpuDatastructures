#pragma once

#include <vector>

struct KeyValue
{
    unsigned key;
    unsigned value;
};

// const unsigned kHashTableCapacity = 128 * 1024 * 1024;
const unsigned kHashTableCapacity = 128*128*128;
const bool kDebug = false;


const unsigned kNumKeyValues = kHashTableCapacity / 2;

const unsigned kEmpty = 0xffffffff;
const unsigned kWasPresentButDeleted = kEmpty - 1;

KeyValue* create_hashtable();

void insert_hashtable(KeyValue* hashtable, const KeyValue* kvs, unsigned num_kvs);

void lookup_hashtable(KeyValue* hashtable, KeyValue* kvs, unsigned num_kvs);

void delete_hashtable(KeyValue* hashtable, const KeyValue* kvs, unsigned num_kvs);

std::vector<KeyValue> iterate_hashtable(KeyValue* hashtable);

void destroy_hashtable(KeyValue* hashtable);
