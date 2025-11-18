#ifndef GPUDS_KERNELS_LINKEDLIST_CUH
#define GPUDS_KERNELS_LINKEDLIST_CUH

const unsigned NULL_ID = std::numeric_limits<unsigned>::max();

/**
 * @brief A buffer which can contain multiple blocked lists.
 * To create one, cast a block of memory to this struct.
 */
struct BlockedListBuffer
{
    int buffer_size = 0;
    char buffer_data[0];

    __device__ inline char *index(unsigned idx)
    {
        return &buffer_data[idx];
    }
};

/**
 * @brief A node of a blocked linked list. This node can contain more than one element.
 * Indexes into a blocked list buffer (not stored in the list node for compactness).
 */
struct ListNode
{
    unsigned block_size;
    unsigned next_node;
    char data[0];
};

/**
 * @brief A blocked linked list. Stores pointers to the first and last element for easy concatenation.
 * Pointers are indexes into the buffer.
 */
struct BlockedList
{
    BlockedListBuffer *buffer;
    unsigned start_pointer;
    unsigned end_pointer;
};

BlockedList newBlockedList(BlockedListBuffer *buffer);

__device__ ListNode *getNode(BlockedList *bl, int index);

__device__ unsigned getIndex(BlockedList *bl, ListNode *ln);

__device__ unsigned resolveListEnd(BlockedListBuffer *start, unsigned block_idx);

__device__ unsigned resolveListEnd(BlockedList *bl);

__device__ void addToList(BlockedList *bl, ListNode *ln);

__device__ BlockedList *concatLists(BlockedList *a, BlockedList *b);

__device__ ListNode *newNode(BlockedListBuffer *buffer, unsigned block_size);

#endif