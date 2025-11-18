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

__host__ __device__ ListNode *getNode(BlockedList *bl, int index);

__device__ unsigned getIndex(BlockedList *bl, ListNode *ln);

__device__ unsigned resolveListEnd(BlockedListBuffer *start, unsigned block_idx);

__device__ unsigned resolveListEnd(BlockedList *bl);

__device__ void addToList(BlockedList *bl, ListNode *ln);

__device__ BlockedList *concatLists(BlockedList *a, BlockedList *b);

__device__ ListNode *newNode(BlockedListBuffer *buffer, unsigned block_size);

/**
 * @brief Iterator which can be used to iterate through the elements of a
 * BlockedList. Requires all blocks of the blocked list to be sized a multiple
 * of the size of datatype given.
 *
 * @tparam T Type of value to extract
 */
template <typename T>
struct ListIterator
{
    BlockedList *list;
    ListNode *node;
    int element_index;
    bool finished = false;

    /**
     * @brief Construct a new ListIterator object starting at the beginning of the
     * linked list given
     *
     * @param bl The linked list given
     */
    __host__ __device__ ListIterator(BlockedList *bl)
    {
        list = bl;
        if (bl->start_pointer == NULL_ID)
            finished = true;
        else
        {
            node = getNode(bl, bl->start_pointer);
            element_index = 0;
        }
    }

    /**
     * @brief Gets the next element in the list, if there are any left.
     *
     * @param result Where the next element will be stored.
     * @return true If an element was returned
     * @return false If we're at the end of the list. In this case, "result" is unchanged.
     */
    __host__ __device__ bool next(T *result)
    {
        if (finished)
            return false;

        *result = *((T *)&node->data[element_index]);
        element_index += sizeof(T);
        if (element_index >= node->block_size)
        {
            element_index = 0;
            if (node->next_node == NULL_ID)
                finished = true;
            else
                node = getNode(list, node->next_node);
        }

        return true;
    }

    /**
     * @brief Returns if there are any elements left in the list being iterated through.
     *
     * @return true
     * @return false
     */
    __host__ __device__ bool hasNext()
    {
        return !finished;
    }
};

#endif