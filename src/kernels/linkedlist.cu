#include <cuda_runtime.h>
#include <stdio.h>

#include "linkedlist.cuh"

/**
 * @brief A blocked linked list representation which is easily portable to the GPU.
 * There are several features that make this linked list unique compared to conventional implementations.
 *
 * 1. Blocked nodes - each node has a variable capacity (block_size) allowing for fewer pointer dereferences.
 * 2. Buffer system with integer indexing - this linked list is designed to be placed inside of a large contiguous section
 * of pre-allocated memory. Such a section can be easily transferred to the GPU, since it does not rely fragmented heap space
 * from malloc() calls. Additionally, pointers are stored as indices relative to the start of the buffer, meaning that
 * transferring data over different buffers (such as CPU -> GPU) will require fewer pointers to be rewritten.
 */

__device__ ListNode *BlockedListBuffer::allocateBlock(unsigned block_size_char)
{
  int alloc_size = sizeof(ListNode) + block_size_char;
  auto alloc_end = atomicAdd(&buffer_allocated, alloc_size);
  assert(alloc_end < buffer_size);
  auto alloc_start = alloc_end - alloc_size;
  auto ln = (ListNode *)index(alloc_start);
  ln->block_size = block_size_char;
  ln->next_node = NULL_ID;
  return ln;
}

/**
 * @brief Creates a new blocked linked list in a given buffer of memory.
 * Does not modify the blocked list buffer (yet) but configures the list to point to it.
 *
 * @param blb A blocked list buffer in which the list's nodes will reside
 * @return BlockedList struct which represents the start and end indices of a list.
 */
__host__ __device__ BlockedList newBlockedList(BlockedListBuffer *blb)
{
  BlockedList bl;
  bl.buffer = blb;
  bl.start_pointer = NULL_ID;
  bl.end_pointer = NULL_ID;
  return bl;
}

/**
 * @brief Syntactic sugar for dereferencing an index pointer inside of a blocked list.
 * Won't actually check if the pointer is garbage or actually points to a list node.
 *
 * @param bl Blocked list whose buffer to index inside of
 * @param index The index to look at
 * @return ListNode* A list node at which the index points to, representing a list node.
 */
__host__ __device__ ListNode *getNode(BlockedList *bl, int index)
{
  return (ListNode *)&(bl->buffer->buffer_data[index]);
}

/**
 * @brief Syntactic sugar for getting the index into a blocked list buffer of a node
 *
 * @param bl Blocked list which contains...
 * @param ln A list node
 * @return int The index into the blocked list's buffer of the node
 */
__host__ __device__ unsigned getIndex(BlockedList *bl, ListNode *ln)
{
  return (unsigned)((char *)ln - (char *)&(bl->buffer->buffer_data));
}

/**
 May return a stale value!
*/
__device__ unsigned resolveListEnd(BlockedListBuffer *start, unsigned block_idx)
{
  ListNode *node = (ListNode *)start->index(block_idx);
  while (node->next_node != NULL_ID)
  {
    block_idx = node->next_node;
    node = (ListNode *)start->index(block_idx);
  }
  return block_idx;
}

/**
 * May return a stale value!
 * Will update the end pointer if it finds it was stale (and untouched in the meantime)
 */
__device__ unsigned resolveListEnd(BlockedList *bl)
{
  unsigned original_end = bl->end_pointer;
  unsigned val = resolveListEnd(bl->buffer, original_end);
  atomicCAS(&bl->end_pointer, original_end, val);
  return val;
}

/**
 * @brief Adds a list node to the end of a blocked list.
 * Assumes the list node resides in the memory of the blocked list's buffer
 *
 * @param bl Blocked list to add node to
 * @param ln List node to add to the blocked list
 */
__device__ void addToList(BlockedList *bl, ListNode *ln)
{
  unsigned idx = getIndex(bl, ln);

  if (atomicCAS(&bl->end_pointer, NULL_ID, idx) == NULL_ID)
  {
    bl->start_pointer = idx;
    return;
  }

  unsigned last_node_idx = 0;
  ListNode *end_node;
  do
  {
    last_node_idx = resolveListEnd(bl);
    end_node = getNode(bl, last_node_idx);
  } while (atomicCAS(&end_node->next_node, NULL_ID, idx) != NULL_ID);
  atomicCAS(&bl->end_pointer, last_node_idx, idx); // May be stale but that's fine. Could be omitted.
}

/**
 * @brief Concatenates two lists inside of the same blocked list.
 * Modifies the first list so that it is the result. The second list remains
 * valid but changes to the first list's nodes will affect it.
 *
 * @param bl1 First blocked list
 * @param bl2 Second blocked list. THIS SHOULD NOT BE THE RHS ARG TO A CONCURRENT CONCAT.
 * @return BlockedList* Same as bl1, but now the old last node of bl1 points to the first node of bl2,
 * and the new last node of bl1 is that of bl2.
 */
__device__ BlockedList *concatLists(BlockedList *bl1, BlockedList *bl2)
{
  addToList(bl1, getNode(bl2, bl2->start_pointer));
  atomicCAS(&bl1->end_pointer, bl2->start_pointer, bl2->end_pointer);
  return bl1;
}

/**
 * @brief Allocates a new list node inside of a blocked list buffer.
 *
 * @param blb A blocked list buffer to allocate the node inside of
 * @param size How many elements which can be put in the node
 * @return ListNode* The new list node
 */
__device__ ListNode *newNode(BlockedListBuffer *blb, int size)
{
  int index = atomicAdd(&(blb->buffer_size), sizeof(ListNode) + size * sizeof(char));
  ListNode *ln = (ListNode *)&(blb->buffer_data[index]);
  ln->block_size = size;
  ln->next_node = NULL_ID;
  return ln;
}
