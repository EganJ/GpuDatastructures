import numpy as np

class PriorityQueue:

    def __init__(self):

        self.priority_to_queue: dict[int, list[int]] = {}

    def mass_insert_one_at_a_time(self, priorities: list[int], values: list[int]):

        for ind, p in enumerate(priorities):
            if p in self.priority_to_queue:
                # We can add in directly.
                self.priority_to_queue[p].append(values[ind])
            else:
                self.priority_to_queue[p] = [values[ind]]
    
    def mass_insert_all_together(self, priorities: list[int], values: list[int]):

        pvals = np.unique(priorities)

        for i in range(len(pvals)):
            pval = pvals[i]
            inds = np.where(priorities == pval) # Which threads are equal to the priority.
            # Compute where in the global queue line they would place at.
            cumulative_sum_inds = np.cumsum(np.where(priorities == pval, 1, 0))
            if pval not in self.priority_to_queue:
                # We need to make it.
                self.priority_to_queue[pval] = [values[inds]]
            else:
                # It already exists.
                orig_size = len(self.priority_to_queue[pval])
                new_size = orig_size + len(values[inds])
                tmp = np.zeros(new_size)
                tmp[:orig_size] = self.priority_to_queue[pval]
                tmp[orig_size:] = values[inds]
                self.priority_to_queue[pval] = tmp

    def mass_remove_one_at_a_time(self, num_to_remove: int):
        output = np.zeros(num_to_remove)

        ind = 0
        # Repeated sort can get expensive.
        for key in sorted(list(self.priority_to_queue.keys())):
            if len(self.priority_to_queue[key]) + ind < num_to_remove:
                # Add all of them.
                output[ind: ind + len(self.priority_to_queue[key])] = self.priority_to_queue[key]
                ind += len(self.priority_to_queue[key])
                # Delete the key now.
                del self.priority_to_queue[key]
            else:
                # Greater so we need to partially add.
                output[ind: num_to_remove] = self.priority_to_queue[key][:num_to_remove - ind]
                # Remove excess.
                self.priority_to_queue[key] = self.priority_to_queue[key][num_to_remove - ind:]
                break

        return output
