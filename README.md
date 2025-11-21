# Gpu Equality Saturation

# Building and running

1) Edit CUDA_ARCHITECTURES in CMakeLists.txt to reflect the compute architecture of your Nvidia GPU. For example, an RTX 6000 Ada GPU would be 89 and a 5090 would be 120.
2) From a new "build" directory, run `cmake ..` and `make`
3) The program should be run from the root directory of the repo, as `./build/main`

## Related

### Egg 
Many of the design elements of our equality saturation tool come from Egg ([website](https://egraphs-good.github.io/) and [paper](https://dl.acm.org/doi/10.1145/3434304)), including
distinct match-lhs, apply-rhs, and repair stages. Notably, where Egg's repair stage was motivated by deduplicating work lists, we are motivated by limitations of the GPU to simplify
data structure operations and concurrency.

### Herbie
Our test suite `bench/` and rule set `rules.txt` are derived from Herbie: [https://github.com/herbie-fp/herbie](https://github.com/herbie-fp/herbie)

## E-graph Format
* E-classes are a:
  1) A (blocked) linked list pointing to member e-nodes
  2) A (blocked) linked list of e-class IDS who have e-nodes with edges into this e-class
* E graphs are:
  1) A flat expanding array of e-classes, allocated to a budget
  2) A flat expanding array of e-nodes, allocated to a budget.
  3) A union-find mapping e-class IDs to their newest union result.
  4) A hash-cons of nodes in the e-graph. The hash function depends only on the opcode and the IDs of each operand e-class. Supports removal/re-key

## Exploration

Exploration is divided into 3 phases, which are looped in sequence until fixpoint or the budget is exhausted. These phases are:

* Match: threads and blocks match rules onto an immutable graph, and records what operations should be done to apply them later.
* Apply: After all matches have been found, mass apply operations to the egraph, deferring invariant repair
* Repair: Following Egg, repair the e-graph invariants damaged by the apply step. 

### Preprocessing

* The ruleset is loaded from the CPU into read-only memory
    * Format: 
        1) A flat array of expr nodes
        2) A flat array of rule metadata (lhs node root idx, rhs node root idx)
* Input expression trees are into initial e-graphs
* Start at the repair step.

### Match

* Decompose the (per node, per rule) loop so that a single warp has only one node assigned, each thread has possibly multiple rules assigned, and a node may have multiple blocks.
* For each match, record the (e-class ID, rhs_rule_node_ID, var_bindings) for the apply section. Each rule has no more than 4 variables, numbered 0-3, so `var_bindings` is a fixed-sized array.

### Apply

* Distribute sections of the `apply` list over blocks. 
* Important: while inserting, multiple different blocks may try to add the same new node. It is ok if they both take up a slot in the memory buffer, but only one should be added to any e-class. This can be done with an atomic op while adding to the hash-cons that returns the e-class only of the first thread to insert that same block.

### Repair


## Limitations

* In general, the same rule may match to the same e-graph node multiple times, for multiple choices of values. This is tricky to implement, and we currently take 
