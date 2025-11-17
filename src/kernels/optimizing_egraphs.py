import numpy as np
import scipy.optimize

# In this problem you have a decision variable for each of the nodes within an eclass and for each eclass.
# if an eclass is chosen one of the nodes must be as well.
# the variables must be {0,1}
# Assume that the objective function is to minimize the cost.

num_classes = 4
num_elm_per_class = 2
costs = np.ones(num_classes * num_elm_per_class)

# Assume each node within the same class will have the same cost.

def add_constraint_for_class(class_id, class_id_to_nodes_in_class, total_number_of_variables):
    """
    This constraint is an equality constraint that must be zero.
    """
    output = np.zeros(total_number_of_variables)
    bounds = 0
    output[class_id] = 1
    for x in class_id_to_nodes_in_class[class_id]:
        output[x] = -1

    return output, bounds

def add_constraint_for_nodes_to_class(node, node_to_classes_it_needs, total_number_of_variables):

    output = np.zeros(total_number_of_variables)
    bounds = 0
    output[node] = len(node_to_classes_it_needs[node])
    for x in node_to_classes_it_needs[node]:
        output[x] = -1
    return output, bounds

def constraint_for_needing_the_full_system(total_number_of_variables):
    out = np.zeros(total_number_of_variables)
    out[0] = 1
    return out

def add_all_constraints_for_nodes_to_classes_need_by_that_node(start_point_of_nodes, node_to_classes_it_needs, total_number_of_variables):

    ub_constraints = []
    ub_rhs = []
    for i in range(start_point_of_nodes, total_number_of_variables):
        tmp, b = add_constraint_for_nodes_to_class(i, node_to_classes_it_needs, total_number_of_variables)
        ub_constraints.append(tmp)
        ub_rhs.append(b)

    return ub_constraints, ub_rhs

def add_all_constraints_to_pick_one_node_in_the_class(num_classes, id_to_nodes_in_class, total_number_of_variables):
    equal_constraints = []
    equal_rhs = []
    for i in range(num_classes):
        tmp, b = add_constraint_for_class(i, id_to_nodes_in_class, 9)
        equal_constraints.append(tmp)
        equal_rhs.append(b)
        
    equal_constraints.append(constraint_for_needing_the_full_system(total_number_of_variables))
    equal_rhs.append(1)
    return equal_constraints, equal_rhs

id_to_nodes_in_class = {}
id_to_nodes_in_class[0] = [4]
id_to_nodes_in_class[1] = [5,6]
id_to_nodes_in_class[2] = [7]
id_to_nodes_in_class[3] = [8]

equal_constraints = []
equal_rhs = []
ub_constraints = []
ub_rhs = []
equal_constraints, equal_rhs = add_all_constraints_to_pick_one_node_in_the_class(4, id_to_nodes_in_class, 9)

node_to_classes_it_needs = {4: [1], 5:[2,3], 6:[2], 7: [3], 8:[]}
ub_constraints, ub_rhs = add_all_constraints_for_nodes_to_classes_need_by_that_node(4, node_to_classes_it_needs, 9)



costs = np.random.random(9)
costs[:4] = 0

bounds = (0,1)
