import json
import glob
import numpy as np

def average_jsons(path_pattern):
    files = glob.glob(path_pattern)
    data_list = [json.load(open(f)) for f in files]

    # Initialize accumulator
    avg = {}

    keys = data_list[0].keys()

    for key in keys:
        values = [d[key] for d in data_list]


        output = sum_the_rounds(values)

        avg[key] = np.mean(output, axis=0) # Average across trials.
    return avg

def sum_the_rounds(values):
    if not isinstance(values, list):
        return np.array(values)

    if not isinstance(values[0], list):
        return np.array(values)
    output = np.zeros((len(values), len(values[0])))
    for i in range(len(values)):
        for j in range(len(values[i])):
            acc = 0
            if isinstance(values[i][j], list):
                for k in range(len(values[i][j])):
                    acc += values[i][j][k]
                output[i,j] = acc
            else:
                output[i,j] = np.array(values[i][j])
    return output

import matplotlib.pyplot as plt

def plot_figure_of_averages_over_trials(list_of_average_maps, key):


    x_values = [mlist["n_input_exprs"] for mlist in list_of_average_maps]
    plt.figure()

    plt.title(f"Time Per Iteration for {list_of_average_maps[0]["n_rules"]} Rules")
    plt.xlabel("Number of Expressions")
    plt.ylabel("Average Kernel Time (s)")

    for i in range(5): # Assume 5 time steps.
        values = [mlist[key][i] for mlist in list_of_average_maps]
        plt.scatter(x_values, values, label=f"Loop {i}")

    plt.grid(1)
    plt.legend()
    plt.savefig("time_to_per_iteration.png")

def plot_figure_of_averages_over_trials_by_rules(list_of_average_maps_by_rule_set, key):

    x_values = [mlist["n_rules"] for mlist in list_of_average_maps_by_rule_set]
    plt.figure()

    plt.title(f"Time To Apply Matches for {list_of_average_maps_by_rule_set[0]["n_input_exprs"]} Expressions")
    plt.xlabel("Number of Rules")
    plt.ylabel("Average Kernel Time (s)")


    for i in range(5): # Assume 5 time steps.
        values = [mlist[key][i] for mlist in list_of_average_maps_by_rule_set]
        plt.scatter(x_values, values, label=f"Loop {i}")

    plt.grid(1)
    plt.legend()
    plt.savefig("time_to_apply_by_rules.png")