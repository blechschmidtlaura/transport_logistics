import numpy as np
from matplotlib import pyplot as plt


def collect_infos_from_instance(instance):
    clients = []
    demands = []
    path = "instances/" + instance + ".txt"
    with open(path) as f:
        line = f.readline()
        row = line.split(' ')
        dimension = int(row[-1])
        line = f.readline()
        row = line.split(' ')
        capacity = int(row[-1])
        indices = []
        f.readline()
        for i in range(dimension):
            line = f.readline()
            if line == '':
                break
            row = line.split(' ')
            indices.append(int(row[0]))
            clients.append([float(row[1]), float(row[2])])
        f.readline()
        for i in range(dimension):
            line = f.readline()
            if line == '':
                break
            row = line.split(' ')
            demands.append(int(row[1]))
    f.close()
    return dimension, capacity, indices, clients, demands

def plot_emissions_per_instance2(node_counts, capacity_list, emissions_by_version, parameter_settings,
                                save_path=None):
    """plot CO2 emissions for different parameter settings and instance configurations."""

    # Lexicographic sorting: First by node_counts, then by capacity_list
    combined_sort_keys = list(zip(node_counts, capacity_list))
    sort_indices = sorted(range(len(combined_sort_keys)),
                          key=lambda idx: (combined_sort_keys[idx][0], combined_sort_keys[idx][1]))
    sorted_x_nodes = np.array(node_counts)[sort_indices]
    sorted_x_capacity = np.array(capacity_list)[sort_indices]
    sorted_emissions_by_version = [emissions_by_version[idx] for idx in sort_indices]
    sorted_emissions_by_parameter = list(map(list, zip(*sorted_emissions_by_version)))

    plt.figure(figsize=(12, 8))
    color = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']
    # Plot each parameter setting and instance with a different color and label
    for emissions, param, color in zip(sorted_emissions_by_parameter, parameter_settings, color):
        plt.plot(emissions, marker='o', color=color, label=f'Parameter = {param}')

    plt.xlabel("Number of Nodes and Capacity", fontsize=14)
    plt.ylabel("CO2 Emissions (kg)", fontsize=14)
    plt.title("CO2 Emissions changing with number of needed clients for hub", fontsize=16)
    plt.legend(title="Parameter Settings", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Create custom x-ticks combining nodes and capacities
    x_ticks_labels = [f"{n} ({c})" for n, c in zip(sorted_x_nodes, sorted_x_capacity)]
    plt.xticks(ticks=range(len(x_ticks_labels)), labels=x_ticks_labels, rotation=45)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_emissions_per_instance(node_counts, capacity_list, emissions_by_version, parameter_settings,
                                 save_path=None):
    # Lexicographic sorting: First by node_counts, then by capacity_list
    combined_sort_keys = list(zip(node_counts, capacity_list))
    sort_indices = sorted(range(len(combined_sort_keys)),
                          key=lambda idx: (combined_sort_keys[idx][0], combined_sort_keys[idx][1]))
    sorted_x_nodes = np.array(node_counts)[sort_indices]
    sorted_x_capacity = np.array(capacity_list)[sort_indices]
    sorted_emissions_by_version = [emissions_by_version[idx] for idx in sort_indices]
    plt.figure(figsize=(12, 8))

    if len(parameter_settings) == 0:
        plt.plot(sorted_emissions_by_version, marker='o')
    else:
        sorted_emissions_by_parameter = list(map(list, zip(*sorted_emissions_by_version)))
        color = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']
        color = color[:len(parameter_settings)]
        # Plot each parameter setting with a different color and label
        for emissions, param, color in zip(sorted_emissions_by_parameter, parameter_settings, color):
            plt.plot(emissions, marker='o', color=color, label=f'Parameter = {param}')
        plt.legend(title="Parameter Settings", fontsize=12)

    plt.xlabel("Number of Nodes and Capacity", fontsize=14)
    plt.ylabel("CO2 Emissions (kg)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Create custom x-ticks combining nodes and capacities
    x_ticks_labels = [f"{n} ({c})" for n, c in zip(sorted_x_nodes, sorted_x_capacity)]
    plt.xticks(ticks=range(len(x_ticks_labels)), labels=x_ticks_labels, rotation=45)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()
