import numpy as np
import matplotlib.pyplot as plt

from data import collect_infos_from_instance
from star_scenario import get_costs_star_scenario
from tour_routing import nearest_neighbor
from utils import get_costs_car_bike, prepare_clients_to_plot, calculate_distance_matrix
from UFL import greedy_heuristic_with_demand, plot_combined_routes_and_tours


def plot_emissions_per_instance(node_counts, capacity_list, emissions_by_version, parameter_settings,
                                save_path=None):
    """
    Plots CO2 emissions for different parameter settings against the sorted x-axis (nodes or capacity).

    :param node_counts: List of node counts for 12 instances.
    :param capacity_list: List of capacities for 12 instances.
    :param emissions_by_version: List of lists containing CO2 emissions for each version.
                                 Each inner list corresponds to a specific parameter setting.
    :param parameter_settings: List of parameter settings (used for labeling different colors).
    :param sort_key: Key to sort the x-axis, either "nodes" or "capacity".
    :param save_path: Optional; Path to save the plot as an image. If None, shows the plot.
    """
    if len(emissions_by_version) != len(node_counts):
        raise ValueError("The number of parameter settings must match the number of emission versions.")
    if len(node_counts) != len(capacity_list):
        raise ValueError("The lengths of node counts and capacity list must match.")

    # Lexicographic sorting: First by node_counts, then by capacity_list
    combined_sort_keys = list(zip(node_counts, capacity_list))
    sort_indices = sorted(range(len(combined_sort_keys)),
                          key=lambda idx: (combined_sort_keys[idx][0], combined_sort_keys[idx][1]))
    sorted_x_nodes = np.array(node_counts)[sort_indices]
    sorted_x_capacity = np.array(capacity_list)[sort_indices]

    sorted_emissions_by_version = [emissions_by_version[idx] for idx in sort_indices]
    """
    for emissions in emissions_by_version:
        print(emissions)
        # Sort emissions according to the indices
        sorted_emissions = [emissions[int(idx)] for idx in sort_indices]
        sorted_emissions_by_version.append(sorted_emissions)
        print(sorted_emissions_by_version)
        """
    # Transpose sorted_emissions_by_version for parameter-wise plotting
    sorted_emissions_by_parameter = list(map(list, zip(*sorted_emissions_by_version)))

    plt.figure(figsize=(12, 8))

    color = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']
    # Plot each parameter setting with a different color and label
    for emissions, param, color in zip(sorted_emissions_by_parameter, parameter_settings, color):
        plt.plot(emissions, marker='o', color=color, label=f'Parameter = {param}')

    plt.xlabel("Number of Nodes (Sorted) and Capacity (Secondary)", fontsize=14)
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


if __name__ == '__main__':
    number_instances = 10
    car_co2 = 772 / 1000  # g per 1km for 1t -> g per 1km for 1kg
    car_capacity = 1500
    bike_capacity = 100
    empty_car_weight = 15000  # in kg
    truck_co2 = 311 / 1000  # g per 1km for 1t -> g per km per kg
    empty_truck_weight = 30000  # 3t per truck
    costs_of_instances = []
    min_assigned_clients = [2, 5, 10, 20, 50, 100]

    for i in range(1, number_instances + 1):
        instance = ""
        if i < 10:
            instance += "0"
        instance += str(i)
        print(instance + ": ")
        costs_for_parameter = []
        for min_clients in min_assigned_clients:
            print(min_clients)
            dimension, capacity, indices, vertices, demands = collect_infos_from_instance(
                instance)  # prepare instance# vertices without routes
            if min_clients > dimension:
                costs_for_parameter.append(0)
                continue
            prepare_clients_to_plot(vertices)
            dist_matrix = calculate_distance_matrix(vertices)
            candidates = vertices  # clients + hub
            hub = vertices[0]
            clients = vertices[1:]
            demands = demands[1:]
            cost_client_car_bike = get_costs_car_bike(clients, candidates, demands, capacity, dist_matrix)
            total_cost, open_candidates, client_assign, client_assignments_idx = greedy_heuristic_with_demand(
                candidates, clients, demands, cost_client_car_bike, min_clients)
            print(total_cost)
            # 1: [0,1,0,1,..], 1=hub
            # 2: [in cluster of hub idx, in cluster of hub idx, ...]

            # scenario 1: calculate demands of each hub: sum of all demands in cluster
            demands_of_cluster = []
            demands_of_cluster_as_list = []
            costs_of_cluster = []
            clusters = []
            hub_ids = [idx for idx, value in enumerate(open_candidates) if value == 1]
            cluster_transport_list = []
            for hub in hub_ids:
                transport_list = []
                client_in_cluster = [idx for idx, value in enumerate(client_assignments_idx) if hub == value]
                clusters.append(client_in_cluster)
                total_demand = demands[hub]  # need this for scenario 2 as demands of each hub + hub itself
                total_demand_list = []
                for client in client_in_cluster:
                    total_demand += demands[client]
                    total_demand_list.append(demands[client])
                demands_of_cluster.append(total_demand)
                demands_of_cluster_as_list.append(total_demand_list)
                coord_of_client = [clients[idx] for idx in client_in_cluster]
                if len(coord_of_client) <= 1:  # no costs at all
                    costs_of_cluster.append(0.0)
                    continue
                total_cost_of_cluster, transport_list = get_costs_star_scenario(clients[hub], coord_of_client,
                                                                                total_demand_list, car_capacity,
                                                                                car_co2, empty_car_weight)
                cluster_transport_list.append(transport_list)
                costs_of_cluster.append(total_cost_of_cluster)
            print(costs_of_cluster)
            # scenario 2_ routing through hubs
            coord_of_hubs = [candidates[idx] for idx in hub_ids]
            route, tour_costs = nearest_neighbor(coord_of_hubs[0], coord_of_hubs, demands_of_cluster, capacity,
                                                 truck_co2,
                                                 empty_truck_weight, True)
            summed_costs = 0
            for cluster_costs in costs_of_cluster:
                summed_costs += cluster_costs
            summed_costs += tour_costs
            print(round(summed_costs, 3))
            costs_for_parameter.append(summed_costs)
            # plot_combined_routes_and_tours(hub_ids, clusters, cluster_transport_list, candidates, route)
        costs_of_instances.append(costs_for_parameter)
    dimension_list = [32, 60, 31, 50, 19, 60, 101, 101, 101, 101]  # , 3001, 4001]  # number of nodes of each instance
    capacity_list = [10000, 10000, 10000, 10000, 16000, 12000, 14090, 18420, 20430, 12970]  # , 10000,
    # 15000]  # capacity of each instance
    emissions_by_version = costs_of_instances
    parameter_settings = min_assigned_clients  # how many clients one hub serves at least
    plot_emissions_per_instance(dimension_list, capacity_list, emissions_by_version, parameter_settings)
