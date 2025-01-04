import numpy as np

from data import collect_infos_from_instance
from star_scenario import get_costs_star_scenario
from UFL import heuristic_big_instances, greedy_heuristic_with_demand
from tour_routing import nearest_neighbor, clarke_and_wright
from utils import get_costs_car_bike, prepare_clients_to_plot, calculate_distance_matrix
from typing import Tuple, List

from matplotlib import pyplot as plt


def plot_combined_routes_and_tours(depot: Tuple[float, float],
                                   clients: List[Tuple[float, float]],
                                   means_transport: List[List[int]],
                                   vertices: List[Tuple[float, float]],
                                   routes: List[int],
                                   save_path=None):
    """
    Combines plotting of routes (bike and car) and tour planning.
    """
    plt.figure(figsize=(12, 10))
    # Flags for labels to avoid duplicate legend entries
    bike_route_added = False
    car_route_added = False
    bike_client_added = False
    car_client_added = False
    depot_added = False
    # Plot bike and car routes from hubs to clients
    for hub_idx, hub in enumerate(depot):
        for client_idx, client in enumerate(clients[hub_idx]):
            transport_modes = means_transport[hub_idx]
            if not transport_modes:  # No clients assigned to the hub
                break
            if transport_modes[client_idx] == 0:  # Bike route
                if not bike_route_added:
                    plt.plot([vertices[hub][0], vertices[client][0]], [vertices[hub][1], vertices[client][1]],
                             color='green', linestyle='--',
                             label='Bike Route')
                    bike_route_added = True
                else:
                    plt.plot([vertices[hub][0], vertices[client][0]], [vertices[hub][1], vertices[client][1]],
                             color='green', linestyle='--')
                if not bike_client_added:
                    plt.scatter(vertices[client][0], vertices[client][1], color='green', label='Client (Bike)', s=100,
                                edgecolor='black', zorder=2)
                    bike_client_added = True
                else:
                    plt.scatter(vertices[client][0], vertices[client][1], color='green', s=100, edgecolor='black',
                                zorder=2)
            elif transport_modes[client_idx] == 1:  # Car route
                if not car_route_added:
                    plt.plot([vertices[hub][0], vertices[client][0]], [vertices[hub][1], vertices[client][1]],
                             color='orange', linestyle='-',
                             label='Car Route')
                    car_route_added = True
                else:
                    plt.plot([vertices[hub][0], vertices[client][0]], [vertices[hub][1], vertices[client][1]],
                             color='orange', linestyle='-')
                if not car_client_added:
                    plt.scatter(vertices[client][0], vertices[client][1], color='orange', label='Client (Car)', s=100,
                                edgecolor='black', zorder=2)
                    car_client_added = True
                else:
                    plt.scatter(vertices[client][0], vertices[client][1], color='orange', s=100, edgecolor='black',
                                zorder=2)
    # Plot tour routes between hubs
    for single_route in routes:
        route_with_depot = [vertices[0]] + single_route + [vertices[0]]  # Include depot at start and end
        for i in range(len(route_with_depot) - 1):
            start = route_with_depot[i]
            end = route_with_depot[i + 1]
            plt.annotate("", xy=end, xytext=start,
                         arrowprops=dict(arrowstyle="->", color='blue', lw=2, alpha=0.7))
    for idx, vertex in enumerate(vertices):
        plt.scatter(*vertex, color='orange', s=150 if idx == 0 else 100)
        plt.text(vertex[0] + 0.1, vertex[1] + 0.1, str(idx), fontsize=10, color="black")
    for hub in depot:
        if not depot_added:
            plt.scatter(vertices[hub][0], vertices[hub][1], color='blue', label='Depots', s=150)
            depot_added = True
        else:
            plt.scatter(vertices[hub][0], vertices[hub][1], color='blue', s=150)
    # Customize the plot
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Combined Routes and Tour Planning")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_emissions_per_instance(node_counts, capacity_list, emissions_by_version, parameter_settings,
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


if __name__ == '__main__':
    number_instances = 12
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
            if i < 11:
                total_cost, open_candidates, client_assign, client_assignments_idx = greedy_heuristic_with_demand(
                    candidates, clients, demands, cost_client_car_bike, min_clients)
            else:
                total_cost, open_candidates, client_assign, client_assignments_idx, clients, warehouses = heuristic_big_instances(
                    hub, clients, demands, capacity, min_clients)
            # open_candidates: [0,1,0,1,..], 1=hub
            # client_assignments_idx: [in cluster of hub idx, in cluster of hub idx, ...]

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
            # scenario 2_ routing through hubs
            coord_of_hubs = [candidates[idx] for idx in hub_ids]
            #route, tour_costs = nearest_neighbor(coord_of_hubs[0], coord_of_hubs, demands_of_cluster, capacity,
                                                 #truck_co2,
                                                 #empty_truck_weight, True)
            dist_matrix_2 = calculate_distance_matrix(coord_of_hubs)
            route, tour_costs = clarke_and_wright(dist_matrix_2, demands_of_cluster, capacity,
                                                 truck_co2,
                                                 empty_truck_weight, coord_of_hubs)
            summed_costs = 0
            for cluster_costs in costs_of_cluster:
                summed_costs += cluster_costs
            summed_costs += tour_costs
            print(round(summed_costs, 3))
            costs_for_parameter.append(summed_costs)
            plot_combined_routes_and_tours(hub_ids, clusters, cluster_transport_list, candidates, route, "results/" + instance + "_3_combined_cw_with_minclients_" + str(min_clients) + ".png")
        costs_of_instances.append(costs_for_parameter)
    dimension_list = [32, 60, 31, 50, 19, 60, 101, 101, 101, 101, 3001, 4001]  # number of nodes of each instance
    capacity_list = [10000, 10000, 10000, 10000, 16000, 12000, 14090, 18420, 20430, 12970, 10000,
                     15000]  # capacity of each instance
    emissions_by_version = costs_of_instances
    parameter_settings = min_assigned_clients  # how many clients one hub serves at least
    plot_emissions_per_instance(dimension_list, capacity_list, emissions_by_version, parameter_settings,
                                "results/emissions_3_assigned_clients_cw.png")
