from data import plot_emissions_per_instance
from data import collect_infos_from_instance
from utils import calculate_distance_matrix, distance
import matplotlib.pyplot as plt


def get_costs_star_scenario(hub_coord, clients, demands, car_capacity, car_co2):
    emitted_value_only_car = 200  # g per 1km, found in recherche
    dist = [distance(hub_coord, clients[j]) for j in range(len(clients))]
    co2_cost = 0
    means_transport = []
    for j in range(len(clients)):
        # condition for bike, <= 500kg, <= 6
        if demands[j] <= 500 and dist[j] * 2 <= 6:  # take bike
            # bike_routes.append(route)
            means_transport.append(0)
            continue
        else:  # take the car
            means_transport.append(1)
            total_load = demands[j]
            while total_load > 0:
                load = min(demands[j],
                           car_capacity)  # if the truck can not carry all demands -> multiple routes to the same client
                co2_cost += car_co2 * load * dist[j] + emitted_value_only_car * 2 * dist[
                    j]  # emissions to client + both ways co2 for only car
                total_load -= load  # reduce the remaining demand
    return co2_cost / 1000, means_transport


def get_costs_star_scenario_emptycar_weight(hub_coord, clients, demands, car_capacity, car_co2, empty_car_weight):
    means_transport = []  # 0 for bike, 1 for car
    dist_matrix = calculate_distance_matrix(clients)
    total_costs = 0
    hub_id = 0
    for id in range(0, len(clients)):
        if hub_coord == clients[id]:
            hub_id = id
            break
    for i in range(0, len(clients)):
        if clients[i] == hub_coord:
            means_transport.append(-1)  # hub
            continue
        # condition for bike, <= 500kg, <= 6
        if demands[i] <= 500 and dist_matrix[hub_id][i] * 2 <= 6:  # take bike
            means_transport.append(0)
            continue
        else:  # take the car
            total_load = demands[i]
            while total_load > 0:
                load = min(total_load,
                           car_capacity)  # if the truck can not carry all demands -> multiple routes to the same client
                total_costs += car_co2 * (load + empty_car_weight) * dist_matrix[hub_id][i]
                total_costs += car_co2 * empty_car_weight * dist_matrix[i][hub_id]  # emissions to return to depot
                total_load -= load  # reduce the remaining demand
            means_transport.append(1)
    total_costs = total_costs / 1000
    return total_costs, means_transport


def plot_routes(depot, clients, means_transport, save_path=None):
    """
    Plots the depot, clients, and routes, distinguishing between bike and car routes.

    Args:
        depot (tuple): Coordinates of the depot (x, y).
        clients (list of tuple): List of coordinates for each client [(x1, y1), (x2, y2), ...].
        means_transport (list of int): List indicating transport means (0 for bike, 1 for car).
    """
    plt.figure(figsize=(10, 6))

    # Plot the depot
    plt.scatter(*depot, color='blue', label='Depot', s=200, edgecolor='black', zorder=3)

    # Flags for labels (to avoid duplicate labels in the legend)
    bike_route_added = False
    car_route_added = False
    bike_client_added = False
    car_client_added = False

    # Plot clients and routes
    for i, client in enumerate(clients):
        if means_transport[i] == 0:  # Bike route
            if not bike_route_added:
                plt.plot([depot[0], client[0]], [depot[1], client[1]], color='green', linestyle='--',
                         label='Bike Route')
                bike_route_added = True
            else:
                plt.plot([depot[0], client[0]], [depot[1], client[1]], color='green', linestyle='--')

            if not bike_client_added:
                plt.scatter(*client, color='green', label='Client (Bike)', s=100, edgecolor='black', zorder=2)
                bike_client_added = True
            else:
                plt.scatter(*client, color='green', s=100, edgecolor='black', zorder=2)
        elif means_transport[i] == 1:  # Car route
            if not car_route_added:
                plt.plot([depot[0], client[0]], [depot[1], client[1]], color='orange', linestyle='-', label='Car Route')
                car_route_added = True
            else:
                plt.plot([depot[0], client[0]], [depot[1], client[1]], color='orange', linestyle='-')

            if not car_client_added:
                plt.scatter(*client, color='orange', label='Client (Car)', s=100, edgecolor='black', zorder=2)
                car_client_added = True
            else:
                plt.scatter(*client, color='orange', s=100, edgecolor='black', zorder=2)

    # Adding labels, legend, and grid
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Routes Visualization", )
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    number_instances = 12
    car_co2 = 0.772  # kg per 1km for 1t
    car_capacity = 1500
    bike_capacity = 100
    empty_car_weight = 15000  # 1,5t per car
    emissions_per_instance = []
    for i in range(1, number_instances + 1):
        instance = ""
        if i < 10:
            instance += "0"
        instance += str(i)
        dimension, capacity, indices, clients, demands = collect_infos_from_instance(instance)  # prepare instance
        costs, mean_transport = get_costs_star_scenario(clients[0], clients[1:], demands[1:], car_capacity, car_co2)
        print(str(i) + ":")
        print(round(costs, 3))
        plot_routes(clients[0], clients[1:], mean_transport, save_path=f"results/star_scenario_{instance}.png")
        emissions_per_instance.append(round(costs, 3))
    dimension_list = [32, 60, 31, 50, 19, 60, 101, 101, 101, 101, 3001, 4001]  # number of nodes of each instance
    capacity_list = [10000, 10000, 10000, 10000, 16000, 12000, 14090, 18420, 20430, 12970, 10000,
                     15000]  # capacity of each instance
    print(emissions_per_instance)
    plot_emissions_per_instance(dimension_list, capacity_list, emissions_per_instance, [],
                                "results/emissions_1.png")
