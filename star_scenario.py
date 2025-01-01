from data import collect_infos_from_instance
from utils import calculate_distance_matrix
import matplotlib.pyplot as plt


def get_costs_star_scenario(hub_coord, clients, demands, car_capacity, car_co2):
    dist = [distance((hub_coord, clients[j]) for j in range(len(clients))]
    co2_cost = 0
    means_transport = []
    for j in range (len(clients)): 
        #condition for bike, <= 500kg, <= 6
            if demands[j] <= 500 and dist[j] * 2 <= 6: # take bike
            #bike_routes.append(route)
                means_transport.append(0)
                continue
            else: #take car
                means_transport.append(1)
                total_load= demands[j]
                while total_load > 0:
                    load = min(demands[j], capacity) # if the truck can not carry all demands -> multiple routes to the same client
                    co2_cost += co2_car_emission * load * dist[j]+ 0.2*2*dist[j]
                    total_load -= load  # reeduce the remaining demand
    return co2_cost, means_transport


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
    car_co2 = 0.772   # kg per 1km for 1t 
    car_capacity = 1500
    bike_capacity = 100
    

    for i in range(1, number_instances + 1):
        instance = ""
        if i < 10:
            instance += "0"
        instance += str(i)
        dimension, capacity, indices, clients, demands = collect_infos_from_instance(instance)  # prepare instance
        costs, mean_transport = get_costs_star_scenario(clients[0], clients[1:], demands[1:], car_capacity, car_co2)
        print(str(i) + ":")
        print(round(costs, 3))
        print("number_bikes", mean_transport.count(0))
        print("number_cars", mean_transport.count(1))
        plot_routes(clients[0], clients[1:], mean_transport, save_path="results/star_scenario_" + instance + ".png")
