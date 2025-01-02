from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from data import collect_infos_from_instance
from utils import prepare_clients_to_plot, calculate_distance_matrix, get_hub_id

"""
nearest neighbor heuristic with optional sweep phase (boolean sweep is true or false)
"""


def nearest_neighbor(hub_coord, vertices, client_demands, capacity, truck_co2, empty_truck_weight, sweep):
    demands = client_demands.copy()
    hub_id = get_hub_id(hub_coord, vertices)
    dist_matrix = calculate_distance_matrix(vertices)
    num_points = dist_matrix.shape[0]
    visited = demands  # if the demands are 0 then visited
    routes = []
    costs_2 = 0
    visited[hub_id] = 0  # visited -> 0

    while demands.count(0) < num_points:
        current_node = hub_id  # start at hub
        current_capacity = 0
        route = []
        while True:
            nearest = None
            min_dist = float('inf')  # at first largest value
            new_demand_after_visit = 0
            new_load = 0
            neighbor_candidates = [idx for idx, value in enumerate(visited) if value > 0]  # all clients
            if sweep:
                # Calculate polar angles and sort points by angle
                polar_angles = calculate_polar_angles(vertices)
                sorted_indices = np.argsort(polar_angles)
                neighbor_candidates = [i for i in sorted_indices if visited[i] > 0]  # Sweep phase, not visited = 0
            for neighbor in neighbor_candidates:  # depends on sweep boolean if sorted or not
                if dist_matrix[current_node, neighbor] < min_dist:
                    nearest = neighbor
                    new_load = demands[nearest]
                    min_dist = dist_matrix[current_node, neighbor]
                    if demands[
                        nearest] > capacity and capacity - new_load > 0:  # if demands are in general higher, then you need multiple tours
                        new_load = capacity - new_load
                        new_demand_after_visit = (demands[
                                                      neighbor] - new_load)  # still demands left because just first tour with vehicle
                    elif demands[neighbor] + current_capacity > capacity:
                        continue
            if nearest is None or current_capacity >= capacity:  # No feasible neighbor or fully loaded
                # End the current route
                if current_node != hub_id:  # If not already at the hub
                    costs_2 += dist_matrix[current_node, hub_id] * empty_truck_weight  # Return trip emission
                routes.append(route)  # Save the route
                break  # Exit the inner while-loop to start a new route
            route.append(vertices[nearest])
            visited[nearest] = new_demand_after_visit
            current_capacity += new_load
            current_node = nearest
            costs_2 += truck_co2 * (
                    current_capacity + empty_truck_weight) * min_dist  # truckco2 1t per 1km -> 1kg per 1km
    costs_2 = costs_2 / 1000  # in kg
    return routes, costs_2


def calculate_polar_angles(vertices):
    """Calculate polar angles of all points relative to the depot."""
    vertices_arr = np.array(vertices)
    deltas = (vertices_arr - vertices_arr[0]).tolist()  # Subtract depot and convert to list
    angles = np.arctan2([delta[1] for delta in deltas], [delta[0] for delta in deltas])  # y, x -> polar angles
    return angles


# Function to calculate savings
def calculate_savings(dist_matrix: np.ndarray) -> List[Tuple[float, int, int]]:
    """Calculate savings for merging routes."""
    num_points = dist_matrix.shape[0]
    savings = []
    for i in range(1, num_points):  # Exclude depot (node 0)
        for j in range(i + 1, num_points):
            # Savings formula: s(i, j) = d(i, depot) + d(j, depot) - d(i, j)
            saving = dist_matrix[i, 0] + dist_matrix[j, 0] - dist_matrix[i, j]
            savings.append((saving, i, j))
    # Sort savings in descending order
    savings.sort(reverse=True, key=lambda x: x[0])
    return savings


def clarke_and_wright(dist_matrix: np.ndarray, demands: List[int], capacity: int, truck_co2: float,
                      empty_truck_weight: float, vertices: List[Tuple[float, float]]):
    """Clarke and Wright Savings Algorithm (Optimized for continuous routes)."""
    num_points = dist_matrix.shape[0]
    # Initial routes for each customer, now storing coordinates
    routes = [[vertices[i]] for i in range(1, num_points)]  # Initial routes for each customer
    route_loads = {tuple(vertices[i]): demands[i] for i in range(1, num_points)}  # Track route loads by coordinates
    total_emission = 0  # Emission tracking

    # Calculate savings
    savings = calculate_savings(dist_matrix)

    # Merge routes based on savings
    for saving, i, j in savings:
        # Find the routes containing i and j (using coordinates instead of indices)
        route_i = next((route for route in routes if vertices[i] in route), None)
        route_j = next((route for route in routes if vertices[j] in route), None)

        # Check if i and j are in different routes and merging is feasible
        if route_i is None or route_j is None or route_i == route_j:
            continue

        # Check if the new merged route exceeds capacity
        if route_loads[tuple(route_i[0])] + route_loads[tuple(route_j[0])] <= capacity:
            # Merge the routes in a continuous sequence
            if route_i[-1] == vertices[i] and route_j[0] == vertices[j]:  # Tail-to-head
                route_i.extend(route_j)
                routes.remove(route_j)

            elif route_i[0] == vertices[i] and route_j[-1] == vertices[j]:  # Head-to-tail
                route_j.extend(route_i)
                routes.remove(route_i)

            elif route_i[-1] == vertices[i] and route_j[-1] == vertices[j]:  # Tail-to-tail
                route_i.extend(route_j[::-1])
                routes.remove(route_j)

            elif route_i[0] == vertices[i] and route_j[0] == vertices[j]:  # Head-to-head
                route_j.reverse()
                route_j.extend(route_i)

            # Update route loads and remove merged route
            route_loads[tuple(route_i[0])] += route_loads[tuple(route_j[0])]

    # Ensure all customers are served (if any customer is unassigned, assign to a new route)
    unassigned_customers = [i for i in range(1, num_points) if not any(vertices[i] in route for route in routes)]
    for customer in unassigned_customers:
        routes.append([vertices[customer]])

    # Calculate emissions and finalize routes
    for route in routes:
        current_capacity = 0
        for i in range(0, len(route) - 1):
            # Get the coordinates of the points in the route
            current_capacity += demands[vertices.index(route[i + 1])]
            total_emission += truck_co2 * (current_capacity + empty_truck_weight) * dist_matrix[
                vertices.index(route[i]), vertices.index(route[i + 1])]
        # Add return to depot
        total_emission += truck_co2 * empty_truck_weight * dist_matrix[vertices.index(route[len(route) - 1]), 0]

    total_emission = total_emission / 1000  # Convert grams to kilograms
    return routes, total_emission


def plot_tour_planning(vertices: List[Tuple[float, float]], routes: List[List[int]], save_path=None):
    plt.figure(figsize=(10, 10))
    # Plot depot as a red point
    plt.scatter(vertices[0][0], vertices[0][1], color='red', s=150, label='Depot')

    # Plot routes
    for route in routes:
        # Add depot (node 0) to the beginning and end of the route
        route_with_depot = [vertices[0]] + route + [vertices[0]]
        for i in range(len(route_with_depot) - 1):
            start = route_with_depot[i]
            end = route_with_depot[i + 1]

            # Draw the route with arrows
            plt.annotate("", xy=end, xytext=start,
                         arrowprops=dict(arrowstyle="->", color='green', lw=2))

    # Draw all customer nodes
    for idx, vertex in enumerate(vertices):
        if idx != 0:  # Skip depot (node 0)
            plt.scatter(*vertex, color='blue', s=100)
            plt.text(vertex[0] + 0.1, vertex[1] + 0.1, str(idx), fontsize=10, color="black")

    plt.axis('equal')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Tour Planning Visualization")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    number_instances = 12

    truck_co2 = 311 / 1000  # g per 1km for 1t -> g per km per kg
    empty_truck_weight = 30000  # 3t per truck
    for i in range(1, number_instances + 1):
        car_routes = []
        bike_routes = []
        instance = ""
        if i < 10:
            instance += "0"
        instance += str(i)
        dimension, capacity, indices, clients, demands = collect_infos_from_instance(
            instance)  # prepare instance# vertices without routes
        prepare_clients_to_plot(clients)
        dist_matrix = calculate_distance_matrix(clients)

        # route with nearest neighbor heuristic
        routes, costs_nn = nearest_neighbor(clients[0], clients, demands, capacity, truck_co2, empty_truck_weight,
                                            False)
        print(instance + ": ")
        print("NN:")
        print(round(costs_nn, 3))
        plot_tour_planning(clients, routes, "results/" + "NN_" + instance + ".png")

        routes_sn, costs_sn = nearest_neighbor(clients[0], clients, demands, capacity, truck_co2, empty_truck_weight,
                                             True)
        print("SN")
        print(round(costs_sn, 3))
        plot_tour_planning(clients, routes_sn, "results/" + "SN_" + instance + ".png")

        routes_cw, total_emission = clarke_and_wright(dist_matrix, demands, capacity, truck_co2, empty_truck_weight,
                                                      clients)
        print("CW")
        print(round(total_emission, 3))
        plot_tour_planning(clients, routes_cw, "results/" + "CW_" + instance + ".png")
