from typing import Tuple, List

from matplotlib import pyplot as plt

from data import collect_infos_from_instance
from utils import get_costs_car_bike, distance, distance_matrix, split_and_select


def index(elt, list):
    for i in range(len(list)):
        if list[i] == elt:
            return i
    return -1  # Retourne -1 si l'élément n'est pas trouvé


def best_distribution(candidate, candidates, clients, demands, assignments, nb_client_candidate, cost_client_car_bike):
    # Create copies of the current assignments and client counts for modification
    nb_client_candidat_copy = nb_client_candidate.copy()
    assignmentclients_candidate = assignments.copy()
    saving_candidat = 0  # Initialize the CO2 savings for the current candidate
    demands_candidat = 0  # Initialize the demand associated with the candidate
    id1 = index(candidate, candidates)  # Get the index of the new refrigerator candidate
    for j in range(len(clients)):
        id = index(assignmentclients_candidate[j], candidates)  # Get the index of the current assignment for client j
        # Check if assigning the client to the new candidate reduces CO2 emissions
        if cost_client_car_bike[id][j] > cost_client_car_bike[id1][j]:
            saving_candidat += -cost_client_car_bike[id][j] + cost_client_car_bike[id1][j]
            demands_candidat += demands[j]
            assignmentclients_candidate[j] = candidate
            nb_client_candidat_copy[id1] += 1
            nb_client_candidat_copy[id] -= 1
        # If the CO2 emissions are the same, assign the client to the nearest candidate
        if cost_client_car_bike[id][j] == cost_client_car_bike[id1][j] and distance(clients[j],
                                                                                    candidates[id1]) < distance(
            clients[j], candidates[id]):  # dist_matrix[j][id1] < dist_matrix[j][id]:
            demands_candidat += demands[j]
            assignmentclients_candidate[j] = candidate
            nb_client_candidat_copy[id1] += 1
            nb_client_candidat_copy[id] -= 1

            # add the cost of freezing the demands of the clients assigned to the new candidat for one day
    saving_candidat += demands_candidat * 0.042

    # another idea wished_depots = 6 print("bebe",(len(clients) / wished_depots)) if min(elt for elt in
    # nb_client_candidat_copy if elt > 0) > (len(clients) / wished_depots) and saving_candidat < 0:
    if min(elt for elt in nb_client_candidat_copy if elt > 0) > 5 and saving_candidat < 0:
        # We assign clients to the new candidate, if we ensure savings from opening the candidate, and assign more
        # than 5 clients to this candidate.
        return saving_candidat, assignmentclients_candidate, nb_client_candidat_copy
    else:
        # We don't open the candidate and we have null saving
        return 0, assignments, nb_client_candidate


def greedy_heuristic_with_demand(candidates, clients, demands, cost_client_car_bike):
    nb_clients = len(clients)
    # Initially, all clients are assigned to the first candidate, which represents the main depot
    clients_assignments = [candidates[0]] * nb_clients
    total_cost = sum(cost_client_car_bike[0][j] for j in range(nb_clients))
    # List to store savings
    big_saving = []

    # List to track the state (open/closed) of candidates
    candidates_open = [0] * len(candidates)
    candidates_open[0] = 1  # The first candidate (central depot) is marked as open
    # Temporary list of candidates to consider
    candidates_temporary = candidates.copy()
    candidates_temporary.pop(0)  # Remove the main depot from the temporary list of candidats

    # Initiate a list of number of clients assigned to each candidate
    nb_client_candidate = [0] * len(candidates)
    # All clients are assigned initially to the depot
    nb_client_candidate[0] = len(clients)

    while candidates_temporary:
        saving_cost = [0] * len(candidates_temporary)

        # Initialize the matrix for client assignments and number of clients per candidate
        clients_assignment_matrix = [clients_assignments] * len(candidates_temporary)
        nb_client_candidate_matrix = [nb_client_candidate] * len(candidates_temporary)

        # Evaluate savings for each candidate in the temporary list, and update the matrix for client assignments and
        # number of clients per candidate
        for cand_temp in range(len(candidates_temporary)):
            saving_cost[cand_temp], clients_assignment_matrix[cand_temp], nb_client_candidate_matrix[
                cand_temp] = best_distribution(
                candidates_temporary[cand_temp], candidates, clients, demands, clients_assignment_matrix[cand_temp],
                nb_client_candidate_matrix[cand_temp], cost_client_car_bike)

            # Find the candidate with the best saving cost
        smallest_saving_cost = min(saving_cost)

        id_smallest_saving_cost = saving_cost.index(smallest_saving_cost)
        # If a negative saving cost is found, update assignments and open the selected candidate
        if smallest_saving_cost < 0:
            big_saving.append(smallest_saving_cost)
            # Select the candidate with the greatest savings
            candidate_elected = candidates_temporary[id_smallest_saving_cost]
            id_candidate_elected_candidates = index(candidate_elected, candidates)

            # Mark the selected candidate as open
            candidates_open[id_candidate_elected_candidates] = 1
            # Remove the selected candidate from the temporary list
            candidates_temporary.pop(id_smallest_saving_cost)

            # Update client assignments and number of clients per candidate
            clients_assignments = clients_assignment_matrix[id_smallest_saving_cost]
            nb_client_candidate = nb_client_candidate_matrix[id_smallest_saving_cost]
        else:
            break  # Stop if no saving is found

    # Calculate the total cost, accounting for all realized savings
    total_cost = sum(big_saving) + total_cost
    client_assignments_idx = [index(clients_assignments[elem], candidates) for elem in range(len(clients_assignments))]
    return total_cost, candidates_open, clients_assignments, client_assignments_idx


def plot_clients_refrigerator(clients, warehouses, assignments, save_path=None):
    """
    Trace les clients et entrepôts avec des flèches montrant les affectations.

    :param clients: Liste des coordonnées des clients [(x1, y1), (x2, y2), ...].
    :param warehouses: Liste des coordonnées des entrepôts [(xe1, ye1), (xe2, ye2), ...].
    :param assignments: Liste d'affectation des entrepôts pour chaque client [(xe1, ye1), ...].
    """
    plt.figure(figsize=(10, 8))

    # Trace des entrepôts
    plt.scatter(*(warehouses[0]), color='green', s=500, label='Depot')

    assignments = [tuple(assignment) for assignment in assignments]

    for warehouse in set(assignments):
        plt.scatter(*warehouse, color='blue', s=500)

    # Trace des clients et des flèches
    for i, client in enumerate(clients):
        # Différencier le premier client
        color = 'red'
        plt.scatter(*client, color=color, s=50, label='Client' if i == 0 else "")
        plt.arrow(assignments[i][0], assignments[i][1],
                  client[0] - assignments[i][0], client[1] - assignments[i][1],
                  head_width=0.3, head_length=0.3, fc='gray', ec='gray', length_includes_head=True)
    # plt.scatter(*(52.0, 24.0), color='yellow')
    # Légendes et personnalisation
    plt.title("Affectation des clients aux entrepôts")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_combined_routes_and_tours(depot: Tuple[float, float],
                                   clients: List[Tuple[float, float]],
                                   means_transport: List[List[int]],
                                   vertices: List[Tuple[float, float]],
                                   routes: List[int],
                                   save_path=None):
    """
    Combines plotting of routes (bike and car) and tour planning.
    :param depot: Coordinates of the depot (x, y).
    :param clients: List of client coordinates [(x1, y1), (x2, y2), ...].
    :param means_transport: List of lists indicating transport means for each hub (0 for bike, 1 for car).
    :param vertices: List of vertices (depot and hubs) coordinates [(xd, yd), ...].
    :param routes: List of hub indices in visiting order for the tour.
    :param save_path: Path to save the plot as an image. If None, shows the plot.
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
            if transport_modes == []:  # No clients assigned to the hub
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


if __name__ == "__main__":
    car_co2 = 772 / 1000  # g per 1km for 1t -> g per 1km for 1kg
    car_capacity = 1500
    bike_capacity = 100
    empty_car_weight = 15000  # in kg
    truck_co2 = 311 / 1000  # g per 1km for 1t -> g per km per kg
    empty_truck_weight = 30000  # 3t per truck
    ##Loop through instances "01" to "10"
    for i in range(1, 11):
        instance = ""
        if i < 10:
            instance += "0"
        instance += str(i)
        print(instance + ": ")
        dimension, capacity, indices, clients, demands = collect_infos_from_instance(
            instance)  # prepare instance# vertices without routes
        # Initialize variables
        candidats = clients  # All clients are candidates initially
        depot = clients[0]  # The depot is the first client
        clients = clients[1:]  # Remove the depot from the clients list
        demands = demands[1:]  # Remove the depot demand

        # Compute the distance matrix
        dist_matrix = distance_matrix(clients, candidats)

        # Compute costs for car and bike transportation
        cost_client_car_bike = get_costs_car_bike(clients, candidats, demands, capacity, dist_matrix)

        # Apply the greedy heuristic algorithm with demand
        total_cost, candidats_ouvert, clients_assignments, client_assignements_idx = greedy_heuristic_with_demand(
            candidats, clients, demands, cost_client_car_bike
        )

        print("customer_assignments:", clients_assignments)
        print("client_assignements_idx:", client_assignements_idx)
        print("candidats_ouvert:", candidats_ouvert)
        print("total cost:", total_cost)
        print("initial cost:", sum(cost_client_car_bike[0][j] for j in range(len(clients))))

        # Plot the clients and candidates for the current instance
        # plot_clients_refrigerateur(clients, candidats, clients_assignments)

        print(f"Completed instance {instance}\n")

    # resolution instance 11,1
    for i in range(11, 13):
        instance = str(i)
        dimension, capacity, indices, clients, demands = collect_infos_from_instance(
            instance)
        # Initialize variables
        candidats = clients  # All clients are candidates initially
        depot = clients[0]  # The depot is the first client
        clients = clients[1:]  # Remove the depot from the clients list
        demands = demands[1:]  # Remove the depot demand
        selected_points, groups = split_and_select(clients)
        total_cost = [0, 0, 0, 0]
        candidats_ouvert = [[], [], [], []]
        clients_assignments = [None, None, None, None]
        client_assignements_idx = [None, None, None, None]
        for i in range(1, 5):  # Parcourir les clés de 1 à 4
            clients = [depot] + groups[i]  # Ajouter 'depot' au début de la liste correspondante
            candidats = selected_points[i]
            dist_matrix = distance_matrix(clients, candidats)
            cost_client_car_bike = get_costs_car_bike(clients, candidats, demands, capacity, dist_matrix)
            total_cost[i - 1], candidats_ouvert[i - 1], clients_assignments[i - 1], client_assignements_idx[
                i - 1] = greedy_heuristic_with_demand(
                candidats, clients, demands, cost_client_car_bike)
        # print("customer_assignments_per_grp", clients_assignments)
        # print("client_assignements_idx_per_grp",client_assignements_idx)
        # print("candidats_ouvert_per_grp", candidats_ouvert)
        # print("total cost_per_grp",total_cost)
        print("total cost", sum(total_cost))
        print(f"Completed instance {instance}\n")
