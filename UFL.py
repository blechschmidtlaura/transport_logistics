from data import collect_infos_from_instance
from star_scenario import get_costs_star_scenario
from tour_routing import nearest_neighbor, sweep_nearest
from utils import get_costs_car_bike, distance, prepare_clients_to_plot, calculate_distance_matrix
import matplotlib.pyplot as plt


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
            saving_cost[cand_temp], clients_assignment_matrix[cand_temp], nb_client_candidate_matrix[cand_temp] = best_distribution(
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


def routes(nb_clients, client_assignments_idx, cost_client_car_bike, demands):
    means_transport = [-1] * nb_clients
    nb = 0
    for i in range(0, nb_clients):
        if cost_client_car_bike[client_assignments_idx[i]][i] == 0:
            means_transport[i] = 0
        else:
            means_transport[i] = 1
            if demands[i] > 1500:
                nb += 1
    return means_transport, nb


def plot_clients_refrigerator(clients, warehouses, assignments):
    """
    Trace les clients et entrepôts avec des flèches montrant les affectations.

    :param clients: Liste des coordonnées des clients [(x1, y1), (x2, y2), ...].
    :param warehouses: Liste des coordonnées des entrepôts [(xe1, ye1), (xe2, ye2), ...].
    :param assignments: Liste d'affectation des entrepôts pour chaque client [(xe1, ye1), ...].
    """
    plt.figure(figsize=(10, 8))

    # Trace des entrepôts
    plt.scatter(*(warehouses[0]), color='green', s=500, label='Depot')
    set(assignments)

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
    plt.show()


if __name__ == '__main__':
    number_instances = 12
    car_co2 = 772 / 1000  # g per 1km for 1t -> g per 1km for 1kg
    car_capacity = 1500
    bike_capacity = 100
    empty_car_weight = 15000  # in kg
    truck_co2 = 311 / 1000  # g per 1km for 1t -> g per km per kg
    empty_truck_weight = 30000  # 3t per truck

    for i in range(1, number_instances + 1):
        instance = ""
        if i < 10:
            instance += "0"
        instance += str(i)
        print(instance + ": ")
        dimension, capacity, indices, clients, demands = collect_infos_from_instance(
            instance)  # prepare instance# vertices without routes
        prepare_clients_to_plot(clients)
        dist_matrix = calculate_distance_matrix(clients)
        candidates = clients
        clients = clients[1:]
        demands = demands[1:]
        cost_client_car_bike = get_costs_car_bike(clients, candidates, demands, capacity, dist_matrix)
        total_cost, open_candidates, client_assign, client_assignments_idx = greedy_heuristic_with_demand(
            candidates, clients, demands, cost_client_car_bike)
        # 1: [0,1,0,1,..], 1=hub
        # 2: [in cluster of hub idx, in cluster of hub idx, ...]

        # input1 hubs = [hub1, hub2, ...]
        # input2 [cluster for depot, cluster for node 1, cluster for node 2], [] if it is just a client and no hub

        # scenario 1: calculate demands of each hub: sum of all demands in cluster
        demands_of_cluster = []
        demands_of_cluster_as_list = [[]]
        costs_of_cluster = []
        clusters = [[]]
        hub_ids = [idx for idx, value in enumerate(open_candidates) if value == 1]
        for hub in hub_ids:
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
            # print("costcluster" + str(hub))
            # print(total_cost_of_cluster)
            costs_of_cluster.append(total_cost_of_cluster)
        # scenario 2_ routing through hubs
        hub_ids = [idx for idx, value in enumerate(open_candidates) if value == 1]
        coord_of_hubs = [clients[idx] for idx in hub_ids]
        route, tour_costs = nearest_neighbor(coord_of_hubs[0], coord_of_hubs, demands_of_cluster, capacity, truck_co2,
                                             empty_truck_weight)
        # print("costs2")
        # print(tour_costs)
        route2, tour_costs2 = sweep_nearest(coord_of_hubs[0], coord_of_hubs, demands_of_cluster, capacity, truck_co2,
                                            empty_truck_weight)
        # print("costs3")
        # print(tour_costs2)

        summed_costs = 0
        for cluster_costs in costs_of_cluster:
            summed_costs += cluster_costs
        summed_costs += tour_costs
        print(round(summed_costs, 3))

"""
if __name__ == "__main__":
    dimension, capacity, indices, clients, demands = collect_infos_from_instance("09.txt")
    candidates = clients
    depot = clients[0]
    clients = clients[1:]
    demands = demands[1:]
    dist_matrix = distance_matrix(clients, candidates)
    cost_client_car_bike = get_costs_car_bike(clients, candidates, demands, capacity, dist_matrix)
    total_cost, candidates_open, clients_assignments, client_assign_idx = greedy_heuristic_with_demand(candidates,
                                                                                                       clients,
                                                                                                       demands,
                                                                                                       cost_client_car_bike)
    print("customer_assignments", clients_assignments)
    print("client_assignements_idx", client_assign_idx)
    print("candidats_ouvert", candidates_open)
    print("total cost", total_cost)
    print("initial cost", sum(cost_client_car_bike[0][j] for j in range(len(clients))))
    # plot_clients_refrigerateur(clients, candidats, clients_assignments)
    nb_clients = len(clients)
    print("routes, nb ", routes(nb_clients, client_assign_idx, cost_client_car_bike, demands))
    # groups = split_and_select(clients)
    # total_cost = [0, 0, 0, 0]
    # candidats_ouvert = [[], [], [], []]
    # clients_assignments = [[], [], [], []]
    # client_assignements_idx = [[], [], [], []]
    #
    # for i in range(1, 5):  # Parcourir les clés de 1 à 4
    #    candidats = [depot] + groups[i]  # Ajouter 'depot' au début de la liste correspondante
    #    dist_matrix = distance_matrix(clients, candidats)
    #    cost_client_car_bike = get_costs_car_bike(clients, candidats, demands, capacity, dist_matrix)
    #    
    #    total_cost[i - 1], candidats_ouvert[i - 1], clients_assignments[i - 1], client_assignements_idx[i - 1] = greedy_heuristic_with_demand(
    #        candidats, clients, demands, cost_client_car_bike)
    # candidats = [sorted(set(sous_liste), key=sous_liste.index) for sous_liste in clients_assignments]

    # candidats= [depot] + groups[1]
    # dist_matrix = distance_matrix(clients, candidats)
    # cost_client_car_bike = get_costs_car_bike(clients, candidats, demands, capacity, dist_matrix)
    # total_cost, candidats_ouvert, clients_assignments, client_assignements_idx = greedy_heuristic_with_demand(candidats, clients, demands,cost_client_car_bike)
    # print("customer_assignments", clients_assignments)
    # print("client_assignements_idx",client_assignements_idx)
    # print("candidats_ouvert", candidats_ouvert)
    # print("total cost",total_cost)
"""
