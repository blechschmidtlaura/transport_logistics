from data import collect_infos_from_instance
from star_scenario import get_costs_star_scenario
from tour_routing import nearest_neighbor, sweep_nearest
from utils import get_costs_car_bike,  prepare_clients_to_plot, calculate_distance_matrix
from UFL import greedy_heuristic_with_demand, plot_combined_routes_and_tours

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
        dimension, capacity, indices, vertices, demands = collect_infos_from_instance(
            instance)  # prepare instance# vertices without routes
        prepare_clients_to_plot(vertices)
        dist_matrix = calculate_distance_matrix(vertices)
        candidates = vertices  # clients + hub
        hub = vertices[0]
        clients = vertices[1:]
        demands = demands[1:]
        cost_client_car_bike = get_costs_car_bike(clients, candidates, demands, capacity, dist_matrix)
        total_cost, open_candidates, client_assign, client_assignments_idx = greedy_heuristic_with_demand(
            candidates, clients, demands, cost_client_car_bike)
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
                                                                            car_co2)
            cluster_transport_list.append(transport_list)
            costs_of_cluster.append(total_cost_of_cluster)
        # scenario 2_ routing through hubs
        coord_of_hubs = [candidates[idx] for idx in hub_ids]
        route, tour_costs = nearest_neighbor(coord_of_hubs[0], coord_of_hubs, demands_of_cluster, capacity, truck_co2,
                                             empty_truck_weight)
        route2, tour_costs2 = sweep_nearest(coord_of_hubs[0], coord_of_hubs, demands_of_cluster, capacity, truck_co2,
                                            empty_truck_weight)
        summed_costs = 0
        for cluster_costs in costs_of_cluster:
            summed_costs += cluster_costs
        summed_costs += tour_costs
        print(round(summed_costs, 3)
        plot_combined_routes_and_tours(hub_ids, clusters, cluster_transport_list, candidates, route)
