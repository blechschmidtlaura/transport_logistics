import random
import numpy as np
import matplotlib.pyplot as plt


def prepare_clients_to_plot(clients):
    clients_matrix = np.array(clients)
    # draw al vertices as red points
    plt.plot(clients_matrix[:, 0], clients_matrix[:, 1], ".", color="red", markersize=10, alpha=0.5)
    # depot (first point) is blue
    plt.plot(clients_matrix[:, 0][0], clients_matrix[:, 1][0], ".", color="blue", markeredgecolor='blue', markersize=10,
             alpha=1.0)
    for i, (x, y) in enumerate(clients_matrix):
        plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom', color="black")
    plt.axis('on')
    plt.gca().set_aspect('equal')


def plot_instance(instance, clients):
    prepare_clients_to_plot(clients)  # prepare vertices for figure
    nombre = instance[0:-4] + ".png"
    plt.savefig(nombre, format='png', dpi=300, transparent=True, bbox_inches='tight',
                pad_inches=0)  # save figure in png
    plt.show(block=True)


def calculate_distance_matrix(coordinates):  # distances in hectometers
    coords = np.array(coordinates)
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2)) * 0.1  # mapping from hectometers to km
    return dist_matrix


def get_hub_id(hub_coord, vertices):
    for id in range(0, len(vertices)):
        if hub_coord == vertices[id]:
            hub_id = id
            return hub_id


def distance(point1, point2):
    dx = (point1[0] - point2[0]) * 0.1
    dy = (point1[1] - point2[1]) * 0.1
    return np.sqrt(dx ** 2 + dy ** 2)


def split_and_select(points):
    """
    Divise les points en 4 groupes selon leurs coordonnées et sélectionne 10 points aléatoires par groupe.

    Args:
        points (list of tuples): Une liste de tuples (x, y) représentant les coordonnées des points.

    Returns:
        dict: Un dictionnaire contenant 10 points sélectionnés aléatoirement pour chaque groupe.
    """
    if not points:
        return {}

    # Initialiser les bornes avec des valeurs extrêmes
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # Trouver les bornes min et max pour x et y en une seule passe
    for x, y in points:
        if x < min_x: min_x = x
        if x > max_x: max_x = x
        if y < min_y: min_y = y
        if y > max_y: max_y = y

    # Calculer les seuils pour les groupes
    mid_x = min_x + (max_x - min_x) / 2
    mid_y = min_y + (max_y - min_y) / 2

    # Initialiser les groupes
    groups = {1: [], 2: [], 3: [], 4: []}

    # Séparer les points dans les 4 groupes
    for x, y in points:
        if x <= mid_x and y <= mid_y:
            groups[1].append((x, y))
        elif x > mid_x and y <= mid_y:
            groups[2].append((x, y))
        elif x <= mid_x and y > mid_y:
            groups[3].append((x, y))
        else:
            groups[4].append((x, y))
    random.seed(44)
    # Sélectionner aléatoirement 10 points dans chaque groupe
    selected_points = {
        group_id: random.sample(group_points, min(len(group_points), 50))
        for group_id, group_points in groups.items()
    }

    return selected_points, groups


def distance_matrix(clients, candidats):
    # Initialisation
    distances = np.zeros((len(candidats), len(clients)))
    for i in range(len(candidats)):
        for j in range(len(clients)):
            distances[i][j] = distance(clients[j], candidats[i])
    return distances


def get_costs_car_bike(clients, candidats, demands, capacity, dist_matrix):
    co2_cost = np.zeros((len(candidats), len(clients)))
    co2_car_emission = 0.772

    for i in range(len(candidats)):
        for j in range(len(clients)):
            # condition for bike, <= 500kg, <= 6
            if demands[j] <= 500 and dist_matrix[i][j] * 2 <= 6:  # take bike
                co2_cost[i][j] = 0
                # bike_routes.append(route)
                continue
            else:  # take the car
                total_load = demands[j]
                while total_load > 0:
                    load = min(demands[j],
                               capacity)  # if the truck can not carry all demands -> multiple routes to the same client
                    co2_cost[i][j] += co2_car_emission * load * dist_matrix[i][j] + 0.2 * 2 * dist_matrix[i][j]
                    total_load -= load  # reeduce the remaining demand
    return co2_cost
