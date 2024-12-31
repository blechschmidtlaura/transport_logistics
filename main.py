import numpy as np
import matplotlib.pyplot as plt
import sys
import math

from data import collect_infos_from_instance
from star_scenario import get_costs_star_scenario
from utils import calculate_distance_matrix, prepare_clients_to_plot, get_hub_id
from UFL import greedy_heuristic_with_demand

