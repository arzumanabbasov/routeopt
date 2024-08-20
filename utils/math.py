import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import logging

logger = logging.getLogger(__name__)

def create_network_graph(start_location, fallback_speed=50):
    """Create a network graph with speed attributes from OpenStreetMap data."""
    G = ox.graph_from_point(start_location, dist=10000, network_type='drive')
    
    # Add speed attributes to the edges, using fallback_speed if no speed information is available
    try:
        G = ox.add_edge_speeds(G, fallback=fallback_speed)
        G = ox.add_edge_travel_times(G)
    except ValueError as e:
        logger.error(f"Error adding edge speeds: {e}")
        raise
    
    return G

def calculate_shortest_path(G, start, end):
    """
    Calculate the shortest path between two points using Dijkstra's algorithm.
    
    :param G: network graph
    :param start: tuple with the starting point coordinates (latitude, longitude).
    :param end: tuple with the ending point coordinates (latitude, longitude).
    :return: shortest path as a list of nodes
    """
    start_node = ox.distance.nearest_nodes(G, start[1], start[0])
    end_node = ox.distance.nearest_nodes(G, end[1], end[0])
    
    path_length = nx.shortest_path(G, source=start_node, target=end_node, method='dijkstra', weight='length')
    path_time = nx.shortest_path(G, source=start_node, target=end_node, method='dijkstra', weight='travel_time')
    
    return path_length, path_time

def solve_tsp(matrix):
    """
    Solve the Traveling Salesman Problem using OR-Tools.
    
    :param matrix: distance matrix as a DataFrame
    :return: route indices and distance
    """
    def create_data_model(matrix):
        data = {}
        data['distance_matrix'] = matrix.values
        data['num_vehicles'] = 1
        data['depot'] = 0
        return data

    data = create_data_model(matrix)

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    model = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = model.RegisterTransitCallback(distance_callback)
    model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = model.SolveWithParameters(search_parameters)

    index = model.Start(0)
    route_idx, route_distance = [], 0
    while not model.IsEnd(index):
        route_idx.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(model.NextVar(index))
        route_distance += model.GetArcCostForVehicle(previous_index, index, 0)

    route_idx.append(manager.IndexToNode(index))  # Add the final node
    return route_idx, route_distance

def solve_tsp_with_time_limit(matrix, time_matrix, time_limit_hours=4):
    """
    Solve the Traveling Salesman Problem considering a time limit using OR-Tools.
    
    :param matrix: distance matrix as a DataFrame.
    :param time_matrix: travel time matrix as a DataFrame.
    :param time_limit_hours: the maximum allowed time in hours.
    :return: route indices, total distance, and total travel time.
    """
    def create_data_model(matrix, time_matrix):
        data = {}
        data['distance_matrix'] = matrix.values
        data['time_matrix'] = time_matrix.values
        data['num_vehicles'] = 1
        data['depot'] = 0  # Start from the first node
        return data

    data = create_data_model(matrix, time_matrix)

    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    model = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = model.RegisterTransitCallback(distance_callback)
    time_callback_index = model.RegisterTransitCallback(time_callback)

    model.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add time constraint
    time_limit = time_limit_hours * 60 * 60  # Convert hours to seconds
    model.AddDimension(
        time_callback_index,
        0,  # no slack
        time_limit,
        True,  # start cumul to zero
        'Time'
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = model.SolveWithParameters(search_parameters)

    if not solution:
        print("No solution found within the time limit.")
        return [], 0, 0

    index = model.Start(0)
    route_idx, route_distance, total_time = [], 0, 0

    # Iterate through the solution and stop if time exceeds the limit
    while not model.IsEnd(index):
        route_idx.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(model.NextVar(index))
        route_distance += model.GetArcCostForVehicle(previous_index, index, 0)
        travel_time = solution.Value(model.GetDimensionOrDie('Time').CumulVar(index))

        # Stop if the time exceeds the time limit
        if travel_time > time_limit:
            break

        total_time = travel_time  # Keep track of the total time

    return route_idx, route_distance, total_time / 3600  # Convert time back to hours