from utils.math import *
from utils.visualise import *
from scipy.spatial import distance_matrix
import pandas as pd
import numpy as np 


def load_data(dataset: str, city: str) -> pd.DataFrame:
    """
    Load dataset and filter by city.
    
    :param dataset: str with the dataset name
    :param city: str with the city name
    :return: dataframe
    """
    dtf = pd.read_csv(f'datasets/{dataset}.csv')
    dtf = dtf[dtf["REGIONS"] == city][["REGIONS", "Zone", "LATITUDE", "LONGITUDE"]].reset_index(drop=True)
    dtf = dtf.reset_index().rename(columns={"index": "id", "LATITUDE": "y", "LONGITUDE": "x"})
    return dtf


def main():
    dataset = 'GANJA'
    city = 'GANJA'
    id_value = 1

    # Load data
    dtf = load_data(dataset, city)

    # Plot starting point
    plot_starting_point(dtf, id_value)

    # Set starting point and end point
    start = dtf[dtf["id"] == id_value][["y", "x"]].values[0]
    end = dtf[dtf["id"] == 5][["y", "x"]].values[0]

    # Create network graph
    G = create_network_graph(start)
    plot_network_graph(G)

    # Calculate shortest path
    path_length, path_time = calculate_shortest_path(G, start, end)
    plot_shortest_path(G, path_length, color="red")
    plot_shortest_path(G, path_time, color="green")

    # Solve TSP with time limit
    lst_nodes = dtf[["y", "x"]].values.tolist()
    distance_matrix_ = pd.DataFrame(distance_matrix(lst_nodes, lst_nodes), index=dtf["id"].tolist(), columns=dtf["id"].tolist())
    
    # Generate time matrix using the time from the network graph
    time_matrix_ = pd.DataFrame(index=dtf["id"].tolist(), columns=dtf["id"].tolist())
    for i in range(len(lst_nodes)):
        for j in range(len(lst_nodes)):
            _, time_path = calculate_shortest_path(G, lst_nodes[i], lst_nodes[j])
            time_matrix_.iat[i, j] = nx.shortest_path_length(G, source=time_path[0], target=time_path[-1], weight='travel_time')
    
    route_idx, route_distance, total_time = solve_tsp_with_time_limit(distance_matrix_, time_matrix_)

    print("TSP Route with Time Constraint:", route_idx)
    print("Total Distance:", route_distance)
    print("Total Time:", total_time / 3600, "hours")

    # Plot TSP route with time constraint
    map_ = plot_tsp_route(dtf, route_idx, start, end)
    map_.save("maps/tsp_route_time_constrained.html")


if __name__ == "__main__":
    main()