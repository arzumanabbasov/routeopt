import folium
import numpy as np
import pandas as pd
from folium import plugins
import matplotlib.pyplot as plt
from sklearn import preprocessing
import osmnx as ox


def plot_map(dtf, y, x, start, zoom=12, tiles="cartodbpositron", popup=None, size=None, color=None,
             legend=False, lst_colors=None, marker=None):
    """
    Plot map with markers. This function is a wrapper of folium.Map(). It allows to easily plot a map with a set of markers
    that have a popup and a color. It also allows to add a legend and a marker. The size of the markers can be scaled based
    on a column of the dataframe. The color of the markers can be based on a column of the dataframe. The legend can be added
    only if the color is not None. The marker can be added only if the marker is not None. The size of the markers is scaled 
    based on the MinMaxScaler from sklearn. The color of the markers is randomly generated if lst_colors is None. The tiles 
    can be changed to "OpenStreetMap", "Stamen Terrain", "Stamen Water Color ", "Stamen Toner", "cartodbdark_matter".
    
    :param dtf: dataframe with the data
    :param y: column of dtf with the latitude
    :param x: column of dtf with the longitude
    :param start: list with the latitude and longitude of the starting point
    :param zoom: int with the initial zoom
    :param tiles: str with the tileset to use
    :param popup: column of dtf with the popup info
    :param size: column of dtf with the size of the markers
    :param color: column of dtf with the color of the markers
    :param legend: bool to add the legend
    :param lst_colors: list with the colors to use
    :param marker: column of dtf with the marker info
    
    
    :return: map
    """
    data = dtf.copy()

    if color is not None:
        lst_elements = sorted(list(dtf[color].unique()))
        lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(len(lst_elements))] \
            if lst_colors is None else lst_colors
        data["color"] = data[color].apply(lambda x: lst_colors[lst_elements.index(x)])

    if size is not None:
        scaler = preprocessing.MinMaxScaler(feature_range=(3, 15))
        data["size"] = scaler.fit_transform(data[size].values.reshape(-1, 1)).reshape(-1)

    # Initialize map
    map_ = folium.Map(location=start, tiles=tiles, zoom_start=zoom)

    if size is not None and color is None:
        data.apply(lambda row: folium.CircleMarker(
            location=[row[y], row[x]],
            popup=row[popup],
            color='#3186cc',
            fill=True,
            radius=row["size"]
        ).add_to(map_), axis=1)
    elif size is None and color is not None:
        data.apply(lambda row: folium.CircleMarker(
            location=[row[y], row[x]],
            popup=row[popup],
            color=row["color"],
            fill=True,
            radius=5
        ).add_to(map_), axis=1)
    elif size is not None and color is not None:
        data.apply(lambda row: folium.CircleMarker(
            location=[row[y], row[x]],
            popup=row[popup],
            color=row["color"],
            fill=True,
            radius=row["size"]
        ).add_to(map_), axis=1)
    else:
        data.apply(lambda row: folium.CircleMarker(
            location=[row[y], row[x]],
            popup=row[popup],
            color='#3186cc',
            fill=True,
            radius=5
        ).add_to(map_), axis=1)

    # Add tile layers with proper attribution
    layers = [
        {"name": "cartodbpositron", "attr": "&copy; <a href='https://carto.com/attributions'>CARTO</a>"},
        {"name": "openstreetmap", "attr": "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"},
        {"name": "Stamen Terrain", "attr": "&copy; <a href='https://stamen.com/attribution'>Stamen</a>"},
        {"name": "Stamen Water Color", "attr": "&copy; <a href='https://stamen.com/attribution'>Stamen</a>"},
        {"name": "Stamen Toner", "attr": "&copy; <a href='https://stamen.com/attribution'>Stamen</a>"},
        {"name": "cartodbdark_matter", "attr": "&copy; <a href='https://carto.com/attributions'>CARTO</a>"}
    ]
    for layer in layers:
        folium.TileLayer(layer["name"], attr=layer["attr"]).add_to(map_)
    folium.LayerControl(position='bottomright').add_to(map_)

    # Add legend
    if color is not None and legend:
        legend_html = f"""<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; 
                        z-index:9999; font-size:14px;">&nbsp;<b>{color}:</b><br>"""
        for i in lst_elements:
            legend_html += f"""&nbsp;<i class="fa fa-circle fa-1x" style="color:{lst_colors[lst_elements.index(i)]}"></i>
                            &nbsp;{i}<br>"""
        legend_html += """</div>"""
        map_.get_root().html.add_child(folium.Element(legend_html))

    # Add markers
    if marker is not None:
        lst_elements = sorted(list(dtf[marker].unique()))
        lst_colors = ["black", "red", "blue", "green", "pink", "orange", "gray"]  # 7 colors
        if len(lst_elements) > len(lst_colors):
            raise Exception(f"Marker has more unique values ({len(lst_elements)}) than available colors ({len(lst_colors)})")
        elif len(lst_elements) == 2:
            data[data[marker] == lst_elements[1]].apply(lambda row: folium.Marker(
                location=[row[y], row[x]],
                popup=row[marker],
                draggable=False,
                icon=folium.Icon(color=lst_colors[0])
            ).add_to(map_), axis=1)
        else:
            for i in lst_elements:
                data[data[marker] == i].apply(lambda row: folium.Marker(
                    location=[row[y], row[x]],
                    popup=row[marker],
                    draggable=False,
                    icon=folium.Icon(color=lst_colors[lst_elements.index(i)])
                ).add_to(map_), axis=1)

    # Add fullscreen button
    plugins.Fullscreen(
        position="topright",
        title="Expand",
        title_cancel="Exit",
        force_separate_button=True
    ).add_to(map_)

    return map_

def plot_starting_point(dtf, id_value, y_col='y', x_col='x'):
    """
    Plot the data points and highlight the starting point based on a specific ID value.

    Parameters:
    - dtf (pd.DataFrame): The DataFrame containing the data.
    - id_value (int): The ID value to use for identifying the starting point.
    - y_col (str): The name of the column containing y-coordinates (default is 'y').
    - x_col (str): The name of the column containing x-coordinates (default is 'x').
    """
    dtf["base"] = dtf["id"].apply(lambda x: 1 if x == id_value else 0)
    start = dtf[dtf["base"] == 1][[y_col, x_col]].values[0]

    print("start =", start)
    plt.scatter(dtf[x_col], dtf[y_col], color="black", label="Data Points")
    plt.scatter(start[1], start[0], color="red", label="Starting Point")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()

def plot_map_with_lines(dtf, start, y_col='y', x_col='x', zoom=11, tiles="cartodbpositron", 
                        popup_col='id', color_col='base', lst_colors=None):
    """
    Create a folium map with points from a DataFrame and draw lines from a starting point
    to each point in the DataFrame.

    Parameters:
    - dtf (pd.DataFrame): The DataFrame containing the data.
    - start (list or tuple): The starting point coordinates as [y, x].
    - y_col (str): The name of the column containing y-coordinates (default is 'y').
    - x_col (str): The name of the column containing x-coordinates (default is 'x').
    - zoom (int): The zoom level of the map (default is 11).
    - tiles (str): The tile layer for the map (default is "cartodbpositron").
    - popup_col (str): The column name for popup information (default is 'id').
    - color_col (str): The column name for color coding (default is 'base').
    - lst_colors (list): List of colors for the color coding (default is None).

    Returns:
    - folium.Map: The created folium map with points and lines.
    """
    map_ = folium.Map(location=start, zoom_start=zoom, tiles=tiles)

    if color_col in dtf.columns:
        lst_elements = sorted(list(dtf[color_col].unique()))
        if lst_colors is None:
            lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(len(lst_elements))]
        color_dict = dict(zip(lst_elements, lst_colors))
        dtf["color"] = dtf[color_col].apply(lambda x: color_dict.get(x, '#3186cc'))
    else:
        dtf["color"] = '#3186cc'
    
    for _, row in dtf.iterrows():
        folium.CircleMarker(
            location=[row[y_col], row[x_col]],
            popup=row[popup_col],
            color=row["color"],
            fill=True,
            radius=5
        ).add_to(map_)

    for _, row in dtf.iterrows():
        points = [start, [row[y_col], row[x_col]]]
        folium.PolyLine(points, color="red", weight=0.5, opacity=0.5).add_to(map_)

    folium.plugins.Fullscreen(
        position="topright",
        title="Expand",
        title_cancel="Exit",
        force_separate_button=True
    ).add_to(map_)

    folium.LayerControl(position='bottomright').add_to(map_)

    return map_

def plot_network_graph(G):
    """
    Plot the network graph.
    
    :param G: network graph
    """
    fig, ax = ox.plot_graph(G, bgcolor="black", node_size=5, node_color="white", figsize=(16, 8))
    plt.show()
    print([i for i in G.nodes()][:10])

def plot_shortest_path(G, path, color, line_width=5):
    """
    Plot the shortest path on the network graph.
    
    :param G: network graph
    :param path: shortest path as a list of nodes
    :param color: color of the path line
    :param line_width: width of the path line
    """
    fig, ax = ox.plot_graph_route(G, path, route_color=color, route_linewidth=line_width,
                                  node_size=1, bgcolor='black', node_color="white",
                                  figsize=(16, 8))
    plt.show()

def plot_tsp_route(dtf, route_idx, start, end):
    """
    Plot the TSP route on the map.
    
    :param dtf: DataFrame containing node locations
    :param route_idx: route indices
    """
    route_nodes = [dtf[["y", "x"]].iloc[i].tolist() for i in route_idx]
    map_ = plot_map(dtf, y="y", x="x", start=start, zoom=12, tiles="cartodbpositron", popup="id", marker="base", color="base")

    for i in range(len(route_nodes) - 1):
        points = [route_nodes[i], route_nodes[i + 1]]
        folium.PolyLine(points, color="blue", weight=5, opacity=0.8).add_to(map_)

    return map_