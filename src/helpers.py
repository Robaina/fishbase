import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import networkx as nx
import numpy as np
import json
import csv
from pathlib import Path
import pandas as pd
import seaborn as sns
from typing import List
import geopandas as gpd
import contextily as ctx

this_file_path = Path(__file__)


with open(f"{this_file_path.parent}/data/trophic_levels.json", "r") as f:
    food_categories = json.load(f)


def plot_family_distribution(df):
    family_counts = df["B.family"].value_counts()
    classes = df["B.class"].dropna().unique()

    class_color_map = {cls: color for cls, color in zip(classes, plt.cm.tab10.colors)}

    fig, ax = plt.subplots(figsize=(15, 6))

    for family in family_counts.index:
        family_class = df[df["B.family"] == family]["B.class"].iloc[0]
        color = class_color_map.get(family_class, "grey")
        ax.bar(family, family_counts[family], color=color)

    ax.set_title("Distribution at Family Level")
    ax.set_xlabel("Family")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=90, labelsize=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=class_color_map[cls]) for cls in classes
    ]
    ax.legend(
        handles, classes, title="Class", bbox_to_anchor=(1.05, 1), loc="upper left"
    )

    plt.tight_layout()
    plt.show()


def simplify_prey_category(category: str) -> tuple:
    """
    Simplify prey categories into broader groups and return the trophic level.

    Args:
        category (str): The original prey category.

    Returns:
        tuple: (simplified category, trophic level)
    """
    # Create a mapping of subcategories to main categories
    category_mapping = {}
    for main_category, data in food_categories.items():
        for subcategory in data["subcategories"]:
            category_mapping[subcategory.lower()] = (
                main_category,
                data["trophic_level"],
            )

    # Check if the category is in our mapping
    if category.lower() in category_mapping:
        return category_mapping[category.lower()]

    # If not found, return 'Other' with a default trophic level
    return ("Other", 2.5)


def save_graph(graph: nx.DiGraph, file_path: str) -> None:
    """
    Save the directed graph to a CSV or TSV file in a human-readable format.

    Args:
        graph (nx.DiGraph): Directed graph to save.
        file_path (str): Path to save the CSV or TSV file.
    """
    edges = list(graph.edges())
    edges_df = pd.DataFrame(edges, columns=["Predator", "Prey"])
    edges_df.to_csv(
        file_path, sep="\t" if file_path.endswith(".tsv") else ",", index=False
    )


def build_trophic_web(trophic_info_csv, station_id: str = None):
    # Read the CSV file
    df = pd.read_csv(trophic_info_csv)

    # Filter the dataframe if station_id is provided
    if station_id is not None:
        df = df[df["Stations"].str.contains(station_id, na=False)]

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for all finfish species and prey categories
    all_nodes = set(df["Species"].dropna().unique()) | set(
        df["Prey_category"].dropna().unique()
    ) - {"Fish"}
    for node in all_nodes:
        if node in df["Species"].values:
            node_type = "finfish"
            trophic_level = df[df["Species"] == node]["Troph"].mean()
        else:
            node_type = "prey_category"
            trophic_level = (
                df[df["Prey_category"] == node]["Prey_Troph"].dropna().iloc[0]
                if not df[df["Prey_category"] == node]["Prey_Troph"].dropna().empty
                else np.nan
            )

        if pd.notna(trophic_level):
            G.add_node(node, type=node_type, trophic_level=trophic_level)

    # Process each row in the dataframe
    for _, row in df.iterrows():
        predator = row["Species"]
        prey_category = row["Prey_category"]
        predator_troph = row["Troph"]
        prey_troph = row["Prey_Troph"]

        if pd.isna(predator) or pd.isna(predator_troph) or predator not in G.nodes():
            continue

        if pd.notna(prey_category):
            if prey_category == "Fish":
                # Link to all fish nodes with lower trophic level
                for fish_node in G.nodes():
                    if (
                        G.nodes[fish_node]["type"] == "finfish"
                        and G.nodes[fish_node]["trophic_level"] < predator_troph
                        and fish_node != predator
                    ):  # Prevent self-loops
                        # G.add_edge(predator, fish_node)
                        G.add_edge(fish_node, predator)
            elif prey_category in G.nodes():
                # Link to the prey category
                # G.add_edge(predator, prey_category)
                G.add_edge(prey_category, predator)

    # Ensure all fish nodes are connected
    for node in list(G.nodes()):
        if G.nodes[node]["type"] == "finfish" and G.out_degree(node) == 0:
            # Find the prey category with the highest trophic level lower than this fish
            possible_prey = [
                n
                for n in G.nodes()
                if G.nodes[n]["trophic_level"] < G.nodes[node]["trophic_level"]
                and n != node
            ]
            if possible_prey:
                best_prey = max(
                    possible_prey, key=lambda x: G.nodes[x]["trophic_level"]
                )
                # G.add_edge(node, best_prey)
                G.add_edge(best_prey, node)
            else:
                # If no suitable prey found, remove the isolated node
                G.remove_node(node)
    return G


def add_asv_ids_to_graph_data(
    graph_file_path: str, mapping_file_path: str, output_file_path: str
) -> None:
    """
    Adds a new column "ID" to the graph TSV file by mapping species to ASV IDs from the mapping file.

    Args:
    graph_file_path (str): Path to the graph TSV file.
    mapping_file_path (str): Path to the mapping TSV file.
    output_file_path (str): Path to save the updated TSV file.
    """
    graph_df = pd.read_csv(graph_file_path, sep="\t")
    mapping_df = pd.read_csv(mapping_file_path)
    species_to_asv = {}
    for _, row in mapping_df.iterrows():
        species = row["Species"]
        asv_ids = row["ID"]
        species_to_asv[species] = asv_ids

    graph_df["ID"] = (
        graph_df["Predator"].map(species_to_asv).fillna("")
        + ","
        + graph_df["Prey"].map(species_to_asv).fillna("")
    )
    graph_df["ID"] = graph_df["ID"].str.strip(",")
    graph_df.to_csv(output_file_path, sep="\t", index=False)


def visualize_trophic_web(
    G,
    figsize=(20, 12),
    node_size=5000,
    node_margin=33,
    title="Trophic Web",
    colorbar=True,
    figure_file=None,
    no_figure=False,
):
    # Create a new figure
    fig, ax = plt.subplots(figsize=figsize)

    def hierarchical_layout(
        G, subset_key="layer", width=1, height=1, offset_factor=0.05
    ):
        pos = {}
        layers = {}
        for node in G.nodes():
            layer = G.nodes[node][subset_key]
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)

        max_layer = max(layers.keys())
        for layer, nodes in layers.items():
            y = height * (1 - layer / max_layer)  # Invert y-coordinate
            nodes.sort()  # Sort nodes alphabetically for consistent ordering
            x_step = width / (len(nodes) + 1)
            for i, node in enumerate(nodes, start=1):
                x = i * x_step
                offset = offset_factor * (-1) ** i  # Zigzag pattern
                pos[node] = (x, y + offset)
        return pos

    # Define the layers for hierarchical layout
    for node in G.nodes():
        G.nodes[node]["layer"] = int(G.nodes[node].get("trophic_level", 0))

    pos = hierarchical_layout(G)

    # Get trophic levels
    trophic_levels = np.array(
        [G.nodes[node].get("trophic_level", 0) for node in G.nodes]
    )
    max_trophic_level = trophic_levels.max()
    min_trophic_level = trophic_levels.min()

    # Normalize trophic levels to range between 0 and 1
    norm = Normalize(vmin=min_trophic_level, vmax=max_trophic_level)

    # Use the viridis color map
    cmap = plt.cm.viridis
    node_colors = cmap(norm(trophic_levels))

    # Draw the nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9,
        margins=0.15,
        cmap=cmap,
        ax=ax,
    )

    # Draw the edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="lightgray",
        arrows=True,
        arrowsize=20,
        min_target_margin=node_margin,
        min_source_margin=node_margin,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    # Draw the labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_color="black",
        font_weight="bold",
        ax=ax,
    )

    # Set the title
    ax.set_title(title, fontsize=16)

    # Remove axis
    ax.invert_yaxis()
    ax.axis("off")

    # Add colorbar if requested
    if colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("Trophic Level", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Show the plot
    plt.tight_layout()
    if figure_file:
        plt.savefig(figure_file, dpi=300)
    if not no_figure:
        plt.show()
    plt.close()


def visualize_trophic_web_horizontal(
    G,
    figsize=(20, 12),
    node_size=5000,
    node_margin=33,
    title="Trophic Web",
    colorbar=True,
    figure_file=None,
    no_figure=False,
):
    # Create a new figure
    fig, ax = plt.subplots(figsize=figsize)

    def hierarchical_layout(G, subset_key="layer", offset_factor=0.02):
        pos = nx.multipartite_layout(G, subset_key=subset_key)

        # Extract layers and their positions
        layers = {}
        for node, (x, y) in pos.items():
            layer = G.nodes[node][subset_key]
            if layer not in layers:
                layers[layer] = []
            layers[layer].append((node, x, y))

        # Apply zig-zag pattern
        for layer, nodes in layers.items():
            nodes.sort(key=lambda x: x[2])  # Sort by y-coordinate
            zigzag = [(-1) ** i for i in range(len(nodes))]
            for i, (node, x, y) in enumerate(nodes):
                pos[node] = (x + zigzag[i] * offset_factor, y)

        return pos

    # Define the layers for hierarchical layout
    for node in G.nodes():
        G.nodes[node]["layer"] = int(G.nodes[node].get("trophic_level", 0))

    pos = hierarchical_layout(G)

    # Get trophic levels
    trophic_levels = np.array(
        [G.nodes[node].get("trophic_level", 0) for node in G.nodes]
    )
    max_trophic_level = trophic_levels.max()
    min_trophic_level = trophic_levels.min()

    # Normalize trophic levels to range between 0 and 1
    norm = Normalize(vmin=min_trophic_level, vmax=max_trophic_level)

    # Use the viridis color map
    cmap = plt.cm.viridis
    node_colors = cmap(norm(trophic_levels))

    # Draw the nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_size,
        alpha=0.9,
        margins=0.15,
        cmap=cmap,
        ax=ax,
    )

    # Draw the edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="lightgray",
        arrows=True,
        arrowsize=20,
        min_target_margin=node_margin,
        min_source_margin=node_margin,
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    # Draw the labels
    nx.draw_networkx_labels(
        G, pos, font_size=10, font_color="black", font_weight="bold", ax=ax
    )

    # Set the title
    ax.set_title(title, fontsize=16)

    # Remove axis
    ax.axis("off")

    # Add colorbar if requested
    if colorbar:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
        cbar = plt.colorbar(sm, cax=cax, orientation="vertical")
        cbar.set_label("Trophic Level", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Show the plot
    plt.tight_layout()
    if figure_file:
        plt.savefig(figure_file, dpi=300)
    if not no_figure:
        plt.show()
    plt.close()


def process_station_data(station_file, station_ids):
    # Read the station data
    df = pd.read_csv(station_file, index_col=0)

    # Create a dictionary to store ASV to station mappings
    asv_stations = {}

    for index, row in df.iterrows():
        asv_id = row["ID"]
        present_stations = []
        for station in station_ids:
            station_cols = [col for col in df.columns if col.startswith(f"{station}.")]
            if np.all(row[station_cols] > 0):
                present_stations.append(station)

        if not present_stations:
            present_stations = ["not_found"]

        asv_stations[asv_id] = present_stations

    return asv_stations


def update_trophic_data(trophic_file, asv_stations, output_file):
    # Read the trophic data
    df = pd.read_csv(trophic_file)

    def get_stations(asv_list):
        if pd.isna(asv_list):
            return "not_found"
        stations = set()
        for asv in str(asv_list).split(","):
            stations.update(asv_stations.get(asv.strip(), ["not_found"]))
        return ",".join(sorted(stations))

    # Add new column with stations
    df["Stations"] = df["ID"].apply(get_stations)

    # Save updated dataframe
    df.to_csv(output_file, index=False)


def add_station_to_trophic_info(station_file, trophic_file, output_file, station_ids):
    asv_stations = process_station_data(station_file, station_ids)
    update_trophic_data(trophic_file, asv_stations, output_file)
    print(f"Updated trophic data saved to {output_file}")


def plot_trophic_level_distribution(
    df: pd.DataFrame,
    station_ids: List[str] = None,
    output_file: str = None,
    figure_file: str = None,
    no_figure: bool = False,
    title: str = "Distribution of Trophic Levels",
):
    """
    Plot the distribution of trophic levels.

    Args:
        df (pd.DataFrame): Dataframe containing trophic information.
        station_ids (List[str], optional): List of station IDs to filter species by. Default is None.
        output_file (str, optional): Path to save the plot. If None, the plot will be displayed instead.

    Returns:
        None
    """
    # Remove rows with NaN trophic levels
    df = df.dropna(subset=["Troph"])

    if station_ids is None or len(station_ids) == 0:
        # Plot for all stations
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x="Troph", kde=True, ax=ax)
        ax.set_xlabel("Trophic Level")
        ax.set_ylabel("Counts")
        ax.set_title(title)
    else:
        # Determine the number of rows and columns for subplots
        n = len(station_ids)
        cols = min(3, n)
        rows = (n - 1) // cols + 1

        fig, axes = plt.subplots(
            rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False
        )
        axes = axes.flatten()

        colors = plt.cm.rainbow(np.linspace(0, 1, n))

        for i, station_id in enumerate(station_ids):
            station_df = df[df["Stations"].str.contains(station_id, na=False)]

            sns.histplot(
                data=station_df, x="Troph", kde=True, ax=axes[i], color=colors[i]
            )
            axes[i].set_xlabel("Trophic Level")
            axes[i].set_ylabel("Counts")
            axes[i].set_title(title)

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()

    # Save or display the plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    if figure_file:
        plt.savefig(figure_file)
    if not no_figure:
        plt.show()
    plt.close()


def add_prey_categories_to_trophic_info(csv_path, json_path, output_path):
    # Load the JSON data
    with open(json_path, "r") as json_file:
        food_categories = json.load(json_file)

    # Create a lookup dictionary for subcategories
    subcategory_lookup = {}
    for category, data in food_categories.items():
        for subcategory in data["subcategories"]:
            subcategory_lookup[subcategory.lower()] = {
                "category": category,
                "trophic_level": data["trophic_level"],
            }

    # Read the CSV and write the new CSV simultaneously
    with open(csv_path, "r", newline="") as input_file, open(
        output_path, "w", newline=""
    ) as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        # Read the header
        header = next(reader)

        # Find the index of 'Food III' column
        food_iii_index = header.index("FoodIII")

        # Insert new columns after 'Food III'
        new_header = (
            header[: food_iii_index + 1]
            + ["Prey_category", "Prey_Troph"]
            + header[food_iii_index + 1 :]
        )
        writer.writerow(new_header)

        # Process each row
        for row in reader:
            new_row = row[: food_iii_index + 1]

            # Check only Food III column
            food = row[food_iii_index].lower()
            if food in subcategory_lookup:
                new_row.append(subcategory_lookup[food]["category"])
                new_row.append(str(subcategory_lookup[food]["trophic_level"]))
            else:
                # If no match found
                new_row.extend(["", ""])

            new_row.extend(row[food_iii_index + 1 :])
            writer.writerow(new_row)

    print(f"Updated CSV saved to {output_path}")


def plot_galapagos_stations(stations_csv, output_figure_path=None, figsize=(10, 10)):
    """
    Plots the sampling stations on a map of the Galápagos Islands.

    Parameters:
    - stations_csv (str): Path to the CSV file containing the sampling stations.
    - output_figure_path (str or None): Path to save the output figure. If None, the figure is only displayed.
    """
    df = pd.read_csv(stations_csv)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon2, df.lat2))
    gdf.set_crs(epsg=4326, inplace=True)

    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(ax=ax, color="blue", markersize=50)
    ctx.add_basemap(ax, crs=gdf.crs.to_string())

    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf["SiteID"]):
        ax.text(x, y, label, fontsize=8, ha="right")

    ax.set_title("Sampling Stations in the Galápagos Islands")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    if output_figure_path:
        plt.savefig(output_figure_path)
    else:
        plt.show()
