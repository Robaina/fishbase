import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


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


def simplify_prey_category(category: str) -> str:
    """
    Simplify prey categories into broader groups.

    Args:
        category (str): The original prey category.

    Returns:
        str: The simplified prey category.
    """
    category_mapping = {
        "Phytoplankton and Algae": [
            "phytoplankton",
            "benthic algae/weeds",
            "plants",
            "other plants",
        ],
        "Zooplankton and Crustaceans": [
            "zooplankton",
            "plank. crust.",
            "benth. crust.",
            "jellyfish/hydroids",
        ],
        "Mollusks": ["mollusks", "cephalopods", "gastropods", "bivalves"],
        "Worms": ["worms", "polychaetes", "non-annelids"],
        "Other Invertebrates": ["echinoderms", "cnidarians", "sponges/tunicates"],
        "Detritus": ["detritus", "debris", "carcasses"],
    }

    for simplified, originals in category_mapping.items():
        if category in originals:
            return simplified
    return "Other"


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


def create_trophic_web(df: pd.DataFrame) -> nx.DiGraph:
    """
    Create a trophic web graph from the given dataframe.

    Args:
        df (pd.DataFrame): Dataframe containing trophic information.

    Returns:
        nx.DiGraph: Directed graph representing the trophic web.
    """
    G = nx.DiGraph()

    species_trophic_levels = df.groupby("Species")["Troph"].mean().to_dict()
    for species, trophic_level in species_trophic_levels.items():
        G.add_node(species, trophic_level=trophic_level, node_type="fish")

    prey_categories = set()
    for col in ["FoodI", "FoodII", "FoodIII"]:
        prey_categories.update(map(simplify_prey_category, df[col].dropna().unique()))

    for category in prey_categories:
        if category != "Other":
            G.add_node(category, trophic_level=1.0, node_type="prey")

    for _, row in df.iterrows():
        predator = row["Species"]
        prey_items = [row["FoodI"], row["FoodII"], row["FoodIII"]]

        for prey in prey_items:
            if pd.notna(prey):
                simplified_prey = simplify_prey_category(prey)
                if simplified_prey in G.nodes():
                    G.add_edge(predator, simplified_prey)
                elif simplified_prey == "Other":
                    potential_prey = [
                        species
                        for species in G.nodes()
                        if G.nodes[species]["node_type"] == "fish"
                        and G.nodes[species]["trophic_level"]
                        < G.nodes[predator]["trophic_level"]
                    ]
                    for fish_prey in potential_prey:
                        G.add_edge(predator, fish_prey)

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
        species = row["B.species"]
        asv_ids = row["ID"]
        species_to_asv[species] = asv_ids

    graph_df["ID"] = (
        graph_df["Predator"].map(species_to_asv).fillna("")
        + ","
        + graph_df["Prey"].map(species_to_asv).fillna("")
    )
    graph_df["ID"] = graph_df["ID"].str.strip(",")
    graph_df.to_csv(output_file_path, sep="\t", index=False)


def visualize_trophic_web(G: nx.DiGraph) -> None:
    """
    Visualize the trophic web graph.

    Args:
        G (nx.DiGraph): Directed graph representing the trophic web.
    """
    plt.figure(figsize=(30, 20))

    pos = nx.spring_layout(G, k=0.9, iterations=50)

    fish_predators = {
        u
        for u, v in G.edges()
        if G.nodes[u]["node_type"] == "fish" and G.nodes[v]["node_type"] == "fish"
    }

    node_colors = [
        (
            "salmon"
            if G.nodes[node]["node_type"] == "fish" and node in fish_predators
            else "lightblue" if G.nodes[node]["node_type"] == "fish" else "lightgreen"
        )
        for node in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(
        G, pos, edge_color="gray", arrows=True, arrowsize=20, alpha=0.5, width=0.5
    )

    label_pos = {k: (v[0], v[1] + 0.02) for k, v in pos.items()}
    nx.draw_networkx_labels(
        G, label_pos, font_size=12, font_weight="bold", font_family="sans-serif"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()
