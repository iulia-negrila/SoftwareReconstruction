import os
from git import Repo
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import ast
from pydriller import Repository
from pydriller.domain.commit import ModificationType
from collections import defaultdict
import matplotlib as mpl
import community.community_louvain as community_louvain
from matplotlib import colors as mcolors
import py4cytoscape as p4c
import yaml
from graphviz import Digraph

# ---------- 1. Clone Repo If Not Exists Locally ----------

PROJECT_NAME = "zeeguu-api"
PROJECT_GITHUB = "https://github.com/zeeguu/api.git"
CODE_ROOT_FOLDER = os.path.join(os.getcwd(), PROJECT_NAME)

if not os.path.exists(CODE_ROOT_FOLDER):
    print("Cloning repository...")
    Repo.clone_from(PROJECT_GITHUB, CODE_ROOT_FOLDER)
else:
    print("Repository already exists.")
    print(CODE_ROOT_FOLDER)


# ---------- 2. Helper Function for Paths & Module Names ----------

def file_path(file_name):
    return os.path.join(CODE_ROOT_FOLDER, file_name)


def module_name_from_file_path(full_path):
    file_name = full_path[len(CODE_ROOT_FOLDER) + 1:]  # +1 to skip trailing slash
    file_name = file_name.replace(os.path.sep, ".")
    file_name = file_name.replace("__init__.py", "")
    file_name = file_name.replace(".py", "")
    return file_name


# ---------- 3. Import Parsing using AST (Abstract Syntax Tree) ----------

def imports_from_file_ast(file_path_str, internal_only=True, project_root="zeeguu"):
    try:
        with open(file_path_str, encoding='utf8') as f:
            tree = ast.parse(f.read(), filename=file_path_str)
    except Exception as e:
        print(f"Failed to parse {file_path_str}: {e}")
        return []

    imports = []

    # go through every node in AST (functions, classes, imports etc.)
    for node in ast.walk(tree):
        # handle lines like 'import flask, os, sys'
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        # handle lines like 'from flask import Flask'
        elif isinstance(node, ast.ImportFrom):
            if node.module: # this gives you 'flask' for example
                imports.append(node.module)

    # filter out all external libraries (like flask) and keep only the ones from zeeguu folder
    if internal_only:
        imports = [imp for imp in imports if imp.startswith(project_root)]

    return list(set(imports))  # remove duplicates


# ---------- 4. Dependency Graph Construction ----------

def dependencies_digraph(code_root_folder):
    files = Path(code_root_folder).rglob("*.py")
    G = nx.DiGraph()

    for file in files:
        # Skip files in 'test', 'tests', or 'tools' folders
        if any(part in {"test", "tests", "tools"} for part in file.parts):
            continue

        file_path_str = str(file)
        source_module = module_name_from_file_path(file_path_str)
        G.add_node(source_module)

        targets = imports_from_file_ast(file_path_str)
        print(f"{source_module} imports {targets}")
        for target in targets:
            if G.has_edge(source_module, target):
                G[source_module][target]['weight'] += 1
            else:
                G.add_edge(source_module, target, weight=1)
    return G


# ---------- 5. Graph Drawing ----------

MAIN_MODULES = ["zeeguu.core", "zeeguu.api", "zeeguu.cl", "zeeguu.config", "zeeguu.logging"] # ["zeeguu.core", "zeeguu.api"]

def is_main_module(module_name):
    return any(module_name.startswith(m) for m in MAIN_MODULES)

def filter_to_main_modules(G):
    H = nx.DiGraph()
    for source, target in G.edges:
        if is_main_module(source) and is_main_module(target):
            weight = G[source][target].get('weight', 1)
            H.add_edge(source, target, weight=weight)
    return H

# collapse to package (depth=3 -> zeeguu.core.model)
def collapse_to_package(module_name, depth=3):
    return '.'.join(module_name.split('.')[:depth])

def collapse_graph_by_module(G, depth=3):
    collapsed_G = nx.DiGraph()

    for source, target in G.edges:
        collapsed_src = collapse_to_package(source, depth)
        collapsed_tgt = collapse_to_package(target, depth)

        if collapsed_src == collapsed_tgt:
            continue  # skip self-dependencies

        # Get original edge weight, default to 1
        original_weight = G[source][target].get('weight', 1)

        # Add or accumulate weight in collapsed graph
        if collapsed_G.has_edge(collapsed_src, collapsed_tgt):
            collapsed_G[collapsed_src][collapsed_tgt]['weight'] += original_weight
        else:
            collapsed_G.add_edge(collapsed_src, collapsed_tgt, weight=original_weight)

    return collapsed_G

def draw_graph(G, size=(20, 20), with_labels=True, show_weights=True):
    plt.figure(figsize=size)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, node_size=500, node_color="skyblue", edge_color="gray", with_labels=with_labels, font_size=8)

    if show_weights:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Dependency Graph")
    plt.axis("off")
    plt.show()

def get_top_level_group(module_name):
    """Extract top-level group like api, core, cl, etc."""
    parts = module_name.split(".")
    if len(parts) >= 2 and parts[0] == "zeeguu":
        return parts[1]  # api, core, cl, etc.
    return "external"

# Colour nodes based on highest level modules (core, api, logging, cl, config)
def draw_colored_graph_top_level(G, size=(25, 25), with_labels=True, show_weights=True):
    groups = set(get_top_level_group(node) for node in G.nodes)

    # Assign consistent colors
    color_palette = plt.get_cmap("tab10")
    group_colors = {
        group: color_palette(i % 10)[:3]
        for i, group in enumerate(groups)
    }

    node_colors = [group_colors[get_top_level_group(node)] for node in G.nodes]

    # Plot
    plt.figure(figsize=size)
    pos = nx.spring_layout(G, k=1.8, iterations=100)
    nx.draw(G, pos, node_size=1000, node_color=node_colors, edge_color="gray", with_labels=with_labels, font_size=8)

    if show_weights:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)


    plt.title("Zeeguu Dependency Graph (Top-Level Modules)")
    plt.axis("off")
    plt.show()

def compute_pagerank(G):
    return nx.pagerank(G)

def draw_graph_with_pagerank(G, size=(25, 25), with_labels=True, top_k=10, show_weights=True):
    pagerank = compute_pagerank(G)
    max_rank = max(pagerank.values())

    # Normalize sizes for better visualization
    node_sizes = [300 + 3000 * (pagerank[node] / max_rank) for node in G.nodes]

    # Color nodes by top-level group
    groups = set(get_top_level_group(node) for node in G.nodes)
    color_palette = plt.get_cmap("tab10")
    group_colors = {
        group: color_palette(i % 10)[:3]
        for i, group in enumerate(groups)
    }
    node_colors = [group_colors[get_top_level_group(node)] for node in G.nodes]

    # Layout and draw
    plt.figure(figsize=size)
    pos = nx.spring_layout(G, k=1.8, iterations=100)
    nx.draw(G, pos, node_size=node_sizes, node_color=node_colors, edge_color="gray", with_labels=with_labels, font_size=6)

    if show_weights:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    # Print top-k important nodes
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print("\nTop {} modules by PageRank:".format(top_k))
    for rank, (node, score) in enumerate(top_nodes, 1):
        print(f"{rank}. {node} — score: {score:.4f}")

    plt.title("Zeeguu Dependency Graph (PageRank Highlighted)")
    plt.axis("off")
    plt.show()

def extract_subgraph_by_top_pagerank(G, top_k=10, hops=1):
    # Get top k nodes by pageRank
    pagerank = nx.pagerank(G)
    top_nodes = [node for node, _ in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    # Get nodes reachable within `hops` steps from top nodes
    nodes_to_include = set()
    for node in top_nodes:
        ego_net = nx.ego_graph(G, node, radius=hops, undirected=False)
        nodes_to_include.update(ego_net.nodes)

    subG = G.subgraph(nodes_to_include).copy()
    return subG


def draw_graph_with_communities(G, size=(25, 25), with_labels=True, show_weights=True):
    # Compute communities using Louvain method
    partition = community_louvain.best_partition(G.to_undirected())

    # Assign a color to each community
    communities = set(partition.values())
    color_palette = plt.get_cmap("tab20")
    community_colors = {
        comm: color_palette(i % 20) for i, comm in enumerate(communities)
    }
    node_colors = [community_colors[partition[node]] for node in G.nodes]

    # Layout and draw
    plt.figure(figsize=size)
    pos = nx.spring_layout(G, k=1.8, iterations=100)
    nx.draw(G, pos, node_size=800, node_color=node_colors, edge_color="gray", with_labels=with_labels, font_size=8)

    if show_weights:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title("Dependency Graph with Community Coloring (Louvain)")
    plt.axis("off")
    plt.show()

    # Print community assignment summary
    print("\nCommunity sizes:")
    from collections import Counter
    counts = Counter(partition.values())
    for comm, count in counts.items():
        print(f"Community {comm}: {count} modules")

# 5.1 Evolutionary Analysis - churn

def module_name_from_rel_path(rel_path):
    """Convert relative file path to module name."""
    if rel_path is None:
        return None
    file_name = rel_path.replace("/__init__.py", "").replace("\\__init__.py", "")
    file_name = file_name.replace("/", ".").replace("\\", ".")
    file_name = file_name.replace(".py", "")
    return file_name

def compute_churn(repo_dir):
    """
    Computes churn (number of commits touching each module) using PyDriller
    """

    churn_counts = defaultdict(int)

    print("Computing churn...")
    for commit in Repository(repo_dir).traverse_commits():
        for mod in commit.modified_files:
            try:
                new_path = mod.new_path
                old_path = mod.old_path

                if new_path is None and old_path is None:
                    continue

                if mod.change_type == ModificationType.RENAME:
                    churn_counts[module_name_from_rel_path(new_path)] = churn_counts.get(module_name_from_rel_path(old_path), 0) + 1
                    churn_counts.pop(module_name_from_rel_path(old_path), None)

                elif mod.change_type == ModificationType.DELETE:
                    churn_counts.pop(module_name_from_rel_path(old_path), None)

                elif mod.change_type == ModificationType.ADD:
                    churn_counts[module_name_from_rel_path(new_path)] = 1

                else: # modification to existing file
                    churn_counts[module_name_from_rel_path(old_path)] += 1
            except Exception as e:
                print(f"Something went wrong with: {mod}")
                pass

    print("\n--- Churn Computation Finished ---\n")
    return churn_counts

def draw_graph_with_churn(G, churn_map, size=(20, 20), with_labels=True):
    fig, ax = plt.subplots(figsize=size)  # CREATE AXES explicitly
    pos = nx.spring_layout(G, k=1.8, iterations=100)

    # Normalize churn values for coloring
    churn_values = [churn_map.get(node, 0) for node in G.nodes]
    max_churn = max(churn_values) if churn_values else 1  # prevent division by zero

    # Print churn value for each node
    print("\nChurn values per module:")
    for node in G.nodes:
        print(f"{node}: {churn_map.get(node, 0)}")

    # Create a color map
    cmap = plt.get_cmap("Reds")
    norm = mpl.colors.Normalize(vmin=0, vmax=max_churn)

    # Map normalized colors manually
    node_colors = [cmap(norm(churn_map.get(node, 0))) for node in G.nodes]

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=800,
        alpha=0.9,
        ax=ax  # <--- draw nodes on 'ax'
    )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", ax=ax)

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    # Add colorbar on the correct Axes
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Churn")

    plt.title("Dependency Graph with Churn Highlighted")
    plt.axis("off")
    plt.show()


def draw_combined_graph(G, churn_map, size=(25, 25), top_k_labels=10, show_weights=True):
    # Compute community (Louvain)
    partition = community_louvain.best_partition(G.to_undirected())
    communities = set(partition.values())

    # Assign colors to communities
    color_palette = plt.get_cmap("tab20")
    community_colors = {
        com: color_palette(i % 20)[:3]
        for i, com in enumerate(communities)
    }
    node_colors = [community_colors[partition[node]] for node in G.nodes]

    # Compute PageRank
    pagerank = nx.pagerank(G)
    max_rank = max(pagerank.values())
    node_sizes = [400 + 3000 * (pagerank[node] / max_rank) for node in G.nodes]

    # Compute churn-based border width
    max_churn = max(churn_map.values()) if churn_map else 1
    edge_widths = [1 + 4 * (churn_map.get(node, 0) / max_churn) for node in G.nodes]

    # Layout
    pos = nx.spring_layout(G, k=1.8, iterations=100)
    fig, ax = plt.subplots(figsize=size)

    # Draw nodes with edge color (borders) based on churn
    for i, node in enumerate(G.nodes):
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[node],
            node_color=[node_colors[i]],
            node_size=node_sizes[i],
            edgecolors='black',
            linewidths=edge_widths[i],
            ax=ax
        )

    # Draw edges
    # Scale edge width by weight
    weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_weight = max(weights) if weights else 1
    scaled_widths = [1 + 4 * (w / max_weight) for w in weights]

    nx.draw_networkx_edges(G, pos, width=scaled_widths, edge_color="gray", alpha=0.6, arrows=True, ax=ax)

    if show_weights:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k_labels]
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)

    plt.title("Combined Dependency View: Community (color), Importance/PageRank (size), Churn (border)")
    plt.axis("off")
    plt.show()

    # Print top nodes
    print("\nTop {} modules by PageRank:".format(top_k_labels))
    for rank, (node, score) in enumerate(top_nodes, 1):
        print(f"{rank}. {node} — score: {score:.4f}")

# 5.2 Visualization -> use Cytoscape

def export_graph_to_graphml(G, filename, pagerank=None, churn=None, community=None):
    # Add attributes to nodes
    for node in G.nodes:
        if pagerank:
            G.nodes[node]['pagerank'] = pagerank.get(node, 0)
        if churn:
            G.nodes[node]['churn'] = churn.get(node, 0)
        if community:
            G.nodes[node]['community'] = community.get(node, -1)

    # Export to GraphML
    nx.write_graphml(G, filename)
    print(f"Graph exported to {filename}")


# automation script - export to cytoscape
def visualize_in_cytoscape(graphml_file="zeeguu_dependency.graphml"):
    """
    Load the GraphML file into Cytoscape and apply visualizations using py4cytoscape.
    Visualization matches the combined_graph from NetworkX:
    - Node size based on PageRank
    - Node border width based on churn
    - Node color based on community
    - Edge width based on weight (number of imports)

    Args:
        graphml_file: Path to the GraphML file
    """

    # Connect to Cytoscape
    try:
        p4c.cytoscape_ping()
        print("Connected to Cytoscape.")
    except Exception as e:
        print("Error: Could not connect to Cytoscape. Is it running?")
        exit()

    # Delete existing network (if any)
    try:
        networks = p4c.get_network_list()
        if "Zeeguu Dependency Network" in networks:
            p4c.delete_network("Zeeguu Dependency Network")
    except:
        pass

    # Load GraphML and rename network
    net_suid = p4c.import_network_from_file(graphml_file)
    p4c.rename_network("Zeeguu Dependency Network", network=net_suid)

    # Extract unique community values safely
    try:
        community_df = p4c.get_table_columns("node", "community")
        community_raw = community_df["community"].tolist()
        print("Community raw: ", community_raw)
        community_values = sorted(set(str(c) for c in community_raw if str(c).isdigit()))
        print("Community values: ", community_values)
    except Exception as e:
        print("Failed to extract community values. Defaulting to 0 only.")
        community_values = [0]

    # Generate community → color mapping (discrete)
    tab20 = plt.get_cmap("tab20")
    color_mapping = {
        str(c): mcolors.to_hex(tab20(i % 20))
        for i, c in enumerate(community_values)
    }
    print("Color mapping : ", color_mapping)

    # Define visual mappings
    node_size = p4c.map_visual_property('NODE_SIZE', 'pagerank', 'c', [0.0, 0.2], [40, 150])
    border_width = p4c.map_visual_property('NODE_BORDER_WIDTH', 'churn', 'c', [0, 300], [1, 12])
    node_color = p4c.map_visual_property('NODE_FILL_COLOR', 'community', 'd',
                                         list(color_mapping.keys()), list(color_mapping.values()))
    edge_width = p4c.map_visual_property('EDGE_WIDTH', 'weight', 'c', [1, 5], [1, 8])
    node_labels = p4c.map_visual_property('NODE_LABEL', 'name', 'p')

    defaults = {
        'NODE_SHAPE': 'ellipse',
        'NODE_LABEL_POSITION': 'c,c,c,0.00,0.00',
        'NODE_LABEL_FONT_SIZE': 12,
        'NODE_BORDER_COLOR': '#000000',
        'NODE_BORDER_PAINT': '#000000',
        'EDGE_TRANSPARENCY': 120,
        'NODE_TRANSPARENCY': 220,
        'EDGE_TARGET_ARROW_SHAPE': 'ARROW',
        'EDGE_STROKE_UNSELECTED_PAINT': '#808080'
    }

    # Create visual style
    style_name = 'Zeeguu_Combined_Style'
    try:
        if style_name in p4c.get_visual_style_names():
            p4c.delete_visual_style(style_name)
    except:
        pass

    p4c.create_visual_style(style_name, defaults=defaults)
    p4c.update_style_mapping(style_name, node_size)
    p4c.update_style_mapping(style_name, border_width)
    p4c.update_style_mapping(style_name, node_color)
    p4c.update_style_mapping(style_name, edge_width)
    p4c.update_style_mapping(style_name, node_labels)

    p4c.set_visual_style(style_name)

    # Apply layout
    p4c.layout_network('force-directed')

    print("Cytoscape visualization complete.")


# Update the export function to ensure we have all necessary attributes
def export_graph_for_cytoscape(G, pagerank, churn_map, partition, filename="zeeguu_dependency.graphml"):
    """
    Export a NetworkX graph to GraphML for Cytoscape with attributes:
    - PageRank
    - Churn
    - Community
    - Name (for labels)
    - Edge weights
    """
    nx.set_node_attributes(G, pagerank, "pagerank")
    nx.set_node_attributes(G, churn_map, "churn")
    nx.set_node_attributes(G, partition, "community")
    nx.set_node_attributes(G, {n: n for n in G.nodes}, "name")

    for u, v, data in G.edges(data=True):
        data["weight"] = int(data.get("weight", 1))

    nx.write_graphml(G, filename)
    print(f"Graph exported to {filename}")
    export_graph_data(G)
    return filename

def export_graph_data(g, label="graph"):
    with open(f"{label}_nodes.txt", "w", encoding="utf8") as nf:
        nf.write(f"Total nodes: {len(g.nodes)}\n")
        nf.write("Name\tCommunity\tPageRank\tChurn\n")
        for node in sorted(g.nodes):
            name = g.nodes[node].get("name", node)
            community = g.nodes[node].get("community", 0)
            pagerank = g.nodes[node].get("pagerank", 0)
            churn = g.nodes[node].get("churn", "?")
            nf.write(f"{name}\t{community}\t{pagerank:.5f}\t{churn}\n")

    # Export edges
    with open(f"{label}_edges.txt", "w", encoding="utf8") as ef:
        ef.write(f"Total edges: {len(g.edges)}\n")
        ef.write("Source\tTarget\tWeight\n")
        for s, t in sorted(g.edges):
            weight = g.edges[s, t].get("weight", 1)
            ef.write(f"{s}\t{t}\t{weight}\n")

    print(f"Exported {len(g.nodes)} nodes and {len(g.edges)} edges to text.")

# 5.3 Deployment view
def generate_deployment_view():
    # === Load docker-compose.yml ===
    with open("zeeguu-api/docker-compose.yml", "r") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})
    networks = compose.get("networks", {})

    # === Create directed graph ===
    G = nx.DiGraph()

    # Add service nodes
    for service_name in services:
        G.add_node(service_name, type="service")

    # Add edges for depends_on relationships
    for service_name, service in services.items():
        depends_on = service.get("depends_on", [])
        for dependency in depends_on:
            G.add_edge(dependency, service_name)

    # Add network edges as infrastructure nodes
    for net_name in networks:
        net_node = f"network:{net_name}"
        G.add_node(net_node, type="network")
        for service_name, service in services.items():
            if net_name in service.get("networks", []):
                G.add_edge(net_node, service_name)

    # === Draw the graph ===
    pos = nx.spring_layout(G, k=1.2, iterations=100)

    # Node styling
    node_colors = [
        "#1f77b4" if not node.startswith("network:") else "#ffcc00"
        for node in G.nodes
    ]
    node_sizes = [1800 if not node.startswith("network:") else 1000 for node in G.nodes]

    plt.figure(figsize=(14, 10))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="gray",
        font_size=9,
        font_weight='bold'
    )
    plt.title("Zeeguu Deployment View (from docker-compose)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # === Export for Cytoscape ===
    nx.write_graphml(G, "zeeguu_deployment.graphml")
    print("Graph exported as 'zeeguu_deployment.graphml' for Cytoscape.")


# ---------- 6. Run Everything ----------

if __name__ == "__main__":
    DG = dependencies_digraph(CODE_ROOT_FOLDER)
    print(f"Graph has {len(DG.nodes)} nodes and {len(DG.edges)} edges.")
    # draw_graph(DG, (25, 25), with_labels=True)
    # draw_colored_graph_top_level(DG)
    collapsed_G = collapse_graph_by_module(DG, depth=3)
    focused_G = filter_to_main_modules(collapsed_G)
    print(f"Collapsed main modules graph: {len(collapsed_G.nodes)} nodes, {len(collapsed_G.edges)} edges")
    #draw_colored_graph_top_level(focused_G, (18, 18), with_labels=True, show_weights=True)
    #draw_graph_with_pagerank(focused_G, size=(20, 20), with_labels=True)

    # subgraph = extract_subgraph_by_top_pagerank(focused_G, top_k=10, hops=1)
    # print(f"Subgraph has {len(subgraph.nodes)} nodes and {len(subgraph.edges)} edges.")
    # draw_graph_with_pagerank(subgraph, size=(18, 18), with_labels=True)

    # churn_map = compute_churn(CODE_ROOT_FOLDER)
    # draw_graph_with_churn(focused_G, churn_map, size=(20, 20), with_labels=True)

    # draw_graph_with_communities(focused_G, size=(22, 22), with_labels=True)

    churn_map = compute_churn(CODE_ROOT_FOLDER)
    pagerank = compute_pagerank(focused_G)
    partition = community_louvain.best_partition(focused_G.to_undirected())

    # draw_combined_graph(focused_G, churn_map, size=(24, 24), top_k_labels=10)

    graphml_file = export_graph_for_cytoscape(focused_G, pagerank, churn_map, partition)
    try:
        visualize_in_cytoscape(graphml_file)
    except Exception as e:
        print(f"Error during Cytoscape visualization: {e}")

    generate_deployment_view()