import os
import random
import re
from git import Repo
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import ast

# ---------- 1. Clone Repo If Not Exists Locally ----------

PROJECT_NAME = "zeeguu-api"
PROJECT_GITHUB = "https://github.com/zeeguu/api.git"
CODE_ROOT_FOLDER = os.path.join(os.getcwd(), PROJECT_NAME)

if not os.path.exists(CODE_ROOT_FOLDER):
    print("Cloning repository...")
    Repo.clone_from(PROJECT_GITHUB, CODE_ROOT_FOLDER)
else:
    print("Repository already exists.")


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

MAIN_MODULES = ["zeeguu.core", "zeeguu.api", "zeeguu.cl", "zeeguu.config", "zeeguu.logging"]

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

def draw_graph(G, size=(20, 20), with_labels=True):
    plt.figure(figsize=size)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, node_size=500, node_color="skyblue", edge_color="gray", with_labels=with_labels, font_size=8)
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

def draw_graph_with_pagerank(G, size=(25, 25), with_labels=True, top_k=10):
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

    # Print top-k important nodes
    top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print("\nTop {} modules by PageRank:".format(top_k))
    for rank, (node, score) in enumerate(top_nodes, 1):
        print(f"{rank}. {node} — score: {score:.4f}")

    plt.title("Zeeguu Dependency Graph (PageRank Highlighted)")
    plt.axis("off")
    plt.show()

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
    draw_graph_with_pagerank(focused_G, size=(20, 20), with_labels=True)