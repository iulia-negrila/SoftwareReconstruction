import os
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
            G.add_edge(source_module, target)
    return G


# ---------- 5. Graph Drawing ----------

def draw_graph(G, size=(20, 20), with_labels=True):
    plt.figure(figsize=size)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, node_size=500, node_color="skyblue", edge_color="gray", with_labels=with_labels, font_size=8)
    plt.title("Dependency Graph")
    plt.axis("off")
    plt.show()


# ---------- 6. Run Everything ----------

if __name__ == "__main__":
    DG = dependencies_digraph(CODE_ROOT_FOLDER)
    print(f"Graph has {len(DG.nodes)} nodes and {len(DG.edges)} edges.")
    draw_graph(DG, (25, 25), with_labels=True)
