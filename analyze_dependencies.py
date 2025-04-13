import os
import re
from git import Repo
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt

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


# ---------- 3. Import Parsing ----------

def import_from_line(line):
    try:
        y = re.search(r"^from (\S+)", line)
        if not y:
            y = re.search(r"^import (\S+)", line)
        return y.group(1) if y else None
    except:
        return None


def imports_from_file(file):
    try:
        with open(file, encoding='utf8') as f:
            lines = f.readlines()
    except:
        return []

    all_imports = []
    for line in lines:
        imp = import_from_line(line)
        if imp:
            all_imports.append(imp)
    return all_imports


# ---------- 4. Dependency Graph Construction ----------

def dependencies_digraph(code_root_folder):
    files = Path(code_root_folder).rglob("*.py")
    G = nx.DiGraph()

    for file in files:
        file_path_str = str(file)
        source_module = module_name_from_file_path(file_path_str)
        G.add_node(source_module)

        for target in imports_from_file(file_path_str):
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
