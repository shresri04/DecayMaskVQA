import torch 
import networkx as nx

def build_text_graph(num_tokens):
    G = nx.Graph()
    for i in range(num_tokens):
        G.add_node(i)
        if i>0:
            G.add_edge(i, i-1)
    return G

def build_vision_graph(num_patches, grid_size):
    G = nx.Graph()
    for i in range(num_patches):
        G.add_node(i)
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r*grid_size + c
            if r+1 < grid_size:
                    G.add_edge(idx, (r+1) * grid_size + c)
            if c+1 < grid_size:
                G.add_edge(idx, r * grid_size + (c+1))
    return G

def shortest_path_distance(G, max_dist = 5):
    n = len(G.nodes)
    dist= torch.full((n, n), max_dist, dtype=torch.float)

    for i in G.nodes:
        lengths= nx.single_source_shortest_path_length(G, i, cutoff=max_dist)
        for j, d in lengths.items():
            dist[i, j] = min(d, max_dist)
    return dist