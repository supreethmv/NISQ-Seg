import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def image_to_grid_graph(gray_img, sigma=0.5):
    h, w = gray_img.shape
    nodes = np.zeros((h*w, 1))
    edges = []
    nx_elist = []
    min_weight, max_weight = 1, 0
    for i in range(h*w):
        x, y = i // w, i % w
        nodes[i] = gray_img[x, y]
        # Check neighbors (up and left)
        if x > 0:
            j = (x-1)*w + y
            weight = 1 - np.exp(-((gray_img[x, y] - gray_img[x-1, y])**2) / (2 * sigma**2))
            edges.append((i, j, weight))
            nx_elist.append(((x, y), (x-1, y), np.round(weight, 2)))
            min_weight, max_weight = min(min_weight, weight), max(max_weight, weight)
        if y > 0:
            j = x*w + y-1
            weight = 1 - np.exp(-((gray_img[x, y] - gray_img[x, y-1])**2) / (2 * sigma**2))
            edges.append((i, j, weight))
            nx_elist.append(((x, y), (x, y-1), np.round(weight, 2)))
            min_weight, max_weight = min(min_weight, weight), max(max_weight, weight)
    
    a, b = -1, 1
    if max_weight - min_weight:
        normalized_edges = [(node1, node2, -1 * np.round(((b - a) * ((weight - min_weight) / (max_weight - min_weight))) + a, 2)) for node1, node2, weight in edges]
    else:
        normalized_edges = [(node1, node2, -1 * np.round(weight, 2)) for node1, node2, weight in edges]
    return nodes, edges, normalized_edges

def generate_problem_instance(height, width):
    image = np.random.rand(height, width)
    _, _, normalized_elist = image_to_grid_graph(image)
    G = nx.grid_2d_graph(height, width)
    G.add_weighted_edges_from(normalized_elist)
    return G, image

def draw_graph(G, image):
  pixel_values = image
  plt.figure(figsize=(8,8))
  default_axes = plt.axes(frameon=True)
  pos = {(x,y):(y,-x) for x,y in G.nodes()}
  nx.draw_networkx(G,
                  pos=pos,
                  node_color=1-pixel_values,
                  with_labels=True,
                  node_size=3000,
                  cmap=plt.cm.Greys,
                  alpha=0.8,
                  ax=default_axes)
  nodes = nx.draw_networkx_nodes(G, pos, node_color=1-pixel_values,
                  node_size=3000,
                  cmap=plt.cm.Greys)
  nodes.set_edgecolor('k')
  edge_labels = nx.get_edge_attributes(G, "weight")
  nx.draw_networkx_edge_labels(G,
                              pos=pos,
                             edge_labels=edge_labels)