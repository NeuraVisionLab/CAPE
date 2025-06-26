import numpy as np
import networkx as nx
import time
import copy

def pixel_graph(skeleton):

    _skeleton = copy.deepcopy(np.uint8(skeleton))
    _skeleton[0,:] = 0
    _skeleton[:,0] = 0
    _skeleton[-1,:] = 0
    _skeleton[:,-1] = 0
    G = nx.Graph()

    # add one node for each active pixel
    xs,ys = np.where(_skeleton>0)
    G.add_nodes_from([(int(x),int(y)) for i,(x,y) in enumerate(zip(xs,ys))])

    # add one edge between each adjacent active pixels
    for (x,y) in G.nodes():
        patch = _skeleton[x-1:x+2, y-1:y+2]
        patch[1,1] = 0
        for _x,_y in zip(*np.where(patch>0)):
            if not G.has_edge((x,y),(x+_x-1,y+_y-1)):
                G.add_edge((x,y),(x+_x-1,y+_y-1))

    for n,data in G.nodes(data=True):
        data['pos'] = np.array(n)[::-1]

    return G

def compute_angle_degree(c, p0, p1):
    p0c = np.sqrt((c[0] - p0[0]) ** 2 + (c[1] - p0[1]) ** 2)
    p1c = np.sqrt((c[0] - p1[0]) ** 2 + (c[1] - p1[1]) ** 2)
    p0p1 = np.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)

    # Prevent division by zero
    denominator = 2 * p1c * p0c
    if denominator == 0:
        return 0  # or some default value like 180

    cos_value = (p1c**2 + p0c**2 - p0p1**2) / denominator

    # Clip values to prevent NaN issues
    cos_value = np.clip(cos_value, -1.0, 1.0)

    return np.arccos(cos_value) * 180 / np.pi


def distance_point_line(c,p0,p1):
    return np.linalg.norm(np.cross(p0-c, c-p1))/np.linalg.norm(p1-p0)

def decimate_nodes_angle_distance(G, angle_range=(110,240), dist=0.3, verbose=True):

    H = copy.deepcopy(G)

    def f():
        start = time.time()
        nodes = list(H.nodes())
        np.random.shuffle(nodes)
        changed = False
        for n in nodes:

            ajacent_nodes = list(nx.neighbors(H, n))
            if n in ajacent_nodes:
                ajacent_nodes.remove(n)
            if len(ajacent_nodes)==2:
                angle = compute_angle_degree(n, *ajacent_nodes)
                d = distance_point_line(np.array(n), np.array(ajacent_nodes[0]), np.array(ajacent_nodes[1]))
                if d<dist or (angle>angle_range[0] and angle<angle_range[1]):
                    H.remove_node(n)
                    H.add_edge(*ajacent_nodes)
                    changed = True
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not f():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def remove_close_nodes(G, dist=10, verbose=True):

    H = copy.deepcopy(G)
    def _remove_close_nodes():
        edges = list(H.edges())
        changed = False
        for (s,t) in edges:
            if H.has_node(s) and H.has_node(t):
                d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
                if d<dist:
                    if len(H.edges(s))==2:
                        ajacent_nodes = list(nx.neighbors(H, s))
                        if s in ajacent_nodes:
                            ajacent_nodes.remove(s)
                        if t in ajacent_nodes:
                            ajacent_nodes.remove(t)
                        if len(ajacent_nodes)==1:
                            d = np.sqrt((s[0]-ajacent_nodes[0][0])**2+(s[1]-ajacent_nodes[0][1])**2)
                            if d<dist:
                                H.remove_node(s)
                                H.add_edge(ajacent_nodes[0], t)
                                changed = True
                    elif len(H.edges(t))==2:
                        ajacent_nodes = list(nx.neighbors(H, t))
                        if s in ajacent_nodes:
                            ajacent_nodes.remove(s)
                        if t in ajacent_nodes:
                            ajacent_nodes.remove(t)
                        if len(ajacent_nodes)==1:
                            d = np.sqrt((t[0]-ajacent_nodes[0][0])**2+(t[1]-ajacent_nodes[0][1])**2)
                            if d<dist:
                                H.remove_node(t)
                                H.add_edge(ajacent_nodes[0], s)
                                changed = True
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not _remove_close_nodes():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def remove_small_dangling(G, length=10, verbose=True):

    H = copy.deepcopy(G)
    edges = list(H.edges())
    for (s,t) in edges:
        d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
        if d<length:
            edge_count_s = len(H.edges(s))
            edge_count_t = len(H.edges(t))
            if edge_count_s==1:
                H.remove_node(s)
            if edge_count_t==1:
                H.remove_node(t)

    return H

def merge_close_intersections(G, dist=10, verbose=True):

    H = copy.deepcopy(G)
    def _merge_close_intersections():
        edges = list(H.edges())
        changed = False
        for (s,t) in edges:
            d = np.sqrt((s[0]-t[0])**2+(s[1]-t[1])**2)
            if d<dist:
                if len(H.edges(s))>2 and len(H.edges(t))>2:
                    ajacent_nodes = list(nx.neighbors(H, s))
                    if t in ajacent_nodes:
                        ajacent_nodes.remove(t)
                    H.remove_node(s)
                    for n in ajacent_nodes:
                        H.add_edge(n, t)
                    changed = True
                else:
                    pass
        return changed

    while True:
        if verbose:
            print("Remaining nodes:", len(H.nodes()))
        if not _merge_close_intersections():
            break

    if verbose:
        print("Finished. Remaining nodes:", len(H.nodes()))

    return H

def graph_from_skeleton(skeleton, angle_range=(135,225), dist_line=3,
                        dist_node=10, verbose=True, max_passes=20, relabel=True):
    """
    Parameters
    ----------
    skeleton : numpy.ndarray
        binary skeleton
    angle_range : (min,max) in degree
        two connected edges are merged into one if the angle between them
        is in this range
    dist_line : pixels
        two connected edges are merged into one if the distance between
        the central node to the line connecting the external nodes is
        lower then this value.
    dist_node : pixels
        two nodes that are connected by an edge are "merged" if their distance is
        lower than this value.
    """
    if verbose: print("Creation of densly connected graph.")
    G = pixel_graph(skeleton)

    for i in range(max_passes):

        if verbose: print("Pass {}:".format(i))

        n = len(G.nodes())

        if verbose: print("\tFirst decimation of nodes.")
        G = decimate_nodes_angle_distance(G, angle_range, dist_line, verbose)

        if verbose: print("\tFirst removing close nodes.")
        G = remove_close_nodes(G, dist_node, verbose)


        if verbose: print("\tRemoving short danglings.")
        G = remove_small_dangling(G, length=dist_node)

        if verbose: print("\tMerging close intersections.")
        G = merge_close_intersections(G, dist_node, verbose)

        if n==len(G.nodes()):
            break

    if relabel:
        mapping = dict(zip(G.nodes(), range(len(G.nodes()))))
        G = nx.relabel_nodes(G, mapping)

    return G
