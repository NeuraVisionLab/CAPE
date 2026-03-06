"""
Graph and image helpers for CREMI 2D dataset: load/save graphs, crop, rotate, flip.
"""
from __future__ import annotations

import os
from typing import Tuple

import networkx as nx
import numpy as np
import torch
import torchvision.transforms.functional as TF
from shapely.geometry import Point, LineString, box


def load_graph_txt(filename: str) -> nx.Graph:
    G = nx.Graph()
    nodes, edges = [], []
    with open(filename, "r") as f:
        switch = True
        i = 0
        for line in f:
            line = line.strip()
            if len(line) == 0 and switch:
                switch = False
                continue
            if switch:
                x, y = line.split()
                G.add_node(i, pos=(float(x), float(y)))
                i += 1
            else:
                a, b = line.split()
                G.add_edge(int(a), int(b))
    return G


def graph_to_txt(G: nx.Graph) -> str:
    nodes = list(G.nodes())
    lines = []
    for n in nodes:
        lines.append("{:.6f} {:.6f}".format(G.nodes[n]["pos"][0], G.nodes[n]["pos"][1]))
    lines.append("")
    for s, t in G.edges():
        lines.append("{} {}".format(nodes.index(s), nodes.index(t)))
    return "\r\n".join(lines) + "\r\n"


def txt_to_graph(content: str) -> nx.Graph:
    G = nx.Graph()
    lines = content.strip().splitlines()
    switch = True
    node_index = 0
    for line in lines:
        line = line.strip()
        if len(line) == 0 and switch:
            switch = False
            continue
        if switch:
            x, y = line.split()
            G.add_node(node_index, pos=(float(x), float(y)))
            node_index += 1
        else:
            a, b = line.split()
            G.add_edge(int(a), int(b))
    return G


def update_graph_positions_for_padding(graph: nx.Graph, pad_left: int, pad_top: int) -> nx.Graph:
    for node in graph.nodes:
        if "pos" in graph.nodes[node]:
            x, y = graph.nodes[node]["pos"]
            graph.nodes[node]["pos"] = (x + pad_left, y + pad_top)
    return graph


def crop_graph(
    graph: nx.Graph,
    xmin: float, ymin: float, xmax: float, ymax: float,
    precision: int = 8,
) -> nx.Graph:
    bounding_box = box(xmin, ymin, xmax, ymax)
    cropped = nx.Graph()
    node_positions = {}
    inside_nodes = {}
    coord_to_node = {}
    for n, data in graph.nodes(data=True):
        pos = data["pos"]
        node_positions[n] = pos
        x, y = pos
        if xmin <= x <= xmax and ymin <= y <= ymax:
            new_pos = (x - xmin, y - ymin)
            inside_nodes[n] = new_pos
            cropped.add_node(n, pos=new_pos)
            key = (round(new_pos[0], precision), round(new_pos[1], precision))
            coord_to_node[key] = n
    max_node_index = max(graph.nodes()) if graph.nodes() else 0
    for u, v, data in graph.edges(data=True):
        u_pos, v_pos = node_positions[u], node_positions[v]
        line = LineString([u_pos, v_pos])
        if u in inside_nodes and v in inside_nodes:
            if u != v:
                cropped.add_edge(u, v, **data)
            continue
        if not bounding_box.intersects(line):
            continue
        intersection = bounding_box.intersection(line)
        if intersection.is_empty:
            continue
        if intersection.geom_type == "Point":
            pts = [(intersection.x, intersection.y)]
        elif intersection.geom_type == "MultiPoint":
            pts = [(pt.x, pt.y) for pt in intersection.geoms]
        elif intersection.geom_type == "LineString":
            pts = list(intersection.coords)
        else:
            continue
        pts.sort(key=lambda pt: line.project(Point(pt)))
        new_nodes = []
        for pt in pts:
            new_pos = (pt[0] - xmin, pt[1] - ymin)
            key = (round(new_pos[0], precision), round(new_pos[1], precision))
            if key in coord_to_node:
                node_id = coord_to_node[key]
            else:
                max_node_index += 1
                node_id = max_node_index
                cropped.add_node(node_id, pos=new_pos)
                coord_to_node[key] = node_id
            new_nodes.append(node_id)
        endpoints = ([u] if u in inside_nodes else []) + new_nodes + ([v] if v in inside_nodes else [])
        for i in range(len(endpoints) - 1):
            if endpoints[i] != endpoints[i + 1]:
                cropped.add_edge(endpoints[i], endpoints[i + 1], **data)
    return cropped


def rotate_image(image: torch.Tensor, angle: float) -> torch.Tensor:
    return TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)


def rotate_label(label: torch.Tensor, angle: float) -> torch.Tensor:
    if label.dim() == 3:
        label = label.squeeze(0)
    rotated = TF.rotate(label.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST)
    return (rotated > 0.5).to(torch.float32).squeeze(0)


def rotate_graph(graph: nx.Graph, angle_degrees: float, center: Tuple[float, float]) -> nx.Graph:
    angle_rad = np.radians(-angle_degrees)
    cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    out = nx.Graph()
    for node, (x, y) in graph.nodes(data="pos"):
        orig = np.array([x, y])
        translated = orig - np.array(center)
        rotated = rot @ translated + np.array(center)
        out.add_node(node, pos=tuple(rotated))
    out.add_edges_from(graph.edges())
    return out


def flip_image(image: torch.Tensor, direction: str = "horizontal") -> torch.Tensor:
    if direction == "horizontal":
        return torch.flip(image, dims=[2])
    if direction == "vertical":
        return torch.flip(image, dims=[1])
    raise ValueError("direction must be 'horizontal' or 'vertical'")


def flip_skeleton(skeleton: torch.Tensor | np.ndarray, direction: str) -> torch.Tensor:
    if hasattr(skeleton, "cpu"):
        skeleton = skeleton.cpu().numpy()
    arr = np.asarray(skeleton)
    if arr.ndim < 2:
        arr = np.atleast_2d(arr)
    if direction == "horizontal":
        arr = np.fliplr(arr)
    elif direction == "vertical":
        arr = np.flipud(arr)
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")
    return torch.tensor(arr.copy(), dtype=torch.float32)


def flip_graph(graph: nx.Graph, direction: str, image_size: Tuple[int, int]) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    H, W = image_size
    flipped_pos = {}
    for node, data in graph.nodes(data=True):
        x, y = data["pos"]
        if direction == "horizontal":
            flipped_pos[node] = (W - x, y)
        else:
            flipped_pos[node] = (x, H - y)
    out = graph.copy()
    nx.set_node_attributes(out, flipped_pos, "pos")
    return out
