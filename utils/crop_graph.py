import numpy as np
from shapely.geometry import box, LineString, Point
import networkx as nx

def crop_graph_2D(graph, xmin, ymin, xmax, ymax, precision=8):
    
    bounding_box = box(xmin, ymin, xmax, ymax)
    
    cropped_graph = nx.Graph()
    node_positions = {}      
    inside_nodes = {}        
    coord_to_node = {}      
    
    
    for n, data in graph.nodes(data=True):
        pos = data['pos']
        node_positions[n] = pos
        x, y = pos
        if xmin <= x <= xmax and ymin <= y <= ymax:
            new_pos = (x - xmin, y - ymin)
            inside_nodes[n] = new_pos
            cropped_graph.add_node(n, pos=new_pos)
    
            key = (round(new_pos[0], precision), round(new_pos[1], precision))
            coord_to_node[key] = n
    
    max_node_index = max(graph.nodes()) if graph.nodes else 0

    for u, v, data in graph.edges(data=True):
        u_pos = node_positions[u]
        v_pos = node_positions[v]
        line = LineString([u_pos, v_pos])
        
        if u in inside_nodes and v in inside_nodes:
            if u != v:
                cropped_graph.add_edge(u, v, **data)
            continue
        
        if not bounding_box.intersects(line):
            continue

        intersection = bounding_box.intersection(line)
        if intersection.is_empty:
            continue

        if intersection.geom_type == 'Point':
            pts = [(intersection.x, intersection.y)]
        elif intersection.geom_type == 'MultiPoint':
            pts = [(pt.x, pt.y) for pt in intersection.geoms]
        elif intersection.geom_type == 'LineString':
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
                cropped_graph.add_node(node_id, pos=new_pos)
                coord_to_node[key] = node_id
            new_nodes.append(node_id)
        
        endpoints = []
        if u in inside_nodes:
            endpoints.append(u)
        endpoints.extend(new_nodes)
        if v in inside_nodes:
            endpoints.append(v)

        for i in range(len(endpoints) - 1):
            if endpoints[i] != endpoints[i+1]:
                cropped_graph.add_edge(endpoints[i], endpoints[i+1], **data)
    
    return cropped_graph


def crop_graph_3D(graph, xmin, ymin, zmin, xmax, ymax, zmax, precision=8):
    
    def _to_voxel(u, lo, hi):
        v = int(np.floor(u - lo))          
        return max(0, min(v, hi - lo - 1)) 

    def _segment_box_intersections(p0, p1):

        pts = []
        (x0, y0, z0), (x1, y1, z1) = p0, p1
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0

        for plane, (k0, k1, p) in (
            ("x", (x0, dx, xmin)), ("x", (x0, dx, xmax)),
            ("y", (y0, dy, ymin)), ("y", (y0, dy, ymax)),
            ("z", (z0, dz, zmin)), ("z", (z0, dz, zmax)),
        ):
            k0, dk, plane_val = k0, k1, p
            if dk == 0:                        
                continue
            t = (plane_val - k0) / dk
            if 0 < t < 1:                      
                x = x0 + t * dx
                y = y0 + t * dy
                z = z0 + t * dz
                if xmin - 1e-6 <= x <= xmax + 1e-6 and \
                ymin - 1e-6 <= y <= ymax + 1e-6 and \
                zmin - 1e-6 <= z <= zmax + 1e-6:
                    pts.append((x, y, z))

        pts.sort(key=lambda pt: (pt[0]-x0)**2 + (pt[1]-y0)**2 + (pt[2]-z0)**2)
        return pts

    cropped = nx.Graph()
    inside_nodes, pos_cache, coord2id = {}, {}, {}

    for n, d in graph.nodes(data=True):
        x, y, z = d["pos"]
        pos_cache[n] = (x, y, z)
        if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
            vx, vy, vz = (_to_voxel(x, xmin, xmax),
                        _to_voxel(y, ymin, ymax),
                        _to_voxel(z, zmin, zmax))
            
            inside_nodes[n] = (vx, vy, vz)
            cropped.add_node(n, pos=inside_nodes[n])
            coord2id[(vx, vy, vz)] = n

    next_id = (max(graph.nodes()) if graph.nodes else 0) + 1

    for u, v, edata in graph.edges(data=True):
        p0, p1 = pos_cache[u], pos_cache[v]

        if (u not in inside_nodes) and (v not in inside_nodes):

            if (p0[0] < xmin and p1[0] < xmin) or (p0[0] > xmax and p1[0] > xmax) \
            or (p0[1] < ymin and p1[1] < ymin) or (p0[1] > ymax and p1[1] > ymax) \
            or (p0[2] < zmin and p1[2] < zmin) or (p0[2] > zmax and p1[2] > zmax):
                continue

        split_pts = _segment_box_intersections(p0, p1)

        node_chain = []
        if u in inside_nodes:
            node_chain.append(u)

        for pt in split_pts:
            
            vz = _to_voxel(pt[2], zmin, zmax)
            vy = _to_voxel(pt[1], ymin, ymax)
            vx = _to_voxel(pt[0], xmin, xmax)
        
            key = (vx, vy, vz)
            if key in coord2id:
                node_id = coord2id[key]
            else:
                node_id = next_id
                next_id += 1
                cropped.add_node(node_id, pos=(vx, vy, vz))
                coord2id[key] = node_id
            node_chain.append(node_id)

        if v in inside_nodes:
            node_chain.append(v)

        for a, b in zip(node_chain[:-1], node_chain[1:]):
            if a != b:
                cropped.add_edge(a, b, **edata)

    return cropped

