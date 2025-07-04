import torch
from torch import nn
import numpy as np
import skimage.graph
import random
import cv2
from skimage.morphology import skeletonize
from utils.graph_from_skeleton_3D import graph_from_skeleton as graph_from_skeleton_3D
from utils.graph_from_skeleton_2D import graph_from_skeleton as graph_from_skeleton_2D
from utils.crop_graph import crop_graph_2D, crop_graph_3D
from skimage.draw import line_nd
from scipy.ndimage import binary_dilation, generate_binary_structure
import networkx as nx


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CAPE(nn.Module):
    def __init__(self, window_size=128, three_dimensional=False, dilation_radius=10, shifting_radius=5, is_binary=False, distance_threshold=20, single_edge=False):
        super().__init__()
        """
        Initialize the CAPE loss module.

        Args:
            window_size (int): Size of the square patch (window) to process at a time.
            three_dimensional (bool): If True, operate in 3D mode; otherwise, operate in 2D.
            dilation_radius (int): Radius used to dilate ground-truth paths for masking.
            shifting_radius (int): Radius for refining start/end points to lowest-cost nearby pixels.
            is_binary (bool): If True, treat inputs as binary maps (invert predictions/ground truth).
            distance_threshold (float): Maximum value used for clipping ground-truth distance maps.
            single_edge (bool): If True, sample a single edge at a time; otherwise, sample a path.
        """
        self.window_size = window_size
        self.three_dimensional = three_dimensional
        self.dilation_radius = dilation_radius
        self.shifting_radius = shifting_radius
        self.is_binary = is_binary
        self.distance_threshold = distance_threshold
        self.single_edge = single_edge
        

    def _random_connected_pair(self, G):
        """
        Pick two distinct nodes that are in the same connected component.
        """
        node1 = random.choice(list(G.nodes()))
        reachable = list(nx.node_connected_component(G, node1))
        if len(reachable) == 1:
            return self._random_connected_pair(G)
        node2 = random.choice([n for n in reachable if n != node1])
        return node1, node2


    def _dilate_path_2D(self, shape, path_pts, radius):
        """
        Rasterise a poly-line into a thick 2D mask.
        """
        mask = np.zeros(shape, dtype=np.uint8)
        for p, q in zip(path_pts[:-1], path_pts[1:]):
            cv2.line(mask,
                    (int(p[0]), int(p[1])),
                    (int(q[0]), int(q[1])),
                    1, int(radius))
        return mask
    
    
    def _dilate_path_3D(self, shape, path_positions, radius):
        """
        Rasterise a poly-line into a thick 3D mask.
        """
        mask = np.zeros(shape, dtype=np.uint8)

        for p1, p2 in zip(path_positions[:-1], path_positions[1:]):
            temp = np.zeros(shape, dtype=np.uint8)
            
            rr, cc, zz = line_nd(tuple(map(int, p1)), tuple(map(int, p2)))
            temp[zz, cc, rr] = 1

            struct = generate_binary_structure(3, 1)
            dilated_segment = binary_dilation(temp, structure=struct, iterations=int(radius))

            mask = np.logical_or(mask, dilated_segment)

        return mask.astype(np.uint8)
        
        
    def draw_line_with_thickness_3D(self, volume, start_point, end_point, value=1, thickness=1):
        """
        Draw a 3D line with specified thickness between two points in a volume using dilation.
        """
        rr, cc, zz = line_nd(start_point, end_point)
        volume[zz, cc, rr] = value
        
        struct = generate_binary_structure(3, 1)
        dilated_volume = binary_dilation(volume, structure=struct, iterations=thickness)
        
        return dilated_volume
      

    def find_min_in_radius_2D(self, array: np.ndarray, center: tuple, radius: float):
        """
        Finds the coordinates of the minimum value inside a given radius from a center point in a 2D array.
        """
        x0, y0 = center
        height, width = array.shape

        y_min, y_max = max(0, y0 - int(radius)), min(height, y0 + int(radius) + 1)
        x_min, x_max = max(0, x0 - int(radius)), min(width, x0 + int(radius) + 1)

        sub_image = array[y_min:y_max, x_min:x_max]
        
        min_idx = np.unravel_index(np.argmin(sub_image), sub_image.shape)

        min_coords = (y_min + min_idx[0], x_min + min_idx[1])
        return min_coords
    
    
    def find_min_in_radius_3D(self, array: np.ndarray, center: tuple, radius: float):
        """
        Finds the coordinates of the minimum value inside a given radius from a center point in a 3D array.
        """
        x0, y0, z0 = center
        depth, height, width = array.shape

        z_min, z_max = max(0, z0 - int(radius)), min(depth, z0 + int(radius) + 1)
        y_min, y_max = max(0, y0 - int(radius)), min(height, y0 + int(radius) + 1)
        x_min, x_max = max(0, x0 - int(radius)), min(width, x0 + int(radius) + 1)

        sub_volume = array[z_min:z_max, y_min:y_max, x_min:x_max]
        
        
        min_idx = np.unravel_index(np.argmin(sub_volume), sub_volume.shape)
        
        min_coords = (z_min + min_idx[0], y_min + min_idx[1], x_min + min_idx[2])
        return min_coords
    
    
    def path_cost_2D(self, cost_tensor, pred_cost_map, start_point, end_point, dilation_radius=20, extra_path=None):  
        """
        Compute the shortest path cost in 2D using Dijkstra's algorithm.
        """
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point   = (int(end_point[0]), int(end_point[1]))
        dilation_radius = int(dilation_radius)

        if extra_path is None:                            
            dilated_image = np.zeros_like(pred_cost_map, dtype=np.uint8)
            cv2.line(dilated_image, start_point, end_point,
                    color=1, thickness=int(dilation_radius))
        else:                                             
            dilated_image = self._dilate_path_2D(pred_cost_map.shape,
                                                extra_path,
                                                dilation_radius)

        pred_cost_map = self.distance_threshold - pred_cost_map
        dilated_image = dilated_image * pred_cost_map
        dilated_image = self.distance_threshold - dilated_image
        dilated_image = np.where(dilated_image == self.distance_threshold, float('inf'), dilated_image)
        path_cost = torch.tensor(0.0, requires_grad=True).to(device)
        
        start_refined = self.find_min_in_radius_2D(dilated_image, start_point, radius=self.shifting_radius)
        end_refined = self.find_min_in_radius_2D(dilated_image, end_point, radius=self.shifting_radius)

        dilated_image = np.maximum(dilated_image, 0) + 0.00001
        
        try:

            path_coords, _ = skimage.graph.route_through_array(
                dilated_image, start=start_refined, end=end_refined, fully_connected=True, geometric=True)

            path_coords = np.transpose(np.array(path_coords), (1, 0))
            path_cost = torch.sum(cost_tensor[path_coords[0], path_coords[1]] ** 2).to(device)
            
            return path_cost
        
        except Exception as e:

            return path_cost
        
        
    def path_cost_3D(self, cost_tensor, pred_cost_map, start_point, end_point, dilation_radius=5, extra_path=None):
        """
        Compute the shortest path cost in 3D using Dijkstra's algorithm.
        """
        if extra_path is None:
            dilated_image = self.draw_line_with_thickness_3D(
                np.zeros_like(pred_cost_map, dtype=np.uint8),
                start_point, end_point, value=1, thickness=dilation_radius)
            
        else:                                           
            dilated_image = self._dilate_path_3D(pred_cost_map.shape,
                                                extra_path,
                                                dilation_radius)
        
        
        dilated_image = dilated_image.astype(np.uint8)
        dilated_image = np.where(dilated_image, 1, 0)
        
        pred_cost_map_temp = self.distance_threshold - pred_cost_map
        dilated_image = dilated_image * pred_cost_map_temp
        dilated_image = self.distance_threshold - dilated_image
        dilated_image = np.where(dilated_image == self.distance_threshold, float('inf'), dilated_image)
        
        start_refined = self.find_min_in_radius_3D(dilated_image, start_point, radius=self.shifting_radius)
        end_refined = self.find_min_in_radius_3D(dilated_image, end_point, radius=self.shifting_radius)
        
        dilated_image = np.maximum(dilated_image, 0) + 0.00001
        
        try:
            path_coords, _ = skimage.graph.route_through_array(
                dilated_image, start=start_refined, end=end_refined, fully_connected=True, geometric=True
            )
            path_coords = np.array(path_coords).T
            path_cost = torch.sum((cost_tensor[path_coords[0], path_coords[1], path_coords[2]]) ** 2).to(device)
            
            return path_cost
        
        except Exception as e:
            return torch.tensor(0.0, requires_grad=True).to(device)
        

    def forward(self, predictions, ground_truths):
        """
        Compute the average CAPE loss over a batch of predictions and ground truths.

        The method splits each prediction volume/mask into patches (windows), extracts
        or receives a graph representation of the skeletonized ground truth in each window,
        samples paths from the graph, computes the squared-distance sum along each predicted path,
        and accumulates these costs to return the mean loss per batch.

        Args:
            predictions (torch.Tensor): Distance maps of shape
                - (batch, H, W) for 2D
                - (batch, D, H, W) for 3D
            ground_truths (Union[nx.Graph, np.ndarray, torch.Tensor]):
                - Graph objects for direct skeleton-based sampling,
                - Or binary masks (arrays or tensors) to skeletonize.

        Returns:
            torch.Tensor: Scalar tensor representing the mean CAPE loss over the batch.
        """
        batch_size = predictions.size(0)
        total_loss = 0.0

        if isinstance(ground_truths[0], nx.Graph):
            gt_type = 0
        
        elif isinstance(ground_truths, np.ndarray):
            gt_type = 1
            
        elif isinstance(ground_truths, torch.Tensor):
            ground_truths = ground_truths.detach().cpu().numpy()
            gt_type = 1
        
        
        if self.is_binary:
            
            predictions = 1 - predictions
            
            if gt_type == 1:
                ground_truths = 1 - ground_truths
                
            self.distance_threshold = 1
        
        
        
        # ── 2D MODE ───────────────────────────────────────────────────────────────
        
        if self.three_dimensional == False:

            for b in range(batch_size):

                full_prediction_map = predictions[b]
                
                # NO GRAPH INPUT
                if gt_type == 1:
                    full_ground_truth_mask = (ground_truths[b] == 0).astype(np.uint8)
                
                # GRAPH INPUT    
                elif gt_type == 0:
                    complete_graph = ground_truths[b]

                assert predictions.shape[-1] % self.window_size == 0, "Width must be divisible by window size"
                assert predictions.shape[-2] % self.window_size == 0, "Height must be divisible by window size"

                num_windows_height = predictions.shape[-2] // self.window_size
                num_windows_width = predictions.shape[-1] // self.window_size

                crop_loss_sum = 0.0
                

                for i in range(num_windows_height):
                    for j in range(num_windows_width):
                        window_pred = full_prediction_map[i * full_prediction_map.shape[0] // num_windows_height:(i + 1) * full_prediction_map.shape[0] // num_windows_height, :]
                        window_pred = window_pred[:, j * full_prediction_map.shape[1] // num_windows_width:(j + 1) * full_prediction_map.shape[1] // num_windows_width]
                        
                        # NO GRAPH INPUT
                        if gt_type == 1:
                            
                            window_gt = full_ground_truth_mask[i * full_prediction_map.shape[0] // num_windows_height:(i + 1) * full_prediction_map.shape[0] // num_windows_height, :]
                            window_gt = window_gt[:, j * full_prediction_map.shape[1] // num_windows_width:(j + 1) * full_prediction_map.shape[1] // num_windows_width]

                            skeleton = skeletonize(window_gt)
                            graph = graph_from_skeleton_2D(skeleton, angle_range=(175,185), verbose=False)
                        
                        # GRAPH INPUT
                        elif gt_type == 0:
                            graph = crop_graph_2D(complete_graph,
                                                ymin=i * full_prediction_map.shape[0] // num_windows_height,
                                                xmin=j * full_prediction_map.shape[1] // num_windows_width,
                                                ymax=(i + 1) * full_prediction_map.shape[0] // num_windows_height,
                                                xmax=(j + 1) * full_prediction_map.shape[1] // num_windows_width)

                        window_loss = 0.0
                        
                        if self.single_edge == False:
                        
                            while list(graph.edges()):
                                n1, n2 = self._random_connected_pair(graph)

                                path_nodes = nx.shortest_path(graph, n1, n2)
                                path_pos   = [graph.nodes[n]['pos'] for n in path_nodes]

                                single_loss = self.path_cost_2D(
                                    cost_tensor=window_pred,
                                    pred_cost_map=window_pred.detach().cpu().numpy(),
                                    start_point=path_pos[0], end_point=path_pos[-1],
                                    dilation_radius=self.dilation_radius,
                                    extra_path=path_pos
                                )

                                graph.remove_edges_from(zip(path_nodes[:-1], path_nodes[1:]))
                                window_loss += single_loss                            
                            
                        else:
                            
                            edges = list(graph.edges())
                            
                            while list(graph.edges()):
                                edges = list(graph.edges())
        
                                edge = random.choice(edges)
                                node_1 = edge[0]
                                node_2 = edge[1]
                                
                                node_idx1 = list(graph.nodes).index(node_1)
                                node_idx2 = list(graph.nodes).index(node_2)

                                node1 = list(graph.nodes)[node_idx1]
                                node2 = list(graph.nodes)[node_idx2]

                                node1_pos = graph.nodes()[node1]['pos']
                                node2_pos = graph.nodes()[node2]['pos']

                                single_loss = self.path_cost_2D(
                                    cost_tensor=window_pred,
                                    pred_cost_map=window_pred.detach().cpu().numpy(),
                                    start_point=node1_pos, end_point=node2_pos,
                                    dilation_radius=self.dilation_radius
                                )
                                
                                graph.remove_edge(*edge)
                                window_loss += single_loss
                            
                        crop_loss_sum += window_loss
                        
                total_loss += crop_loss_sum
                
            return total_loss / batch_size if batch_size > 0 else 0

        # ── 3D MODE ───────────────────────────────────────────────────────────────

        else:
            
            for b in range(batch_size):
                full_prediction_map = predictions[b]
                
                # NO GRAPH INPUT
                if gt_type == 1:
                    full_ground_truth_mask = (ground_truths[b] == 0).astype(np.uint8)
                
                # GRAPH INPUT    
                elif gt_type == 0:
                    complete_graph = ground_truths[b]
                    
                assert predictions.shape[-3] % self.window_size == 0, "Depth must be divisible by window size"
                assert predictions.shape[-2] % self.window_size == 0, "Height must be divisible by window size"
                assert predictions.shape[-1] % self.window_size == 0, "Width must be divisible by window size"

                num_windows_depth = predictions.shape[-3] // self.window_size
                num_windows_height = predictions.shape[-2] // self.window_size
                num_windows_width = predictions.shape[-1] // self.window_size
                
                crop_loss_sum = 0.0
                for d in range(num_windows_depth):
                    for i in range(num_windows_height):
                        for j in range(num_windows_width):
                            window_pred = full_prediction_map[
                                d * self.window_size:(d + 1) * self.window_size,
                                i * self.window_size:(i + 1) * self.window_size,
                                j * self.window_size:(j + 1) * self.window_size
                            ]
                            
                            # NO GRAPH INPUT
                            if gt_type == 1:
                                
                                window_gt = full_ground_truth_mask[
                                    d * self.window_size:(d + 1) * self.window_size,
                                    i * self.window_size:(i + 1) * self.window_size,
                                    j * self.window_size:(j + 1) * self.window_size
                                ]
                                
                                skeleton = skeletonize(window_gt)
                                graph = graph_from_skeleton_3D(skeleton, angle_range=(175,185), verbose=False)
                            
                            # GRAPH INPUT
                            elif gt_type == 0:
                                graph = crop_graph_3D(       
                                        complete_graph,
                                        xmin=j * self.window_size,
                                        ymin=i * self.window_size,
                                        zmin=d * self.window_size,
                                        xmax=j * self.window_size + self.window_size,
                                        ymax=i * self.window_size + self.window_size,
                                        zmax=d * self.window_size + self.window_size)
                            
                            window_loss = 0.0
                            
                            if self.single_edge == False:
                            
                                while list(graph.edges()):
                                    n1, n2 = self._random_connected_pair(graph)

                                    path_nodes = nx.shortest_path(graph, n1, n2)
                                    path_pos   = [graph.nodes[n]['pos'] for n in path_nodes]

                                    single_loss = self.path_cost_3D(
                                        cost_tensor=window_pred,
                                        pred_cost_map=window_pred.detach().cpu().numpy(),
                                        start_point=path_pos[0], end_point=path_pos[-1],
                                        dilation_radius=self.dilation_radius,
                                        extra_path=path_pos
                                    )
                                    
                                    graph.remove_edges_from(zip(path_nodes[:-1], path_nodes[1:]))
                                    window_loss += single_loss
                               
                            else: 
                                
                                while list(graph.edges()):
                                    edge = random.choice(list(graph.edges()))
                                    node1, node2 = edge
                                    node1_pos = graph.nodes[node1]['pos']
                                    node2_pos = graph.nodes[node2]['pos']

                                    single_loss = self.path_cost_3D(
                                        cost_tensor=window_pred,
                                        pred_cost_map=window_pred.detach().cpu().numpy(),
                                        start_point=node1_pos, end_point=node2_pos,
                                        dilation_radius=self.dilation_radius
                                    )
                                    
                                    graph.remove_edge(*edge)
                                    window_loss += single_loss
                            
                            crop_loss_sum += window_loss
                            
                total_loss += crop_loss_sum
                
            return total_loss / batch_size if batch_size > 0 else 0


