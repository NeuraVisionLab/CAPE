import torch
from torch import nn
import numpy as np
import skimage.graph
import random
import cv2
from skimage.morphology import skeletonize
from utils.graph_from_skeleton_3D import graph_from_skeleton as graph_from_skeleton_3D
from utils.graph_from_skeleton_2D import graph_from_skeleton as graph_from_skeleton_2D
from skimage.draw import line_nd
from scipy.ndimage import binary_dilation, generate_binary_structure


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CAPE(nn.Module):
    
    def __init__(self, window_size=128, three_dimensional=False, distance_threshold=20, dilation_radius=10): 
        """
        Initialize the CAPE loss module.

        Args:
            window_size (int): Size of the window for processing image patches (default: 128 for 2D, 48 for 3D).
            three_dimensional (bool): Flag to indicate 3D processing (default: False for 2D).
            distance_threshold (float): Maximum distance value for cost map computation (default: 20).
            dilation_radius (int): Radius for dilating the ground truth path mask (default: 10 for 2D, 5 for 3D).
        """
        super().__init__()
        self.window_size = window_size
        self.three_dimensional = three_dimensional
        self.distance_threshold = distance_threshold      
        self.dilation_radius = dilation_radius 
        

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
    
    
    def draw_line_with_thickness_3D(self, volume, start_point, end_point, value=1, thickness=1):
        """
        Draw a 3D line with specified thickness between two points in a volume using dilation.
        """
        rr, cc, zz = line_nd(start_point, end_point) 
        volume[zz, cc, rr] = value 
        
        struct = generate_binary_structure(3, 1) 
        dilated_volume = binary_dilation(volume, structure=struct, iterations=thickness)
        
        return dilated_volume


    def path_cost_2D(self, cost_tensor, pred_cost_map, ground_truth, start_point, end_point, dilation_radius=20):
        """
        Compute the shortest path cost in 2D using Dijkstra's algorithm.
        """
        dilated_image = np.zeros_like(ground_truth, dtype=np.uint8)
      
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point   = (int(end_point[0]), int(end_point[1]))
        dilation_radius = int(dilation_radius)

        cv2.line(dilated_image, start_point, end_point, color=1, thickness=dilation_radius)

        pred_cost_map = self.distance_threshold - pred_cost_map
        dilated_image = dilated_image * pred_cost_map
        dilated_image = self.distance_threshold - dilated_image
        dilated_image = np.where(dilated_image == self.distance_threshold, float('inf'), dilated_image)
        path_cost = torch.tensor(0.0, requires_grad=True).to(device)
        
        start_refined = self.find_min_in_radius_2D(dilated_image, start_point, radius=5)
        end_refined = self.find_min_in_radius_2D(dilated_image, end_point, radius=5)


        dilated_image = np.maximum(dilated_image, 0) + 0.00001
        
        try:

            path_coords, _ = skimage.graph.route_through_array(
                dilated_image, start=start_refined, end=end_refined, fully_connected=True, geometric=True)

            path_coords = np.transpose(np.array(path_coords), (1, 0))
            path_cost = torch.sum(cost_tensor[path_coords[0], path_coords[1]] ** 2).to(device)
    
            return path_cost
        
        except Exception as e:

            return path_cost
        
        
    def path_cost_3D(self, cost_tensor, pred_cost_map, ground_truth, start_point, end_point, dilation_radius=5):
        """
        Compute the shortest path cost in 3D using Dijkstra's algorithm.
        """
        dilated_image = np.zeros_like(ground_truth, dtype=np.uint8)
        dilated_image = self.draw_line_with_thickness_3D(dilated_image, start_point, end_point, value=1, thickness=dilation_radius)
        
        dilated_image = dilated_image.astype(np.uint8)
        dilated_image = np.where(dilated_image, 1, 0)
        
        pred_cost_map_temp = self.distance_threshold - pred_cost_map
        dilated_image = dilated_image * pred_cost_map_temp
        dilated_image = self.distance_threshold - dilated_image
        dilated_image = np.where(dilated_image == self.distance_threshold, float('inf'), dilated_image)

        start_refined = self.find_min_in_radius_3D(dilated_image, start_point, radius=3)
        end_refined = self.find_min_in_radius_3D(dilated_image, end_point, radius=3)
        
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
        Compute the total CAPE loss for a batch of images by processing patches (windows).
        Iteratively samples paths from the ground truth graph and computes the loss for each path.

        Args:
            predictions (torch.Tensor): Predicted distance maps (batch_size, height, width) for 2D
                                      or (batch_size, depth, height, width) for 3D.
            ground_truths (torch.Tensor): Ground truth binary masks with same shape as predictions.

        Returns:
            torch.Tensor: Average CAPE loss across the batch.
        """
        if self.three_dimensional == False:

            batch_size = predictions.size(0)

            total_loss = 0.0

            for b in range(batch_size):

                full_prediction_map = predictions[b]
                full_ground_truth_mask = ground_truths[b].detach().cpu().numpy()

                assert ground_truths.shape[-1] % self.window_size == 0, "Width must be divisible by window size"
                assert ground_truths.shape[-2] % self.window_size == 0, "Height must be divisible by window size"
                
                num_windows_height = ground_truths.shape[-2] // self.window_size
                num_windows_width = ground_truths.shape[-1] // self.window_size

                crop_loss_sum = 0.0

                for i in range(num_windows_height):
                    for j in range(num_windows_width):
                        window_pred = full_prediction_map[i * full_prediction_map.shape[0] // num_windows_height:(i + 1) * full_prediction_map.shape[0] // num_windows_height, :]
                        window_pred = window_pred[:, j * full_prediction_map.shape[1] // num_windows_width:(j + 1) * full_prediction_map.shape[1] // num_windows_width]
                        window_gt = full_ground_truth_mask[i * full_prediction_map.shape[0] // num_windows_height:(i + 1) * full_prediction_map.shape[0] // num_windows_height, :]
                        window_gt = 1 - window_gt[:, j * full_prediction_map.shape[1] // num_windows_width:(j + 1) * full_prediction_map.shape[1] // num_windows_width]

                        skeleton = skeletonize(window_gt)

                        graph = graph_from_skeleton_2D(skeleton, angle_range=(175,185), verbose=False)
                            
                        
                        edges = list(graph.edges())

                        window_loss = 0.0
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
                                window_pred,
                                window_pred.detach().cpu().numpy(),
                                window_gt,
                                node1_pos,
                                node2_pos,
                                self.dilation_radius,
                            )

                            graph.remove_edge(*edge)
                            window_loss += single_loss
                            
                        crop_loss_sum += window_loss
                        
                total_loss += crop_loss_sum
                
            return total_loss / batch_size if batch_size > 0 else 0


        else:
            
            batch_size = predictions.size(0)
            
            total_loss = 0.0

            for b in range(batch_size):
                full_prediction_map = predictions[b]
                full_ground_truth_mask = ground_truths[b].detach().cpu().numpy()

                assert ground_truths.shape[-3] % self.window_size == 0, "Depth must be divisible by window size"
                assert ground_truths.shape[-2] % self.window_size == 0, "Height must be divisible by window size"
                assert ground_truths.shape[-1] % self.window_size == 0, "Width must be divisible by window size"

                num_windows_depth = ground_truths.shape[-3] // self.window_size
                num_windows_height = ground_truths.shape[-2] // self.window_size
                num_windows_width = ground_truths.shape[-1] // self.window_size
                
                crop_loss_sum = 0.0
                for d in range(num_windows_depth):
                    for i in range(num_windows_height):
                        for j in range(num_windows_width):
                            window_pred = full_prediction_map[
                                d * self.window_size:(d + 1) * self.window_size,
                                i * self.window_size:(i + 1) * self.window_size,
                                j * self.window_size:(j + 1) * self.window_size
                            ]
                            window_gt= full_ground_truth_mask[
                                d * self.window_size:(d + 1) * self.window_size,
                                i * self.window_size:(i + 1) * self.window_size,
                                j * self.window_size:(j + 1) * self.window_size
                            ]
                            
                            skeleton = skeletonize(window_gt)

                            graph = graph_from_skeleton_3D(skeleton, angle_range=(175,185), verbose=False)

                            window_loss = 0.0
                            while list(graph.edges()):
                                edge = random.choice(list(graph.edges()))
                                node1, node2 = edge
                                node1_pos = graph.nodes[node1]['pos']
                                node2_pos = graph.nodes[node2]['pos']

                                single_loss = self.path_cost_3D(
                                    window_pred,
                                    window_pred.detach().cpu().numpy(),
                                    window_gt,
                                    node1_pos,
                                    node2_pos,
                                    self.dilation_radius
                                )
                                
                                graph.remove_edge(*edge)
                                window_loss += single_loss
                                
                            crop_loss_sum += window_loss
                            
                total_loss += crop_loss_sum
                
            return total_loss / batch_size if batch_size > 0 else 0


