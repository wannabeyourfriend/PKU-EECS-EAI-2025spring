from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config
from ..vis import Vis


class PointNetFeatureExtractor(nn.Module):
    """PointNet feature extractor for point cloud processing"""
    
    def __init__(self, input_dim=3, feature_dim=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        
    def forward(self, x):
        """
        Forward pass of PointNet feature extractor
        
        Parameters
        ----------
        x : torch.Tensor
            Input point cloud, shape (B, N, 3)
            
        Returns
        -------
        torch.Tensor
            Point-wise features, shape (B, N, feature_dim)
        torch.Tensor
            Global feature vector, shape (B, feature_dim)
        """
        batch_size, num_points, _ = x.shape
        
        # x: (B, N, 3) -> (B, 3, N) for conv1d
        x = x.transpose(2, 1)
        
        # Point-wise feature extraction
        x1 = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x2 = F.relu(self.bn2(self.conv2(x1)))  # (B, 128, N)
        x3 = self.bn3(self.conv3(x2))          # (B, feature_dim, N)
        
        # Global max pooling to get permutation invariant features
        global_feat = torch.max(x3, 2)[0]  # (B, feature_dim)
        
        # Expand global features to each point
        global_feat_expanded = global_feat.unsqueeze(2).expand(-1, -1, num_points)  # (B, feature_dim, N)
        
        # Concatenate point-wise and global features
        point_feat = torch.cat([x1, x2, x3, global_feat_expanded], dim=1)  # (B, 64+128+1024+1024, N)
        
        # Transpose back to (B, N, feature_dim)
        point_feat = point_feat.transpose(2, 1)  # (B, N, 2240)
        
        return point_feat, global_feat


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config
        
        # PointNet feature extractor
        self.feature_extractor = PointNetFeatureExtractor(input_dim=3, feature_dim=1024)
        
        # Coordinate prediction head - predicts 3D coordinates for each point
        self.coord_head = nn.Sequential(
            nn.Linear(2240, 512),  # 64+128+1024+1024 = 2240
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3D coordinates in object frame
        )

    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # Extract features from point cloud
        point_feat, global_feat = self.feature_extractor(pc)  # (B, N, 2240), (B, 1024)
        
        # Predict coordinates for each point
        pred_coord = self.coord_head(point_feat)  # (B, N, 3)
        
        # Compute coordinate prediction loss
        coord_loss = F.mse_loss(pred_coord, coord)
        
        # Compute metrics
        coord_error = torch.mean(torch.norm(pred_coord - coord, dim=2))
        
        loss = coord_loss
        metric = dict(
            loss=loss.item(),
            coord_loss=coord_loss.item(),
            coord_error=coord_error.item(),
        )
        return loss, metric

    def _fit_pose_from_coordinates(self, pc_cam: torch.Tensor, coord_obj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fit pose from corresponding points using Kabsch algorithm
        
        Parameters
        ----------
        pc_cam : torch.Tensor
            Points in camera frame, shape (N, 3)
        coord_obj : torch.Tensor
            Corresponding points in object frame, shape (N, 3)
            
        Returns
        -------
        trans : torch.Tensor
            Translation vector, shape (3,)
        rot : torch.Tensor
            Rotation matrix, shape (3, 3)
        """
        # Convert to numpy for easier computation
        pc_cam_np = pc_cam.detach().cpu().numpy()
        coord_obj_np = coord_obj.detach().cpu().numpy()
        
        # Center the point sets
        centroid_cam = np.mean(pc_cam_np, axis=0)
        centroid_obj = np.mean(coord_obj_np, axis=0)
        
        pc_cam_centered = pc_cam_np - centroid_cam
        coord_obj_centered = coord_obj_np - centroid_obj
        
        # Compute cross-covariance matrix
        H = coord_obj_centered.T @ pc_cam_centered
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute translation
        t = centroid_cam - R @ centroid_obj
        
        # Convert back to torch tensors
        device = pc_cam.device
        dtype = pc_cam.dtype
        
        rot_tensor = torch.from_numpy(R).to(device=device, dtype=dtype)
        trans_tensor = torch.from_numpy(t).to(device=device, dtype=dtype)
        
        return trans_tensor, rot_tensor

    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        batch_size = pc.shape[0]
        device = pc.device
        dtype = pc.dtype
        
        # Extract features and predict coordinates
        point_feat, global_feat = self.feature_extractor(pc)  # (B, N, 2240), (B, 1024)
        pred_coord = self.coord_head(point_feat)  # (B, N, 3)
        
        # Initialize output tensors
        trans_batch = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        rot_batch = torch.zeros(batch_size, 3, 3, device=device, dtype=dtype)
        
        # Process each sample in the batch
        for i in range(batch_size):
            pc_i = pc[i]  # (N, 3)
            coord_i = pred_coord[i]  # (N, 3)
            
            # Fit pose using Kabsch algorithm
            trans_i, rot_i = self._fit_pose_from_coordinates(pc_i, coord_i)
            
            trans_batch[i] = trans_i
            rot_batch[i] = rot_i
        
        return trans_batch, rot_batch
