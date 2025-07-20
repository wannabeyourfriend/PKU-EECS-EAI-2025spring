from typing import Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F

from ..config import Config


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
            Global feature vector, shape (B, feature_dim)
        """
        # x: (B, N, 3) -> (B, 3, N) for conv1d
        x = x.transpose(2, 1)
        
        # Point-wise feature extraction
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.bn3(self.conv3(x))          # (B, feature_dim, N)
        
        # Global max pooling to get permutation invariant features
        x = torch.max(x, 2)[0]  # (B, feature_dim)
        
        return x


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config
        
        # PointNet feature extractor
        self.feature_extractor = PointNetFeatureExtractor(input_dim=3, feature_dim=1024)
        
        # Translation head
        self.trans_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3)  # 3D translation vector
        )
        
        # Rotation head - predict 6D rotation representation (continuous 6D representation)
        # This is more stable than directly predicting rotation matrices or quaternions
        self.rot_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)  # 6D rotation representation
        )

    def _6d_to_rotation_matrix(self, rot_6d):
        """
        Convert 6D rotation representation to rotation matrix
        
        Parameters
        ----------
        rot_6d : torch.Tensor
            6D rotation representation, shape (B, 6)
            
        Returns
        -------
        torch.Tensor
            Rotation matrix, shape (B, 3, 3)
        """
        batch_size = rot_6d.shape[0]
        
        # Reshape to (B, 2, 3)
        rot_6d = rot_6d.view(batch_size, 2, 3)
        
        # First column (normalized)
        a1 = rot_6d[:, 0, :]  # (B, 3)
        a1 = F.normalize(a1, dim=1)
        
        # Second column (orthogonalized and normalized)
        a2 = rot_6d[:, 1, :]  # (B, 3)
        # Gram-Schmidt orthogonalization
        a2 = a2 - torch.sum(a1 * a2, dim=1, keepdim=True) * a1
        a2 = F.normalize(a2, dim=1)
        
        # Third column (cross product)
        a3 = torch.cross(a1, a2, dim=1)
        
        # Stack to form rotation matrix
        rotation_matrix = torch.stack([a1, a2, a3], dim=2)  # (B, 3, 3)
        
        return rotation_matrix

    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # Extract features from point cloud
        features = self.feature_extractor(pc)  # (B, 1024)
        
        # Predict translation and rotation
        pred_trans = self.trans_head(features)  # (B, 3)
        pred_rot_6d = self.rot_head(features)   # (B, 6)
        pred_rot = self._6d_to_rotation_matrix(pred_rot_6d)  # (B, 3, 3)
        
        # Compute losses
        trans_loss = F.mse_loss(pred_trans, trans)
        
        # Rotation loss using Frobenius norm
        rot_loss = F.mse_loss(pred_rot, rot)
        
        # Total loss
        loss = trans_loss + rot_loss
        
        # Compute metrics
        trans_error = torch.mean(torch.norm(pred_trans - trans, dim=1))
        
        # Rotation error in radians
        with torch.no_grad():
            # Compute rotation distance
            rot_diff = torch.bmm(pred_rot, rot.transpose(1, 2))
            trace = torch.diagonal(rot_diff, dim1=1, dim2=2).sum(dim=1)
            rot_error = torch.mean(torch.acos(torch.clamp((trace - 1) / 2, -1, 1)))
        
        metric = dict(
            loss=loss.item(),
            trans_loss=trans_loss.item(),
            rot_loss=rot_loss.item(),
            trans_error=trans_error.item(),
            rot_error=rot_error.item(),
        )
        return loss, metric

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
        """
        # Extract features from point cloud
        features = self.feature_extractor(pc)  # (B, 1024)
        
        # Predict translation and rotation
        pred_trans = self.trans_head(features)  # (B, 3)
        pred_rot_6d = self.rot_head(features)   # (B, 6)
        pred_rot = self._6d_to_rotation_matrix(pred_rot_6d)  # (B, 3, 3)
        
        return pred_trans, pred_rot
