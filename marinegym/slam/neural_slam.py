import torch
import torch.nn as nn
from typing import Optional, Tuple
import numpy as np
from tensordict import TensorDict

class NeuralSLAMWrapper(nn.Module):
    def __init__(
        self,
        map_size: Tuple[int, int] = (100, 100),  # 10m x 10m at 0.1m resolution
        resolution: float = 0.1,
        n_channels: int = 32,
        device: str = "cuda:0"
    ):
        super().__init__()
        self.device = device
        self.map_size = map_size
        self.resolution = resolution
        self.n_channels = n_channels

        # Import Neural-SLAM components
        from external.neural_slam.models import NeuralSLAM
        self.slam = NeuralSLAM(
            map_size=map_size,
            n_channels=n_channels,
            resolution=resolution
        ).to(device)

        # Map storage
        self.map = torch.zeros(
            (1, n_channels, *map_size),
            device=device
        )
        self.pose = torch.zeros(
            (1, 3),  # [x, y, theta]
            device=device
        )
        self.step = 0

    def reset(self, batch_size: int = 1):
        """Reset SLAM for new episode"""
        self.map = torch.zeros(
            (batch_size, self.n_channels, *self.map_size),
            device=self.device
        )
        self.pose = torch.zeros(
            (batch_size, 3),
            device=self.device
        )
        self.step = 0

    def process_frame(
        self,
        rgb: torch.Tensor,      # [B, 1, 3, H, W]
        depth: torch.Tensor,    # [B, 1, 1, H, W]
        pose: torch.Tensor,     # [B, state_dim] - drone state
        intrinsics: Optional[torch.Tensor] = None
    ) -> TensorDict:
        """
        Process one RGBD frame and update SLAM.

        Args:
            rgb: RGB observations [B, 1, 3, H, W]
            depth: Depth observations [B, 1, 1, H, W] (normalized 0-1)
            pose: Drone state [B, state_dim]
            intrinsics: Camera intrinsics [B, ...]

        Returns:
            TensorDict with updated map, pose, and features
        """
        B = rgb.shape[0]

        # Convert depth from normalized to meters
        # Assuming clipping_range was [0.1, 20.0] during normalization
        near, far = 0.1, 20.0
        depth_meters = depth * (far - near) + near  # [B, 1, 1, H, W]

        # Extract position from drone state (first 3 elements)
        # pose is [x, y, z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz]
        position = pose[:, :3]  # [B, 3]
        self.pose[:, :2] = position[:, :2]  # x, y
        self.pose[:, 2] = torch.atan2(pose[:, 10], pose[:, 9])  # yaw from angular velocity

        # Process through Neural-SLAM
        # Reshape inputs: [B, C, H, W]
        rgb_in = rgb.squeeze(1)  # [B, 3, H, W]
        depth_in = depth_meters.squeeze(1).squeeze(1)  # [B, H, W]

        # Neural-SLAM expects depth in meters, RGB in 0-1
        rgb_in = rgb_in.float()
        depth_in = depth_in.float()

        # Forward pass (simplified - see Neural-SLAM for full implementation)
        map_update, new_pose = self.slam(
            rgb_in,
            depth_in,
            self.pose,
            self.map
        )

        # Update map and pose
        self.map = map_update
        self.pose = new_pose
        self.step += 1

        return TensorDict({
            "slam_map": self.map,           # [B, C, H_map, W_map]
            "slam_pose": self.pose,         # [B, 3] (x, y, theta)
            "slam_features": map_update,    # For visualization
        }, batch_size=[B])

    def get_frontiers(self, threshold: float = 0.5) -> torch.Tensor:
        """
        Detect frontiers (boundaries between known/unknown space).

        Args:
            threshold: Occupancy threshold for known space

        Returns:
            Frontier positions [N, 2] in map coordinates
        """
        # Get occupancy from first channel (simplified)
        occupancy = self.map[0, 0]  # [H, W]

        # Find boundaries
        from scipy.ndimage import binary_erosion, binary_dilation
        import cv2 as cv

        # Convert to binary
        binary = (occupancy > threshold).cpu().numpy().astype(np.uint8)

        # Find contours
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Get frontier points (contour points)
        frontiers = []
        for cnt in contours:
            for pt in cnt:
                frontiers.append(pt[0])

        if len(frontiers) == 0:
            return torch.zeros((0, 2), device=self.device)

        return torch.tensor(frontiers, dtype=torch.float32, device=self.device)