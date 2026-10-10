import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F
from marinegym.slam.neural_slam import NeuralSLAMWrapper

class FrontierExploration:
    def __init__(
        self,
        slam: NeuralSLAMWrapper,
        drone_radius: float = 0.5,  # meters
        frontier_threshold: float = 0.5,
        max_frontiers: int = 10,
        device: str = "cuda:0"
    ):
        self.slam = slam
        self.drone_radius = drone_radius
        self.frontier_threshold = frontier_threshold
        self.max_frontiers = max_frontiers
        self.device = device

    def get_frontiers(self) -> torch.Tensor:
        """Get frontier positions in world coordinates"""
        frontiers_map = self.slam.get_frontiers(self.frontier_threshold)

        if len(frontiers_map) == 0:
            return torch.zeros((0, 2), device=self.device)

        # Convert map coordinates to world coordinates
        # Map origin is at (0,0) in world
        frontiers_world = frontiers_map * self.slam.resolution
        return frontiers_world[:self.max_frontiers]

    def select_closest_frontier(
        self,
        drone_pos: torch.Tensor  # [2] (x, y)
    ) -> Tuple[torch.Tensor, float]:
        """
        Select the closest frontier to the drone.

        Returns:
            (frontier_pos, distance)
        """
        frontiers = self.get_frontiers()

        if len(frontiers) == 0:
            return torch.zeros(2, device=self.device), float('inf')

        # Calculate distances
        diff = frontiers - drone_pos.unsqueeze(0)  # [N, 2]
        dists = torch.norm(diff, dim=1)  # [N]

        # Filter frontiers that are too close (already explored)
        valid = dists > self.drone_radius
        if not valid.any():
            return torch.zeros(2, device=self.device), float('inf')

        frontiers = frontiers[valid]
        dists = dists[valid]

        # Select closest
        closest_idx = torch.argmin(dists)
        return frontiers[closest_idx], dists[closest_idx].item()

    def get_exploration_goal(self, drone_pos: torch.Tensor) -> torch.Tensor:
        """Get the next exploration goal position"""
        goal, _ = self.select_closest_frontier(drone_pos)
        return goal

    def compute_exploration_reward(
        self,
        drone_pos: torch.Tensor,
        prev_map_coverage: float
    ) -> Tuple[float, float]:
        """
        Compute exploration reward.

        Returns:
            (reward, new_map_coverage)
        """
        # Current map coverage
        map_occupancy = self.slam.map[0, 0]  # [H, W]
        current_coverage = (map_occupancy > self.frontier_threshold).float().mean().item()

        # Coverage increase reward
        coverage_reward = current_coverage - prev_map_coverage

        # Frontier proximity reward
        goal = self.get_exploration_goal(drone_pos)
        if goal.sum() == 0:  # No frontiers
            proximity_reward = 0.0
        else:
            dist = torch.norm(goal - drone_pos)
            proximity_reward = max(0, 1.0 - dist)  # Reward for being close

        total_reward = coverage_reward * 10.0 + proximity_reward * 5.0
        return total_reward, current_coverage