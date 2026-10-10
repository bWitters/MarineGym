import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils

from marinegym.envs.isaac_env import AgentSpec, IsaacEnv
from marinegym.views import ArticulationView, RigidPrimView
from marinegym.utils.torch import euler_to_quaternion
from marinegym.sensors.camera import Camera
from marinegym.sensors.config import PinholeCameraCfg
from marinegym.slam.neural_slam import NeuralSLAMWrapper
from marinegym.slam.frontier_exploration import FrontierExploration

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

from marinegym.robots.drone import UnderwaterVehicle

class SearchAndRescue(IsaacEnv):
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.time_encoding = cfg.task.time_encoding

        super().__init__(cfg, headless)

        # Initialize SLAM and exploration
        self.slam = NeuralSLAMWrapper(
            map_size=tuple(cfg.task.slam.map_size),
            resolution=cfg.task.slam.resolution,
            n_channels=cfg.task.slam.n_channels,
            device=self.device
        )
        self.exploration = FrontierExploration(
            self.slam,
            drone_radius=cfg.task.exploration.drone_radius,
            device=self.device
        )
        self.prev_map_coverage = 0.0

        self.drone.initialize()
        if self.cfg.task.camera.enabled:
            self.camera.initialize(
                prim_paths_expr=(
                    f"/World/envs/env_.*/{self.drone.name}_0/"
                    "base_link/Camera_0"
                )
            )

        self.target_vis = ArticulationView(
            "/World/envs/env_*/target", reset_xform_properties=False
        )
        self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

    def _design_scene(self):
        drone_model_cfg = self.cfg.task.drone_model
        self.drone, self.controller = UnderwaterVehicle.make(
            drone_model_cfg.name, drone_model_cfg.controller
        )

        from marinegym.robots.robot import ASSET_PATH
        import omni.isaac.core.utils.stage as stage_utils
        stage_utils.add_reference_to_stage(
            usd_path=ASSET_PATH + "/usd/worlds/EmptyMarine.usd",
            prim_path="/World/defaultGroundPlane"
        )

        # Add some obstacles for exploration
        self._add_obstacles()

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.)])[0]
        if self.cfg.task.camera.enabled:
            cam = self.cfg.task.camera
            self.camera = Camera(
                PinholeCameraCfg(
                    sensor_tick=0.0,
                    resolution=tuple(cam.resolution),
                    data_types=list(cam.data_types),
                    usd_params=PinholeCameraCfg.UsdCameraCfg(
                        focal_length=float(cam.focal_length),
                        horizontal_aperture=float(cam.horizontal_aperture),
                        clipping_range=tuple(cam.clipping_range),
                    ),
                )
            )
            camera_path = f"{drone_prim.GetPath().pathString}/base_link/Camera_0"
            self.camera.spawn(
                prim_paths=[camera_path],
                translations=[list(cam.translation)],
                targets=[list(cam.target)],
            )
        return ["/World/defaultGroundPlane"]

    def _add_obstacles(self):
        """Add obstacles for exploration"""
        import omni.isaac.core.utils.prims as prim_utils
        from marinegym.robots.robot import ASSET_PATH

        # Add some simple cube obstacles
        for i in range(5):
            prim_utils.create_prim(
                f"/World/envs/env_0/obstacle_{i}",
                prim_type="Cube",
                translation=[i * 2.0 - 5.0, i * 1.5 - 3.0, 1.0],
                scale=[0.5, 0.5, 1.0],
                color=[0.8, 0.2, 0.2]
            )

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim + 3  # + goal position

        width, height = self.cfg.task.camera.resolution

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec(
                    (1, observation_dim), device=self.device
                ),
                "camera_rgb": UnboundedContinuousTensorSpec(
                    (1, 3, height, width),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "camera_depth": UnboundedContinuousTensorSpec(
                    (1, 1, height, width),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "slam_map": UnboundedContinuousTensorSpec(
                    (1, self.cfg.task.slam.n_channels,
                     self.cfg.task.slam.map_size[1],
                     self.cfg.task.slam.map_size[0]),
                    device=self.device,
                    dtype=torch.float32,
                ),
                "slam_pose": UnboundedContinuousTensorSpec(
                    (1, 3), device=self.device, dtype=torch.float32
                ),
                "exploration_goal": UnboundedContinuousTensorSpec(
                    (1, 2), device=self.device, dtype=torch.float32
                ),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device),
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "map_coverage": UnboundedContinuousTensorSpec(1),
            "frontier_distance": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        self.slam.reset(len(env_ids))
        self.prev_map_coverage = 0.0

        pos = torch.zeros(len(env_ids), 1, 3, device=self.device)
        pos[:, :, :2] = torch.randn(len(env_ids), 1, 2, device=self.device) * 2.0
        pos[:, :, 2] = 2.0
        rot = euler_to_quaternion(torch.zeros(len(env_ids), 1, 3, device=self.device))
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1),
            rot,
            env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)

        self.stats[env_ids] = 0.

    def _compute_state_and_obs(self):
        self.drone_state = self.drone.get_state()

        obs = [self.drone_state[..., 3:]]  # Start with drone state (exclude position)

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        obs = torch.cat(obs, dim=-1)

        # Get camera data
        images = self.camera.get_images()
        rgb = images["rgb"][:, :3].to(dtype=torch.float32).div_(255.0).unsqueeze(1)
        depth = images["distance_to_image_plane"].unsqueeze(1)

        # Process through SLAM
        slam_output = self.slam.process_frame(
            rgb,
            depth,
            self.drone_state.unsqueeze(1),
            self.drone.intrinsics.unsqueeze(1)
        )

        # Get exploration goal
        goal = self.exploration.get_exploration_goal(
            self.drone_state[..., :2]
        ).unsqueeze(0)

        # Add goal to observation
        obs = torch.cat([obs, goal.unsqueeze(1)], dim=-1)

        expected_rgb = (self.num_envs, 1, 3,
                       self.cfg.task.camera.resolution[1],
                       self.cfg.task.camera.resolution[0])
        expected_depth = (self.num_envs, 1, 1,
                         self.cfg.task.camera.resolution[1],
                         self.cfg.task.camera.resolution[0])

        if tuple(rgb.shape) != expected_rgb:
            raise RuntimeError(f"Unexpected RGB shape {tuple(rgb.shape)}; expected {expected_rgb}")
        if tuple(depth.shape) != expected_depth:
            raise RuntimeError(f"Unexpected depth shape {tuple(depth.shape)}; expected {expected_depth}")

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "camera_rgb": rgb,
                    "camera_depth": depth,
                    "slam_map": slam_output["slam_map"],
                    "slam_pose": slam_output["slam_pose"],
                    "exploration_goal": goal.unsqueeze(0),
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # Exploration reward
        exploration_reward, current_coverage = self.exploration.compute_exploration_reward(
            self.drone_state[..., :2],
            self.prev_map_coverage
        )
        self.prev_map_coverage = current_coverage

        # Effort reward
        reward_effort = self.reward_effort_weight * torch.exp(-torch.abs(self.drone.throttle))

        # Action smoothness
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(
            -self.drone.throttle_difference
        )

        # Collision penalty
        collision_penalty = 0.0
        if (self.drone.pos[..., 2] < 0.2).any():
            collision_penalty = -10.0

        reward = (
            exploration_reward * 5.0 +
            reward_effort +
            reward_action_smoothness +
            collision_penalty
        )

        # Check if drone is stuck
        stuck = torch.norm(self.drone.vel[..., :2], dim=-1) < 0.01
        misbehave = (self.drone.pos[..., 2] < 0.1) | stuck
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length - 1).unsqueeze(-1)

        # Update stats
        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["map_coverage"][:] = self.prev_map_coverage
        goal = self.exploration.get_exploration_goal(self.drone_state[..., :2])
        dist = torch.norm(goal - self.drone_state[..., :2], dim=-1)
        self.stats["frontier_distance"][:] = dist.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )