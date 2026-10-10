# /workspace/github__bWitters__MarineGym/marinegym/envs/single/search_and_rescue.py
# MIT License
import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils

from marinegym.envs.isaac_env import AgentSpec, IsaacEnv
from marinegym.views import ArticulationView
from marinegym.utils.torch import euler_to_quaternion
from marinegym.sensors.camera import Camera
from marinegym.sensors.config import PinholeCameraCfg

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec

from marinegym.robots.drone import UnderwaterVehicle

class SearchAndRescue(IsaacEnv):
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.time_encoding = cfg.task.get("time_encoding", False)
        self.mode = cfg.mode

        super().__init__(cfg, headless)

        self.drone.initialize()
        if self.cfg.task.camera.enabled:
            self.camera = Camera(
                PinholeCameraCfg(
                    sensor_tick=0.0,
                    resolution=tuple(cfg.task.camera.resolution),
                    data_types=list(cfg.task.camera.data_types),
                    usd_params=PinholeCameraCfg.UsdCameraCfg(
                        focal_length=float(cfg.task.camera.focal_length),
                        horizontal_aperture=float(cfg.task.camera.horizontal_aperture),
                        clipping_range=tuple(cfg.task.camera.clipping_range),
                    ),
                )
            )
            self.camera.initialize(
                prim_paths_expr=(
                    f"/World/envs/env_.*/{self.drone.name}_0"
                    "base_link/Camera_0"
                )
            )
            if self.camera.count != self.num_envs:
                raise RuntimeError(
                    f"Expected {self.num_envs} cameras, found {self.camera.count}"
                )

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

        # Add obstacles
        self._add_obstacles()

        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 2.0)])[0]
        if self.cfg.task.camera.enabled:
            cam = self.cfg.task.camera
            camera_path = f"{drone_prim.GetPath().pathString}/base_link/Camera_0"
            self.camera.spawn(
                prim_paths=[camera_path],
                translations=[list(cam.translation)],
                targets=[list(cam.target)],
            )
        return ["/World/defaultGroundPlane"]

    def _add_obstacles(self):
        for i in range(5):
            prim_utils.create_prim(
                f"/World/envs/env_0/obstacle_{i}",
                prim_type="Cube",
                translation=[i * 2.0 - 5.0, i * 1.5 - 3.0, 1.0],
                scale=[0.5, 0.5, 1.0],
            )

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = drone_state_dim
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
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)

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

        obs = [self.drone_state[..., 3:]]  # Exclude position (x,y,z)

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        obs = torch.cat(obs, dim=-1)

        images = self.camera.get_images()
        rgb = images["rgb"][:, :3].to(dtype=torch.float32).div_(255.0).unsqueeze(1)
        depth = images["distance_to_image_plane"].unsqueeze(1)

        # Normalize depth to [0,1] based on clipping range
        near, far = self.cfg.task.camera.clipping_range
        depth = (depth * (far - near) + near) / far  # Normalize

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "camera_rgb": rgb,
                    "camera_depth": depth,
                    "intrinsics": self.drone.intrinsics,
                },
                "stats": self.stats.clone(),
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        reward_effort = self.reward_effort_weight * torch.exp(-torch.abs(self.drone.throttle))
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(
            -self.drone.throttle_difference
        )

        reward = reward_effort + reward_action_smoothness

        misbehave = (self.drone.pos[..., 2] < 0.1)
        hasnan = torch.isnan(self.drone_state).any(-1)

        terminated = misbehave | hasnan
        truncated = (self.progress_buf >= self.max_episode_length - 1).unsqueeze(-1)

        self.stats["return"] += reward
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

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