import hydra
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from marinegym import init_simulation_app
from tensordict import TensorDict

from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

@hydra.main(version_base=None, config_path=".", config_name="test_rgbd")
def main(cfg):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)  # Allow adding new keys

    simulation_app = init_simulation_app(cfg)

    from marinegym.envs.isaac_env import IsaacEnv

    # Load env
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=False)
    env = TransformedEnv(base_env, Compose(InitTracker())).eval()

    # Run one episode
    depth_frames = []
    rgb_frames = []
    depth_stats = []

    obs = env.reset()

    for i in range(200):
        # Get observations
        if i == 0:
            obs = env.reset()
        else:
            # Send random action
            action = base_env.action_spec[("agents", "action")].rand()
            action_td = TensorDict({
                ("agents", "action"): action
            }, batch_size=base_env.batch_size)
            obs = env.step(action_td)

        # Extract depth and RGB
        try:
            depth = obs[("agents", "camera_depth")].cpu().numpy()
            rgb = obs[("agents", "camera_rgb")].cpu().numpy()

            # Save for video
            rgb_frames.append((rgb[0,0].transpose(1,2,0) * 255).astype(np.uint8))
            depth_colormap = cv2.applyColorMap(
                (depth[0,0,0] * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            depth_frames.append(depth_colormap)

            # Collect stats
            depth_stats.append([depth.mean(), depth.min(), depth.max()])
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            print(f"Available keys: {obs.keys(True, True)}")
            break

        simulation_app.update()

    if len(rgb_frames) > 0:
        # Save video with depth overlay
        height, width = rgb_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('rgbd_test.mp4', fourcc, 30, (width*2, height))
        for rgb_frame, depth_frame in zip(rgb_frames, depth_frames):
            combined = np.hstack([rgb_frame, depth_frame])
            out.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        out.release()

        # Plot depth stats over time
        stats = np.array(depth_stats)
        plt.figure(figsize=(12, 4))
        plt.plot(stats[:,0], label='Mean Depth')
        plt.plot(stats[:,1], label='Min Depth')
        plt.plot(stats[:,2], label='Max Depth')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Depth')
        plt.legend()
        plt.savefig('depth_stats.png')
        plt.close()

        print("Saved rgbd_test.mp4 and depth_stats.png")

    simulation_app.close()

if __name__ == "__main__":
    main()