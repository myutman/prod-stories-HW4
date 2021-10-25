import os
import sys
import shutil
from gym import spaces

import pandas as pd

import wandb

import torch

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from PIL import Image

from tqdm import tqdm

sys.path.append(os.path.abspath("mapgen"))
os.environ["PYTHONPATH"] = os.path.abspath("mapgen")
from mapgen import Dungeon


class ModifiedDungeon(Dungeon):
    """Use this class to change the behavior of the original env (e.g. remove the trajectory from observation, like here)"""
    def __init__(self,
        width=20,
        height=20,
        max_rooms=3,
        min_room_xy=5,
        max_room_xy=12,
        max_steps: int = 2000
    ):
        observation_size = 11
        super().__init__(
            width=width,
            height=height,
            max_rooms=max_rooms,
            min_room_xy=min_room_xy,
            max_room_xy=max_room_xy,
            observation_size = 11,
            vision_radius = 5,
            max_steps = max_steps
        )

        self.positions = set()
        self.observation_space = spaces.Box(0, 1, [observation_size, observation_size, 4]) # because we remove trajectory and leave only cell types (UNK, FREE, OCCUPIED)
        self.action_space = spaces.Discrete(3)
        self.cum_reward = 0.0

    def step(self, action):
        observation, reward , done, info = super().step(action)

        new_explored = info["new_explored"]

        modified_reward = new_explored * (0.01 + self.cum_reward * 0.5)
        self.cum_reward += reward
        reward = modified_reward - 0.5

        current_position = (self._agent.position.x, self._agent.position.y, self._agent.orientation)
        if current_position in self.positions:
            reward -= 1.0
        self.positions.add(current_position)

        if observation[5, 5, 2] == 1:
            reward -= 1000.0

        logging_info = info.copy()
        logging_info["reward"] = reward

        return observation, reward , done, info

    def reset(self):
        observation = super().reset()
        return observation

if __name__ == "__main__":

    tune.register_env("Dungeon", lambda config: ModifiedDungeon(**config))


    CHECKPOINT_ROOT = "tmp/ppo/dungeon"
    shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)

    config = ppo.DEFAULT_CONFIG.copy()

    config["log_level"] = "INFO"
    config["framework"] = "torch"
    config["env"] = "Dungeon"
    config["env_config"] = {
        "width": 20,
        "height": 20,
        "max_rooms": 3,
        "min_room_xy": 5,
        "max_room_xy": 10,
    }


    config["model"] = {
        "conv_filters": [
            [16, (3, 3), 2],
            [32, (3, 3), 2],
            [32, (3, 3), 1],
        ],
        "post_fcnet_hiddens": [32],
        "post_fcnet_activation": "relu",
        "vf_share_layers": False,
    }


    config["rollout_fragment_length"] = 20
    config["entropy_coeff"] = 0.1
    config["lambda"] = 0.95
    config["vf_loss_coeff"] = 1.0
    config["vf_clip_param"] = 100.0
    config["num_gpus"] = 0

    config_table = pd.DataFrame()

    for key in config.keys():
        config_table[key] = [config[key]]

    logger = wandb.init(name='HW4-ppo-train')
    logger.log({"parameters": wandb.Table(data=config_table)})

    agent = ppo.PPOTrainer(config)

    N_ITER = 500
    s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

    #env = Dungeon(50, 50, 3)

    for n in tqdm(range(N_ITER)):
        result = agent.train()

        print(result["info"])

        file_name = agent.save(CHECKPOINT_ROOT)
        logger.save(file_name)

        metrics = ["entropy", "kl", "vf_loss", "policy_loss", "total_loss"]

        logging_info = {
            "episode_reward_min": result["episode_reward_min"],
            "episode_reward_mean": result["episode_reward_mean"],
            "episode_reward_max": result["episode_reward_max"],
            "episode_len_mean": result["episode_len_mean"],
        }

        for metric in metrics:
            logging_info[metric] = result["info"]["learner"]["default_policy"]["learner_stats"][metric]

        result.copy()
        logger.log(logging_info)

        # sample trajectory
        if (n+1)%5 == 0:
            env = ModifiedDungeon(20, 20, 3, min_room_xy=5, max_room_xy=10)
            obs = env.reset()

            frames = []

            for _ in range(500):
                action = agent.compute_single_action(obs)

                frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
                frames.append(frame)

                obs, reward, done, info = env.step(action)
                if done:
                    break

            frames[0].save(f"out_{n}.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000/60)
            logger.log({"example": wandb.Video(f"out_{n}.gif")})
