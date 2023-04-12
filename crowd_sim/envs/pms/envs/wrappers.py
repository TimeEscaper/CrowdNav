import gym
import numpy as np

from typing import Optional, Union, Dict
from stable_baselines3.common.vec_env import SubprocVecEnv

from crowd_sim.envs.pms.envs.curriculum import AbstractCurriculum
from crowd_sim.envs.pms.utils.loggers import AbstractLogger


class EvalEnvWrapper(gym.Env):

    def __init__(self,
                 env: gym.Env,
                 curriculum: AbstractCurriculum,
                 n_eval_episodes: int,
                 logger: AbstractLogger,
                 train_env: Optional[SubprocVecEnv] = None):
        self._env = env
        self._curriculum = curriculum
        self._n_eval_episodes = n_eval_episodes
        self._logger = logger
        self._train_env = train_env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self._current_episode = 0
        self._current_cumulative_reward = 0.
        self._current_cumulative_reward_components = {}

        self._cumulative_rewards_histories = []
        self._cumulative_rewards_components_histories = {}
        self._success_histories = []

    def step(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)

        self._current_cumulative_reward += reward
        if "reward" in info:
            for k, v in info["reward"].items():
                if k in self._current_cumulative_reward_components:
                    self._current_cumulative_reward_components[k] += v
                else:
                    self._current_cumulative_reward_components[k] = v

        if done:
            success = info["done_reason"] == "success"
            self._success_histories.append(success)
            self._cumulative_rewards_histories.append(self._current_cumulative_reward)

            for component, cumulative_reward in self._current_cumulative_reward_components.items():
                if component in self._cumulative_rewards_components_histories:
                    self._cumulative_rewards_components_histories[component].append(cumulative_reward)
                else:
                    self._cumulative_rewards_components_histories[component] = [cumulative_reward]
            self._current_episode += 1

            if self._current_episode == self._n_eval_episodes:
                mean_cumulative_reward = np.mean(self._cumulative_rewards_histories)
                success_ratio = np.array(self._success_histories).sum() / len(self._success_histories)
                self._logger.log("eval/mean_cumulative_reward", mean_cumulative_reward)
                self._logger.log("eval/success_ratio", success_ratio)
                for component, rewards in self._cumulative_rewards_components_histories.items():
                    self._logger.log(f"eval/mean_reward_components/{component}", np.mean(rewards))

                threshold = self._curriculum.get_success_rate_threshold()
                if threshold is not None:
                    if success_ratio >= threshold:
                        self._curriculum.update_stage()
                        if self._train_env is not None and isinstance(self._train_env, SubprocVecEnv):
                            self._train_env.env_method("update_curriculum")
                current_stage, _ = self._curriculum.get_current_stage()
                self._logger.log("stage_idx", current_stage)

                self._current_episode = 0
                self._success_histories = []
                self._cumulative_rewards_histories = []
                self._cumulative_rewards_components_histories = {}

            self._current_cumulative_reward = 0
            self._current_cumulative_reward_components = {}

        return obs, reward, done, info

    def reset(self):
        return self._env.reset()

    def render(self, mode="human"):
        self._env.render(mode)


class StackHistoryWrapper(gym.Env):

    def __init__(self,
                 env: gym.Env,
                 n_stacks: Union[int, Dict[str, int]]):
        original_space = env.observation_space
        assert isinstance(original_space, gym.spaces.Dict), \
            f"Only dict observation space is supported in StackHistoryWrapper"

        if not isinstance(n_stacks, dict):
            n_stacks = {k: n_stacks for k in original_space.keys()}

        observation_space = {}
        for k, v in original_space.items():
            if k in n_stacks:
                assert isinstance(v, gym.spaces.Box), \
                    f"Only box observation subspaces can be stacked in StackHistoryWrapper"
                if isinstance(v.low, np.ndarray):
                    new_low = np.stack([v.low for _ in range(n_stacks[k])], axis=0)
                    new_high = np.stack([v.high for _ in range(n_stacks[k])], axis=0)
                else:
                    new_low = v.low
                    new_high = v.high
                observation_space[k] = gym.spaces.Box(
                    low=new_low,
                    high=new_high,
                    shape=(n_stacks[k],) + v.shape,
                    dtype=v.dtype
                )
            else:
                observation_space[k] = v

        self.observation_space = gym.spaces.Dict(observation_space)
        self.action_space = env.action_space
        self._env = env
        self._n_stacks = n_stacks

        self._obs_histories = {}

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        self._obs_histories = {k: np.concatenate([v[1:], obs[k][np.newaxis]], axis=0)
                               for k, v in self._obs_histories.items()}
        obs.update({k: v.copy() for k, v in self._obs_histories.items()})

        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()

        self._obs_histories = {k: np.stack([obs[k] for _ in range(v)], axis=0) for k, v in self._n_stacks.items()}
        obs.update({k: v.copy() for k, v in self._obs_histories.items()})

        return obs

    def render(self, mode="human"):
        return self._env.render(mode)

    def update_curriculum(self):
        self._env.update_curriculum()

    def enable_render(self):
        self._env.enable_render()
