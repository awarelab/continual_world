import random

import gym
import numpy as np
from gym.spaces import Box

from task_lists import tasks_avg_return, task_reward_scales


class SuccessCounter(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)
    self.successes = []
    self.current_success = False

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if info.get('success', False):
      self.current_success = True
    if done:
      self.successes.append(self.current_success)
    return obs, reward, done, info

  def pop_successes(self):
    res = self.successes
    self.successes = []
    return res

  def reset(self, **kwargs):
    self.current_success = False
    return self.env.reset(**kwargs)


class ScaleReward(gym.Wrapper):
  def __init__(self, env, div_by_return=False):
    super().__init__(env)
    self.env = env
    if div_by_return:
      self.reward_scale = 1. / tasks_avg_return[env.name]
    else:
      self.reward_scale = task_reward_scales[env.name]

    print(env.name, self.reward_scale)

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    return obs, reward * self.reward_scale, done, info


class OneHotAdder(gym.Wrapper):
  def __init__(self, env, one_hot_idx, one_hot_len, orig_one_hot_dim=0):
    super().__init__(env)
    assert 0 <= one_hot_idx < one_hot_len
    self.to_append = np.zeros(one_hot_len)
    self.to_append[one_hot_idx] = 1.

    orig_obs_low = self.env.observation_space.low
    orig_obs_high = self.env.observation_space.high
    if orig_one_hot_dim > 0:
      orig_obs_low = orig_obs_low[:-orig_one_hot_dim]
      orig_obs_high = orig_obs_high[:-orig_one_hot_dim]
    self.observation_space = Box(
      np.concatenate([orig_obs_low, np.zeros(one_hot_len)]),
      np.concatenate([orig_obs_high, np.ones(one_hot_len)])
    )
    self.orig_one_hot_dim = orig_one_hot_dim

  def _append_one_hot(self, obs):
    if self.orig_one_hot_dim > 0:
      obs = obs[:-self.orig_one_hot_dim]
    return np.concatenate([obs, self.to_append])

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    return self._append_one_hot(obs), reward, done, info

  def reset(self, **kwargs):
    return self._append_one_hot(self.env.reset(**kwargs))


class RandomizationWrapper(gym.Wrapper):
  ALLOWED_KINDS = ['deterministic', 'random_init_all', 'random_init_fixed20',
                   'random_init_small_box']

  def __init__(self, env, subtasks, kind):
    assert kind in RandomizationWrapper.ALLOWED_KINDS
    super().__init__(env)
    self.subtasks = subtasks
    self.kind = kind

    env.set_task(subtasks[0])
    if kind == 'random_init_all':
      env._freeze_rand_vec = False

    if kind == 'random_init_fixed20':
      assert len(subtasks) >= 20

    if kind == 'random_init_small_box':
      diff = env._random_reset_space.high - env._random_reset_space.low
      self.reset_space_low = env._random_reset_space.low + 0.45 * diff
      self.reset_space_high = env._random_reset_space.low + 0.55 * diff

  def reset(self, **kwargs):
    if self.kind == 'random_init_fixed20':
      self.env.set_task(self.subtasks[random.randint(0, 19)])
    elif self.kind == 'random_init_small_box':
      rand_vec = np.random.uniform(
        self.reset_space_low,
        self.reset_space_high,
        size=self.reset_space_low.size)
      self.env._last_rand_vec = rand_vec

    return self.env.reset(**kwargs)
