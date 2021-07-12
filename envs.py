from copy import deepcopy

import gym
import metaworld
import numpy as np
from gym.wrappers import TimeLimit

from utils.wrappers import ScaleReward, SuccessCounter, OneHotAdder, \
  RandomizationWrapper


def get_mt50():
  saved_random_state = np.random.get_state()
  np.random.seed(1)
  MT50 = metaworld.MT50()
  np.random.set_state(saved_random_state)
  return MT50


MT50 = get_mt50()
META_WORLD_TIME_HORIZON = 200
MT50_TASK_NAMES = list(MT50.train_classes)
MW_OBS_LEN = 12
MW_ACT_LEN = 4

def get_task_name(name_or_number):
  if type(name_or_number) is str:
    return name_or_number
  return MT50_TASK_NAMES[name_or_number]


def set_simple_goal(env, name):
  goal = [task for task in MT50.train_tasks if task.env_name == name][0]
  env.set_task(goal)


def get_subtasks(name):
  return [s for s in MT50.train_tasks if s.env_name == name]


def get_mt50_idx(env):
  idx = list(env._env_discrete_index.values())
  assert len(idx) == 1
  return idx[0]


def get_single_env(task, one_hot_idx=0, one_hot_len=1,
                   randomization='deterministic'):
  task_name = get_task_name(task)
  env = MT50.train_classes[task_name]()
  env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
  env = OneHotAdder(env, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len)
  # Currently TimeLimit is needed since SuccessCounter looks at dones.
  env = TimeLimit(env, META_WORLD_TIME_HORIZON)
  env = SuccessCounter(env)
  env.name = task_name
  env.num_envs = 1
  return env


def assert_equal_excluding_goal_dimensions(os1, os2):
  assert np.array_equal(os1.low[:9], os2.low[:9])
  assert np.array_equal(os1.high[:9], os2.high[:9])
  assert np.array_equal(os1.low[12:], os2.low[12:])
  assert np.array_equal(os1.high[12:], os2.high[12:])


def remove_goal_bounds(obs_space):
  obs_space.low[9:12] = -np.inf
  obs_space.high[9:12] = np.inf


class ContinualLearningEnv(gym.Env):
  def __init__(self, envs, steps_per_env):
    for i in range(len(envs)):
      assert envs[0].action_space == envs[i].action_space
      assert_equal_excluding_goal_dimensions(envs[0].observation_space,
                                             envs[i].observation_space)
    self.action_space = envs[0].action_space
    self.observation_space = deepcopy(envs[0].observation_space)
    remove_goal_bounds(self.observation_space)

    self.envs = envs
    self.num_envs = len(envs)
    self.steps_per_env = steps_per_env
    self.steps_limit = self.num_envs * self.steps_per_env
    self.cur_step = 0
    self.cur_seq_idx = 0

  def _check_steps_bound(self):
    if self.cur_step >= self.steps_limit:
      raise RuntimeError('Steps limit exceeded for ContinualLearningEnv!')

  def pop_successes(self):
    all_successes = []
    self.avg_env_success = {}
    for env in self.envs:
      successes = env.pop_successes()
      all_successes += successes
      if len(successes) > 0:
        self.avg_env_success[env.name] = np.mean(successes)
    return all_successes

  def step(self, action):
    self._check_steps_bound()
    obs, rew, done, info = self.envs[self.cur_seq_idx].step(action)
    info['seq_idx'] = self.cur_seq_idx

    self.cur_step += 1
    if self.cur_step % self.steps_per_env == 0:
      # If we hit limit for current env, end the episode.
      # This may cause border episodes to be shorter than 200.
      done = True
      info['TimeLimit.truncated'] = True

      self.cur_seq_idx += 1

    return obs, rew, done, info

  def reset(self):
    self._check_steps_bound()
    return self.envs[self.cur_seq_idx].reset()


def get_cl_env(tasks, steps_per_task, scale_reward=False, div_by_return=False,
               randomization='deterministic'):
  task_names = [get_task_name(task) for task in tasks]
  num_tasks = len(task_names)
  envs = []
  for i, task_name in enumerate(task_names):
    env = MT50.train_classes[task_name]()
    env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
    env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
    env.name = task_name
    env = TimeLimit(env, META_WORLD_TIME_HORIZON)
    env = SuccessCounter(env)
    if scale_reward:
      env = ScaleReward(env, div_by_return)
    envs.append(env)
  cl_env = ContinualLearningEnv(envs, steps_per_task)
  cl_env.name = 'ContinualLearningEnv'
  return cl_env


class MultiTaskEnv(gym.Env):
  def __init__(self, envs, steps_per_env, cycle_mode='episode'):
    # TODO: Implement the step cycle properly
    assert cycle_mode == 'episode'
    for i in range(len(envs)):
      assert envs[0].action_space == envs[i].action_space
      assert_equal_excluding_goal_dimensions(envs[0].observation_space,
                                             envs[i].observation_space)
    self.action_space = envs[0].action_space
    self.observation_space = deepcopy(envs[0].observation_space)
    remove_goal_bounds(self.observation_space)

    self.envs = envs
    self.num_envs = len(envs)
    self.steps_per_env = steps_per_env
    self.cycle_mode = cycle_mode

    self.steps_limit = self.num_envs * self.steps_per_env
    self.cur_step = 0
    self._cur_seq_idx = 0

  def _check_steps_bound(self):
    if self.cur_step >= self.steps_limit:
      raise RuntimeError('Steps limit exceeded for MultiTaskEnv!')

  def pop_successes(self):
    all_successes = []
    self.avg_env_success = {}
    for env in self.envs:
      successes = env.pop_successes()
      all_successes += successes
      if len(successes) > 0:
        self.avg_env_success[env.name] = np.mean(successes)
    return all_successes

  def step(self, action):
    self._check_steps_bound()
    obs, rew, done, info = self.envs[self._cur_seq_idx].step(action)
    info['mt_seq_idx'] = self._cur_seq_idx
    if self.cycle_mode == 'step':
      self._cur_seq_idx = (self._cur_seq_idx + 1) % self.num_envs
    self.cur_step += 1

    return obs, rew, done, info

  def reset(self):
    self._check_steps_bound()
    # TODO: step may be tricky to handle here
    if self.cycle_mode == 'episode':
      self._cur_seq_idx = (self._cur_seq_idx + 1) % self.num_envs
    obs = self.envs[self._cur_seq_idx].reset()
    return obs


def get_mt_env(tasks, steps_per_task, scale_reward=False, div_by_return=False,
               randomization='deterministic'):
  task_names = [get_task_name(task) for task in tasks]
  num_tasks = len(task_names)
  envs = []
  for i, task_name in enumerate(task_names):
    env = MT50.train_classes[task_name]()
    env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
    env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
    env.name = task_name
    env = TimeLimit(env, META_WORLD_TIME_HORIZON)
    env = SuccessCounter(env)
    if scale_reward:
      env = ScaleReward(env, div_by_return)
    envs.append(env)
  mt_env = MultiTaskEnv(envs, steps_per_task)
  mt_env.name = 'MultiTaskEnv'
  return mt_env
