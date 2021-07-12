import random
import numpy as np
import tensorflow as tf


class ReplayBuffer:
  """
  A simple FIFO experience replay buffer for SAC agents.
  """

  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def sample_batch(self, batch_size):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(obs1=tf.convert_to_tensor(self.obs1_buf[idxs]),
                obs2=tf.convert_to_tensor(self.obs2_buf[idxs]),
                acts=tf.convert_to_tensor(self.acts_buf[idxs]),
                rews=tf.convert_to_tensor(self.rews_buf[idxs]),
                done=tf.convert_to_tensor(self.done_buf[idxs]))


class EpisodicMemory:
  """
  Buffer which does not support overwriting old samples
  """

  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.size, self.max_size = 0, size

  def store_multiple(self, obs1, acts, rews, obs2, done):
    assert len(obs1) == len(acts) == len(rews) == len(obs2) == len(done)
    assert self.size + len(obs1) <= self.max_size

    range_start = self.size
    range_end = self.size + len(obs1)
    self.obs1_buf[range_start:range_end] = obs1
    self.obs2_buf[range_start:range_end] = obs2
    self.acts_buf[range_start:range_end] = acts
    self.rews_buf[range_start:range_end] = rews
    self.done_buf[range_start:range_end] = done
    self.size = self.size + len(obs1)

  def sample_batch(self, batch_size):
    batch_size = min(batch_size, self.size)
    idxs = np.random.choice(self.size, size=batch_size, replace=False)
    return dict(obs1=tf.convert_to_tensor(self.obs1_buf[idxs]),
                obs2=tf.convert_to_tensor(self.obs2_buf[idxs]),
                acts=tf.convert_to_tensor(self.acts_buf[idxs]),
                rews=tf.convert_to_tensor(self.rews_buf[idxs]),
                done=tf.convert_to_tensor(self.done_buf[idxs]))


class ReservoirReplayBuffer(ReplayBuffer):
  """
  Buffer for SAC agents implementing reservoir sampling.
  """

  def __init__(self, obs_dim, act_dim, size):
    super().__init__(obs_dim, act_dim, size)
    self.timestep = 0

  def store(self, obs, act, rew, next_obs, done):
    current_t = self.timestep
    self.timestep += 1

    if current_t < self.max_size:
      buffer_idx = current_t
    else:
      buffer_idx = random.randint(0, current_t)
      if buffer_idx >= self.max_size:
        return

    self.obs1_buf[buffer_idx] = obs
    self.obs2_buf[buffer_idx] = next_obs
    self.acts_buf[buffer_idx] = act
    self.rews_buf[buffer_idx] = rew
    self.done_buf[buffer_idx] = done
    self.size = min(self.size + 1, self.max_size)
