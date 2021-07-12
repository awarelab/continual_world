import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class RegularizationHelper:
  def __init__(self, actor, critic1, critic2, regularize_critic=False):
    self.regularize_critic = regularize_critic
    self.actor = actor
    self.critic1 = critic1
    self.critic2 = critic2

    self.actor_variables = actor.common_variables
    self.critic_variables = critic1.common_variables + critic2.common_variables
    self.all_variables = self.actor_variables + self.critic_variables

    self.reg_weights = list(
        tf.Variable(tf.zeros_like(param), trainable=False)
        for param in self.all_variables)

  def merge_weights(self, new_weights):
    if self.reg_weights is not None:
      merged_weights = list(old_reg + new_reg for old_reg, new_reg
                              in zip(self.reg_weights, new_weights))
    else:
      merged_weights = new_weights

    for old_weight, new_weight in zip(self.reg_weights, merged_weights):
      old_weight.assign(new_weight)

  def update_reg_weights(self, replay_buffer, batches_num=10, batch_size=256):
    all_weights = []
    for batch_idx in range(batches_num):
      batch = replay_buffer.sample_batch(batch_size)
      all_weights += [self.get_batch(**batch)]

    mean_weights = []
    for weights in zip(*all_weights):
      mean_weights += [tf.reduce_mean(tf.stack(weights, 0), 0)]

    self.merge_weights(mean_weights)

  def regularize(self, old_params):
    reg_loss = tf.zeros([])
    for new_param, old_param, weight in zip(self.all_variables, old_params, self.reg_weights):
      diffs = (new_param - old_param) ** 2
      weighted_diffs = weight * diffs
      reg_loss += tf.reduce_sum(weighted_diffs)
    return reg_loss

class L2Helper(RegularizationHelper):
  def __init__(self, actor, critic1, critic2, regularize_critic=False):
    super().__init__(actor, critic1, critic2, regularize_critic)

  def update_reg_weights(self, replay_buffer, batches_num=10, batch_size=256):
    if self.regularize_critic:
      new_weights = list(tf.ones_like(param) for param in self.all_variables)
    else:
      new_weights = (
          list(tf.ones_like(param) for param in self.actor_variables)
          + list(tf.zeros_like(param) for param in self.critic_variables))

    self.merge_weights(new_weights)


class EWCHelper(RegularizationHelper):
  def __init__(self, actor, critic1, critic2, regularize_critic=False, critic_reg_coef=1.):
    super().__init__(actor, critic1, critic2, regularize_critic)
    self.critic_reg_coef = critic_reg_coef

  @tf.function
  def get_grads(self, obs1, obs2, acts, rews, done):
    with tf.GradientTape(persistent=True) as g:
      # Main outputs from computation graph
      mu, log_std, pi, logp_pi = self.actor(obs1)
      std = tf.exp(log_std)

      q1 = self.critic1(obs1, acts)
      q2 = self.critic2(obs1, acts)

    # Compute diagonal of the Fisher matrix
    actor_mu_gs = g.jacobian(mu, self.actor.common_variables)
    actor_std_gs = g.jacobian(std, self.actor.common_variables)
    q1_gs = g.jacobian(q1, self.critic1.common_variables)
    q2_gs = g.jacobian(q2, self.critic2.common_variables)
    del g
    return actor_mu_gs, actor_std_gs, q1_gs, q2_gs, std

  def get_batch(self, **batch):
    actor_mu_gs, actor_std_gs, q1_gs, q2_gs, std = self.get_grads(**batch)

    reg_weights = []
    for mu_g, std_g in zip(actor_mu_gs, actor_std_gs):
      if mu_g is None and std_g is None:
        raise ValueError('Both mu and std gradients are None!')
      if mu_g is None:
        mu_g = tf.zeros_like(std_g)
      if std_g is None:
        std_g = tf.zeros_like(mu_g)

      # Broadcasting std for every parameter in the model
      dims_to_add = int(tf.rank(mu_g) - tf.rank(std))
      broad_shape = std.shape + [1] * dims_to_add
      broad_std = tf.reshape(std, broad_shape)  # broadcasting

      # Fisher information, see the derivation
      fisher = 1 / (broad_std ** 2 + 1e-6) * (mu_g ** 2 + 2 * std_g ** 2)

      # Sum over the output dimensions
      fisher = tf.reduce_sum(fisher, 1)

      # Clip from below
      fisher = tf.clip_by_value(fisher, 1e-5, np.inf)

      # Average over the examples in the batch
      reg_weights += [tf.reduce_mean(fisher, 0)]

    critic_coef = self.critic_reg_coef if self.regularize_critic else 0.
    for q_g in q1_gs:
      fisher = q_g ** 2
      reg_weights += [critic_coef * tf.reduce_mean(fisher, 0)]

    for q_g in q2_gs:
      fisher = q_g ** 2
      reg_weights += [critic_coef * tf.reduce_mean(fisher, 0)]

    return reg_weights

class MASHelper(RegularizationHelper):
  def __init__(self, actor, critic1, critic2, regularize_critic=False):
    super().__init__(actor, critic1, critic2, regularize_critic)

  @tf.function
  def get_grads(self, obs1, obs2, acts, rews, done):
    with tf.GradientTape(persistent=True) as g:
      # Main outputs from computation graph
      mu, log_std, pi, logp_pi = self.actor(obs1)

      # Get squared L2 norm per-example
      actor_norm = tf.reduce_sum(mu ** 2, -1) + tf.reduce_sum(log_std ** 2, -1)

      q1 = self.critic1(obs1, acts)
      critic1_norm = q1 ** 2

      q2 = self.critic2(obs1, acts)
      critic2_norm = q2 ** 2

    # Compute gradients for MAS
    actor_gs = g.jacobian(actor_norm, self.actor_variables)
    q1_gs = g.jacobian(critic1_norm, self.critic1.common_variables)
    q2_gs = g.jacobian(critic2_norm, self.critic2.common_variables)
    del g
    return actor_gs, q1_gs, q2_gs

  def get_batch(self, **batch):
    actor_gs, q1_gs, q2_gs = self.get_grads(**batch)

    reg_weights = []
    for g in actor_gs:
      reg_weights += [tf.reduce_mean(tf.abs(g), 0)]

    critic_coef = self.critic_reg_coef if self.regularize_critic else 0.
    for g in q1_gs:
      reg_weights += [critic_coef * tf.reduce_mean(tf.abs(g), 0)]

    for g in q2_gs:
      reg_weights += [critic_coef * tf.reduce_mean(tf.abs(g), 0)]

    return reg_weights
