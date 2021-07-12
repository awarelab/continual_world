import math
import os
import random
import time

import numpy as np
import tensorflow as tf

from methods.agem import AgemHelper
from methods.packnet import PackNetHelper
from methods.regularization import L2Helper, EWCHelper, MASHelper
from methods.vcl import VclHelper
from spinup import models
from spinup.models import PopArtMlpCritic
from spinup.replay_buffers import EpisodicMemory, ReplayBuffer, ReservoirReplayBuffer
from utils.utils import reset_optimizer, reset_weights


def sac(env, test_envs, logger, actor_cl=models.MlpActor, actor_kwargs=None,
        critic_cl=models.MlpCritic, critic_kwargs=None, seed=0,
        steps=2000000, log_every=20000, replay_size=1000000, gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=10000,
        update_after=1000, update_every=50, num_test_eps_stochastic=10,
        num_test_eps_deterministic=1, max_ep_len=200, save_freq_epochs=100,
        reset_buffer_on_task_change=True, buffer_type='fifo',
        reset_optimizer_on_task_change=False, cl_method=None, cl_reg_coef=0.01,
        packnet_retrain_steps=0, regularize_critic=False, vcl_first_task_kl=True,
        episodic_mem_per_task=0, episodic_batch_size=0,
        reset_critic_on_task_change=False, clipnorm=None,
        target_output_std=None, packnet_fake_num_tasks=None,
        agent_policy_exploration=False, critic_reg_coef=1.):
  """
  Non-obvious args:
      polyak (float): Interpolation factor in polyak averaging for target
          networks. Target networks are updated towards main networks
          according to:

          .. math:: \\theta_{\\text{targ}} \\leftarrow
              \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

          where :math:`\\rho` is polyak. (Always between 0 and 1, usually
          close to 1.)

      alpha (float): Entropy regularization coefficient. (Equivalent to
          inverse of reward scale in the original SAC paper.)

      batch_size (int): Minibatch size for SGD.

      start_steps (int): Number of steps for uniform-random action selection,
          before running real policy. Helps exploration.

      update_after (int): Number of env interactions to collect before
          starting to do gradient descent updates. Ensures replay buffer
          is full enough for useful updates.

      update_every (int): Number of env interactions that should elapse
          between gradient descent updates. Note: Regardless of how long
          you wait between updates, the ratio of env steps to gradient steps
          is locked to 1.

      num_test_episodes (int): Number of episodes to test the deterministic
          policy at the end of each epoch.

      max_ep_len (int): Maximum length of trajectory / episode / rollout.

      save_freq_epochs (int): How often (in terms of gap between epochs) to save
          the current policy and value function.
  """

  random.seed(seed)
  tf.random.set_seed(seed)
  np.random.seed(seed)
  env.action_space.seed(seed)

  obs_dim = env.observation_space.shape[0]
  act_dim = env.action_space.shape[0]
  # This implementation assumes all dimensions share the same bound!
  assert np.all(env.action_space.high == env.action_space.high[0])

  num_tasks = env.num_envs

  if actor_kwargs is None:
    actor_kwargs = {}
  if critic_kwargs is None:
    critic_kwargs = {}

  # Share information about action space with policy architecture
  actor_kwargs['action_space'] = env.action_space
  actor_kwargs['input_dim'] = obs_dim
  critic_kwargs['input_dim'] = obs_dim + act_dim

  # Create experience buffer
  if buffer_type == 'fifo':
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                 size=replay_size)
  elif buffer_type == 'reservoir':
    replay_buffer = ReservoirReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                          size=replay_size)

  # Create actor and critic networks
  actor = actor_cl(**actor_kwargs)

  critic1 = critic_cl(**critic_kwargs)
  target_critic1 = critic_cl(**critic_kwargs)
  target_critic1.set_weights(critic1.get_weights())

  critic2 = critic_cl(**critic_kwargs)
  target_critic2 = critic_cl(**critic_kwargs)
  target_critic2.set_weights(critic2.get_weights())

  critic_variables = critic1.trainable_variables + critic2.trainable_variables
  all_variables = actor.trainable_variables + critic_variables
  all_common_variables = (
            actor.common_variables
            + critic1.common_variables
            + critic2.common_variables)

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

  # Setup CL methods
  weights_reg_methods = ['l2', 'ewc', 'mas']
  exp_replay_methods = ['agem']
  if cl_method == 'packnet':
    packnet_models = [actor]
    if regularize_critic:
      packnet_models.extend([critic1, critic2])
    packnet_helper = PackNetHelper(packnet_models)
  elif cl_method in weights_reg_methods:
    old_params = list(
      tf.Variable(tf.identity(param), trainable=False)
      for param in all_common_variables)

    if cl_method == 'l2':
      reg_helper = L2Helper(actor, critic1, critic2, regularize_critic)
    elif cl_method == 'ewc':
      reg_helper = EWCHelper(actor, critic1, critic2, regularize_critic, critic_reg_coef)
    elif cl_method == 'mas':
      reg_helper = MASHelper(actor, critic1, critic2, regularize_critic)

  elif cl_method == 'vcl':
    vcl_helper = VclHelper(actor, critic1, critic2, regularize_critic)
  elif cl_method in exp_replay_methods:
    episodic_mem_size = episodic_mem_per_task * env.num_envs
    episodic_memory = EpisodicMemory(obs_dim=obs_dim, act_dim=act_dim,
                                     size=episodic_mem_size)
    if cl_method == 'agem':
      agem_helper = AgemHelper()


  # For reference on automatic alpha tuning, see
  # "Automating Entropy Adjustment for Maximum Entropy" section
  # in https://arxiv.org/abs/1812.05905
  auto_alpha = False
  if alpha == 'auto':
    auto_alpha = True
    all_log_alpha = tf.Variable(np.ones((num_tasks, 1), dtype=np.float32),
                                trainable=True)
    if target_output_std is None:
      target_entropy = -np.prod(env.action_space.shape).astype(np.float32)
    else:
      target_1d_entropy = np.log(target_output_std *
                                 math.sqrt(2 * math.pi * math.e))
      target_entropy = (np.prod(env.action_space.shape).astype(np.float32) *
                        target_1d_entropy)

  @tf.function
  def get_log_alpha(obs1):
    return tf.squeeze(tf.linalg.matmul(obs1[:, -num_tasks:], all_log_alpha))


  @tf.function
  def get_action(o, deterministic=tf.constant(False)):
    mu, log_std, pi, logp_pi = actor(tf.expand_dims(o, 0))
    if deterministic:
      return mu[0]
    else:
      return pi[0]

  @tf.function
  def vcl_get_stable_action(o, deterministic=tf.constant(False)):
    mu, log_std, pi, logp_pi = actor(tf.expand_dims(o, 0), samples_num=10)
    if deterministic:
      return mu[0]
    else:
      return pi[0]

  def get_learn_on_batch():
    @tf.function
    def learn_on_batch(seq_idx, batch, episodic_batch=None):
      gradients, metrics = get_gradients(seq_idx, **batch)

      if cl_method == 'packnet':
        actor_gradients, critic_gradients, alpha_gradient = gradients
        actor_gradients = packnet_helper.adjust_gradients(
          actor_gradients, actor.trainable_variables,
          tf.convert_to_tensor(seq_idx))
        if regularize_critic:
          critic_gradients = packnet_helper.adjust_gradients(
            critic_gradients, critic_variables, tf.convert_to_tensor(seq_idx))
        gradients = (actor_gradients, critic_gradients, alpha_gradient)
      # Warning: we refer here to the int task_idx in the parent function, not
      # the passed seq_idx.
      elif cl_method == 'agem' and current_task_idx > 0:
        ref_gradients, _ = get_gradients(seq_idx, **episodic_batch)
        gradients, violation = agem_helper.adjust_gradients(gradients, ref_gradients)
        metrics['agem_violation'] = violation

      if clipnorm is not None:
        actor_gradients, critic_gradients, alpha_gradient = gradients
        gradients = (
          tf.clip_by_global_norm(actor_gradients, clipnorm)[0],
          tf.clip_by_global_norm(critic_gradients, clipnorm)[0],
          tf.clip_by_norm(alpha_gradient, clipnorm)
        )

      apply_update(*gradients)
      return metrics

    return learn_on_batch

  def get_gradients(seq_idx, obs1, obs2, acts, rews, done):
    with tf.GradientTape(persistent=True) as g:
      if auto_alpha:
        log_alpha = get_log_alpha(obs1)
      else:
        log_alpha = tf.math.log(alpha)

      # Main outputs from computation graph
      mu, log_std, pi, logp_pi = actor(obs1)
      q1 = critic1(obs1, acts)
      q2 = critic2(obs1, acts)

      # compose q with pi, for pi-learning
      q1_pi = critic1(obs1, pi)
      q2_pi = critic2(obs1, pi)

      # get actions and log probs of actions for next states, for Q-learning
      _, _, pi_next, logp_pi_next = actor(obs2)

      # target q values, using actions from *current* policy
      target_q1 = target_critic1(obs2, pi_next)
      target_q2 = target_critic2(obs2, pi_next)

      # Min Double-Q:
      min_q_pi = tf.minimum(q1_pi, q2_pi)
      min_target_q = tf.minimum(target_q1, target_q2)

      # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
      if critic_cl is PopArtMlpCritic:
        q_backup = tf.stop_gradient(
          critic1.normalize(
            rews + gamma * (1 - done) * (
              critic1.unnormalize(min_target_q, obs2)
              - tf.math.exp(log_alpha) * logp_pi_next), obs1))
      else:
        q_backup = tf.stop_gradient(rews + gamma * (1 - done) * (
          min_target_q - tf.math.exp(log_alpha) * logp_pi_next))

      # Soft actor-critic losses
      pi_loss = tf.reduce_mean(
        tf.math.exp(log_alpha) * logp_pi - min_q_pi)
      q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
      q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
      value_loss = q1_loss + q2_loss

      if auto_alpha:
        alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(
          logp_pi + target_entropy))

      if cl_method in weights_reg_methods:
        reg_loss = reg_helper.regularize(old_params)
        reg_loss_coef = tf.cond(seq_idx > 0, lambda: cl_reg_coef, lambda: 0.)
        reg_loss *= reg_loss_coef

        pi_loss += reg_loss
        value_loss += reg_loss
      elif cl_method == 'vcl':
        reg_loss = vcl_helper.regularize(seq_idx, regularize_last_layer=vcl_first_task_kl)
        reg_loss_coef = tf.cond(
                seq_idx > 0 or vcl_first_task_kl,
                lambda: cl_reg_coef,
                lambda: 0.)
        reg_loss *= reg_loss_coef

        pi_loss += reg_loss
      else:
        reg_loss = 0.

    # Compute gradients
    actor_gradients = g.gradient(pi_loss, actor.trainable_variables)
    critic_gradients = g.gradient(value_loss, critic_variables)
    if auto_alpha:
      alpha_gradient = g.gradient(alpha_loss, all_log_alpha)
    else:
      alpha_gradient = None
    del g

    if critic_cl is PopArtMlpCritic:
      # Stats are shared between critic1 and critic2.
      # We keep them only in critic1.
      critic1.update_stats(q_backup, obs1)

    gradients = (actor_gradients, critic_gradients, alpha_gradient)
    metrics = dict(pi_loss=pi_loss, q1_loss=q1_loss, q2_loss=q2_loss, q1=q1, q2=q2,
                logp_pi=logp_pi, reg_loss=reg_loss, agem_violation=0)
    return gradients, metrics


  def apply_update(actor_gradients, critic_gradients, alpha_gradient):
    optimizer.apply_gradients(
      zip(actor_gradients, actor.trainable_variables))

    optimizer.apply_gradients(
      zip(critic_gradients, critic_variables))

    if auto_alpha:
      optimizer.apply_gradients([(alpha_gradient, all_log_alpha)])

    # Polyak averaging for target variables
    for v, target_v in zip(critic1.trainable_variables,
                           target_critic1.trainable_variables):
      target_v.assign(polyak * target_v + (1 - polyak) * v)
    for v, target_v in zip(critic2.trainable_variables,
                           target_critic2.trainable_variables):
      target_v.assign(polyak * target_v + (1 - polyak) * v)


  def test_agent():
    # TODO: parallelize test phase if we hit significant added walltime.
    for deterministic, num_eps in [(False, num_test_eps_stochastic),
                                   (True, num_test_eps_deterministic)]:
      avg_success = []
      mode = 'deterministic' if deterministic else 'stochastic'
      for seq_idx, test_env in enumerate(test_envs):
        key_prefix = 'test/{}/{}/{}/'.format(mode, seq_idx, test_env.name)

        if cl_method == 'packnet':
          packnet_helper.set_view(seq_idx)

        for j in range(num_eps):
          o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
          while not (d or (ep_len == max_ep_len)):
            if cl_method == 'vcl':
              # Disable multiple samples in VCL for faster evaluation
              action_fn = get_action # = vcl_get_stable_action
            else:
              action_fn = get_action
            o, r, d, _ = test_env.step(
              action_fn(tf.convert_to_tensor(o), tf.constant(deterministic)))
            ep_ret += r
            ep_len += 1
          logger.store({
            key_prefix + 'return': ep_ret,
            key_prefix + 'ep_length': ep_len
          })

        if cl_method == 'packnet':
          packnet_helper.set_view(-1)

        logger.log_tabular(key_prefix + 'return', with_min_and_max=True)
        logger.log_tabular(key_prefix + 'ep_length', average_only=True)
        env_success = test_env.pop_successes()
        avg_success += env_success
        logger.log_tabular(key_prefix + 'success', np.mean(env_success))
      key = 'test/{}/average_success'.format(mode)
      logger.log_tabular(key, np.mean(avg_success))

  start_time = time.time()
  o, ep_ret, ep_len = env.reset(), 0, 0

  # Main loop: collect experience in env and update/log each epoch
  learn_on_batch = get_learn_on_batch()
  current_task_t = 0
  current_task_idx = -1
  reg_weights = None

  for t in range(steps):
    # On task change
    if current_task_idx != getattr(env, 'cur_seq_idx', -1):
      current_task_idx = getattr(env, 'cur_seq_idx')
      current_task_t = 0
      if cl_method in weights_reg_methods and current_task_idx > 0:
        for old_param, new_param in zip(old_params, all_common_variables):
          old_param.assign(new_param)
        reg_helper.update_reg_weights(replay_buffer)

      elif cl_method in exp_replay_methods and current_task_idx > 0:
        new_episodic_mem = replay_buffer.sample_batch(episodic_mem_per_task)
        episodic_memory.store_multiple(**new_episodic_mem)
      elif cl_method == 'vcl' and current_task_idx > 0:
        vcl_helper.update_prior()

      if reset_buffer_on_task_change:
        assert buffer_type == 'fifo'
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                     size=replay_size)
      if reset_critic_on_task_change:
        reset_weights(critic1, critic_cl, critic_kwargs)
        target_critic1.set_weights(critic1.get_weights())
        reset_weights(critic2, critic_cl, critic_kwargs)
        target_critic2.set_weights(critic2.get_weights())

      if reset_optimizer_on_task_change:
        reset_optimizer(optimizer)

      # Update variables list and update function in case model changed.
      # E.g: For VCL after the first task we set trainable=False for layer
      # normalization. We need to recompute the graph in order for TensorFlow
      # to notice this change.
      learn_on_batch = get_learn_on_batch()
      all_variables = actor.trainable_variables + critic_variables
      all_common_variables = (
                actor.common_variables
                + critic1.common_variables
                + critic2.common_variables)


    # Until start_steps have elapsed, randomly sample actions
    # from a uniform distribution for better exploration. Afterwards,
    # use the learned policy.
    if current_task_t > start_steps or (agent_policy_exploration and current_task_idx > 0):
      a = get_action(tf.convert_to_tensor(o))
    else:
      a = env.action_space.sample()

    # Step the env
    o2, r, d, info = env.step(a)
    ep_ret += r
    ep_len += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    d_to_store = d
    if ep_len == max_ep_len or info.get('TimeLimit.truncated'):
      d_to_store = False

    # Store experience to replay buffer
    replay_buffer.store(o, a, r, o2, d_to_store)

    # Super critical, easy to overlook step: make sure to update
    # most recent observation!
    o = o2

    # End of trajectory handling
    if d or (ep_len == max_ep_len):
      logger.store({
        'train/return': ep_ret,
        'train/ep_length': ep_len
      })
      ep_ret, ep_len = 0, 0
      if t < steps - 1:
        o = env.reset()

    # Update handling
    if current_task_t >= update_after and current_task_t % update_every == 0:
      for j in range(update_every):
        batch = replay_buffer.sample_batch(batch_size)
        if cl_method in exp_replay_methods and current_task_idx > 0:
          episodic_batch = episodic_memory.sample_batch(episodic_batch_size)
        else:
          episodic_batch = None
        results = learn_on_batch(
            tf.convert_to_tensor(current_task_idx), batch, episodic_batch)
        logger.store({
          'train/q1_vals': results['q1'],
          'train/q2_vals': results['q2'],
          'train/log_pi': results['logp_pi'],
          'train/loss_pi': results['pi_loss'],
          'train/loss_q1': results['q1_loss'],
          'train/loss_q2': results['q2_loss'],
          'train/loss_reg': results['reg_loss'],
          'train/agem_violation': results['agem_violation'],
        })

        for i in range(num_tasks):
          if auto_alpha:
            logger.store({'train/alpha/{}'.format(i):
                            float(tf.math.exp(all_log_alpha[i][0]))})
          if critic_cl is PopArtMlpCritic:
            logger.store({
              'train/popart_mean/{}'.format(i): critic1.moment1[i][0],
              'train/popart_std/{}'.format(i): critic1.sigma[i][0]
            })

    if (cl_method == 'packnet' and (current_task_t + 1 == env.steps_per_env) and
            current_task_idx < env.num_envs - 1):
      if current_task_idx == 0:
        packnet_helper.set_freeze_biases_and_normalization(True)

      # Each task gets equal share of 'kernel' weights.
      if packnet_fake_num_tasks is not None:
        num_tasks_left = packnet_fake_num_tasks - current_task_idx - 1
      else:
        num_tasks_left = env.num_envs - current_task_idx - 1
      prune_perc = num_tasks_left / (num_tasks_left + 1)
      packnet_helper.prune(prune_perc, current_task_idx)

      reset_optimizer(optimizer)

      for _ in range(packnet_retrain_steps):
        batch = replay_buffer.sample_batch(batch_size)
        learn_on_batch(tf.convert_to_tensor(current_task_idx), batch)

      reset_optimizer(optimizer)

    # End of epoch wrap-up
    if ((t + 1) % log_every == 0) or (t + 1 == steps):
      epoch = (t + 1 + log_every - 1) // log_every

      # Save model
      if (epoch % save_freq_epochs == 0) or (t + 1 == steps):
        dir_prefixes = []
        if current_task_idx == -1:
          dir_prefixes.append('./checkpoints')
        else:
          dir_prefixes.append('./checkpoints/task{}'.format(current_task_idx))
          if current_task_idx == num_tasks - 1:
            dir_prefixes.append('./checkpoints')

        for prefix in dir_prefixes:
          actor.save_weights(os.path.join(prefix, 'actor'))
          critic1.save_weights(os.path.join(prefix, 'critic1'))
          target_critic1.save_weights(os.path.join(prefix, 'target_critic1'))
          critic2.save_weights(os.path.join(prefix, 'critic2'))
          target_critic2.save_weights(os.path.join(prefix, 'target_critic2'))

      # Test the performance of the deterministic version of the agent.
      test_agent()

      # Log info about epoch
      logger.log_tabular('epoch', epoch)
      logger.log_tabular('train/return', with_min_and_max=True)
      logger.log_tabular('train/ep_length', average_only=True)
      logger.log_tabular('total_env_steps', t + 1)
      logger.log_tabular('current_task_steps', current_task_t + 1)
      logger.log_tabular('train/q1_vals', with_min_and_max=True)
      logger.log_tabular('train/q2_vals', with_min_and_max=True)
      logger.log_tabular('train/log_pi', with_min_and_max=True)
      logger.log_tabular('train/loss_pi', average_only=True)
      logger.log_tabular('train/loss_q1', average_only=True)
      logger.log_tabular('train/loss_q2', average_only=True)
      for i in range(num_tasks):
        if auto_alpha:
          logger.log_tabular('train/alpha/{}'.format(i), average_only=True)
        if critic_cl is PopArtMlpCritic:
          logger.log_tabular('train/popart_mean/{}'.format(i),
                             average_only=True)
          logger.log_tabular('train/popart_std/{}'.format(i), average_only=True)
      logger.log_tabular('train/loss_reg', average_only=True)
      logger.log_tabular('train/agem_violation', average_only=True)

      # TODO: We assume here that SuccessCounter is outermost wrapper.
      avg_success = np.mean(env.pop_successes())
      logger.log_tabular('train/success', avg_success)
      if 'seq_idx' in info:
        logger.log_tabular('train/active_env', info['seq_idx'])

      logger.log_tabular('walltime', time.time() - start_time)
      logger.dump_tabular()

    current_task_t += 1
