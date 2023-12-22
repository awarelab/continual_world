import math
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import tensorflow as tf

from continualworld.sac import models
from continualworld.sac.models import PopArtMlpCritic
from continualworld.sac.replay_buffers import ReplayBuffer, ReservoirReplayBuffer
from continualworld.sac.utils.logx import EpochLogger
from continualworld.utils.enums import BufferType
from continualworld.utils.utils import reset_optimizer, reset_weights, set_seed


class SAC:
    def __init__(
        self,
        env: gym.Env,
        test_envs: List[gym.Env],
        logger: EpochLogger,
        actor_cl: type = models.MlpActor,
        actor_kwargs: Dict = None,
        critic_cl: type = models.MlpCritic,
        critic_kwargs: Dict = None,
        seed: int = 0,
        steps: int = 1_000_000,
        log_every: int = 20_000,
        replay_size: int = 1_000_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        lr: float = 1e-3,
        alpha: Union[float, str] = "auto",
        batch_size: int = 128,
        start_steps: int = 10_000,
        update_after: int = 1000,
        update_every: int = 50,
        num_test_eps_stochastic: int = 10,
        num_test_eps_deterministic: int = 1,
        max_episode_len: int = 200,
        save_freq_epochs: int = 100,
        reset_buffer_on_task_change: bool = True,
        buffer_type: BufferType = BufferType.FIFO,
        reset_optimizer_on_task_change: bool = False,
        reset_critic_on_task_change: bool = False,
        clipnorm: float = None,
        target_output_std: float = None,
        agent_policy_exploration: bool = False,
    ):
        """A class for SAC training, for either single task, continual learning or multi-task learning.
        After the instance is created, use run() function to actually run the training.

        Args:
          env: An environment on which training will be performed.
          test_envs: Environments on which evaluation will be periodically performed;
            for example, when env is a multi-task environment, test_envs can be a list of individual
            task environments.
          logger: An object for logging the results.
          actor_cl: Class for actor model.
          actor_kwargs: Kwargs for actor model.
          critic_cl: Class for critic model.
          critic_kwargs: Kwargs for critic model.
          seed: Seed for randomness.
          steps: Number of steps the algorithm will run for.
          log_every: Number of steps between subsequent evaluations and logging.
          replay_size: Size of the replay buffer.
          gamma: Discount factor.
          polyak: Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:
              target_weights <- polyak * target_weights + (1 - polyak) * weights
            (Always between 0 and 1, usually close to 1.)
          lr: Learning rate for the optimizer.
          alpha: Entropy regularization coefficient. Can be either float value,
            or "auto", in which case it is dynamically tuned.
            (Equivalent to inverse of reward scale in the original SAC paper.)
          batch_size: Minibatch size for the optimization.
          start_steps: Number of steps for uniform-random action selection, before running real
            policy. Helps exploration.
          update_after: Number of env interactions to collect before starting to do gradient
            descent updates.  Ensures replay buffer is full enough for useful updates.
          update_every: Number of env interactions that should elapse between gradient descent
            updates.
            Note: Regardless of how long you wait between updates, the ratio of env steps to
            gradient steps is locked to 1.
          num_test_eps_stochastic: Number of episodes to test the stochastic policy in each
            evaluation.
          num_test_eps_deterministic: Number of episodes to test the deterministic policy in each
            evaluation.
          max_episode_len: Maximum length of trajectory / episode / rollout.
          save_freq_epochs: How often, in epochs, to save the current policy and value function.
            (Epoch is defined as time between two subsequent evaluations, lasting log_every steps)
          reset_buffer_on_task_change: If True, replay buffer will be cleared after every task
            change (in continual learning).
          buffer_type: Type of the replay buffer. Either 'fifo' for regular FIFO buffer
            or 'reservoir' for reservoir sampling.
          reset_optimizer_on_task_change: If True, optimizer will be reset after every task change
            (in continual learning).
          reset_critic_on_task_change: If True, critic weights are randomly re-initialized after
            each task change.
          clipnorm: Value for gradient clipping.
          target_output_std: If alpha is 'auto', alpha is dynamically tuned so that standard
            deviation of the action distribution on every dimension matches target_output_std.
          agent_policy_exploration: If True, uniform exploration for start_steps steps is used only
            in the first task (in continual learning). Otherwise, it is used in every task.
        """
        set_seed(seed, env=env)

        if actor_kwargs is None:
            actor_kwargs = {}
        if critic_kwargs is None:
            critic_kwargs = {}

        self.env = env
        self.num_tasks = env.num_envs
        self.test_envs = test_envs
        self.logger = logger
        self.critic_cl = critic_cl
        self.critic_kwargs = critic_kwargs
        self.steps = steps
        self.log_every = log_every
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_eps_stochastic = num_test_eps_stochastic
        self.num_test_eps_deterministic = num_test_eps_deterministic
        self.max_episode_len = max_episode_len
        self.save_freq_epochs = save_freq_epochs
        self.reset_buffer_on_task_change = reset_buffer_on_task_change
        self.buffer_type = buffer_type
        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_critic_on_task_change = reset_critic_on_task_change
        self.clipnorm = clipnorm
        self.agent_policy_exploration = agent_policy_exploration

        self.use_popart = critic_cl is PopArtMlpCritic

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        # This implementation assumes all dimensions share the same bound!
        assert np.all(env.action_space.high == env.action_space.high[0])

        # Share information about action space with policy architecture
        actor_kwargs["action_space"] = env.action_space
        actor_kwargs["input_dim"] = self.obs_dim
        critic_kwargs["input_dim"] = self.obs_dim + self.act_dim

        # Create experience buffer
        if buffer_type == BufferType.FIFO:
            self.replay_buffer = ReplayBuffer(
                obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size
            )
        elif buffer_type == BufferType.RESERVOIR:
            self.replay_buffer = ReservoirReplayBuffer(
                obs_dim=self.obs_dim, act_dim=self.act_dim, size=replay_size
            )

        # Create actor and critic networks
        self.actor = actor_cl(**actor_kwargs)

        self.critic1 = critic_cl(**critic_kwargs)
        self.target_critic1 = critic_cl(**critic_kwargs)
        self.target_critic1.set_weights(self.critic1.get_weights())

        self.critic2 = critic_cl(**critic_kwargs)
        self.target_critic2 = critic_cl(**critic_kwargs)
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.critic_variables = self.critic1.trainable_variables + self.critic2.trainable_variables
        self.all_common_variables = (
            self.actor.common_variables
            + self.critic1.common_variables
            + self.critic2.common_variables
        )
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # For reference on automatic alpha tuning, see
        # "Automating Entropy Adjustment for Maximum Entropy" section
        # in https://arxiv.org/abs/1812.05905
        self.auto_alpha = False
        if alpha == "auto":
            self.auto_alpha = True
            self.all_log_alpha = tf.Variable(
                np.ones((self.num_tasks, 1), dtype=np.float32), trainable=True
            )
            if target_output_std is None:
                self.target_entropy = -np.prod(env.action_space.shape).astype(np.float32)
            else:
                target_1d_entropy = np.log(target_output_std * math.sqrt(2 * math.pi * math.e))
                self.target_entropy = (
                    np.prod(env.action_space.shape).astype(np.float32) * target_1d_entropy
                )

    def adjust_gradients(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: List[tf.Tensor],
        current_task_idx: int,
        metrics: dict,
        episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        return actor_gradients, critic_gradients, alpha_gradient

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        return tf.constant(0.0)

    def on_test_start(self, seq_idx: tf.Tensor) -> None:
        pass

    def on_test_end(self, seq_idx: tf.Tensor) -> None:
        pass

    def on_task_start(self, current_task_idx: int) -> None:
        pass

    def on_task_end(self, current_task_idx: int) -> None:
        pass

    def get_episodic_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        return None

    def get_log_alpha(self, obs: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(tf.linalg.matmul(obs[:, -self.num_tasks :], self.all_log_alpha))

    # @tf.function
    def get_action(self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        mu, log_std, pi, logp_pi = self.actor(tf.expand_dims(o, 0))
        if deterministic:
            return mu[0]
        else:
            return pi[0]

    def get_action_test(
        self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)
    ) -> tf.Tensor:
        return self.get_action(o, deterministic)

    def get_learn_on_batch(self, current_task_idx: int) -> Callable:
        # TODO : decorator causes error:
        # : CommandLine Error: Option 'help-list' registered more than once!
        # LLVM ERROR: inconsistency in registered CommandLine options
        # @tf.function  
        def learn_on_batch(
            seq_idx: tf.Tensor,
            batch: Dict[str, tf.Tensor],
            episodic_batch: Dict[str, tf.Tensor] = None,
        ) -> Dict:
            gradients, metrics = self.get_gradients(seq_idx, **batch)
            # Warning: we refer here to the int task_idx in the parent function, not
            # the passed seq_idx.
            gradients = self.adjust_gradients(
                *gradients,
                current_task_idx=current_task_idx,
                metrics=metrics,
                episodic_batch=episodic_batch,
            )

            if self.clipnorm is not None:
                actor_gradients, critic_gradients, alpha_gradient = gradients
                gradients = (
                    tf.clip_by_global_norm(actor_gradients, self.clipnorm)[0],
                    tf.clip_by_global_norm(critic_gradients, self.clipnorm)[0],
                    tf.clip_by_norm(alpha_gradient, self.clipnorm),
                )

            self.apply_update(*gradients)
            return metrics

        return learn_on_batch

    def get_gradients(
        self,
        seq_idx: tf.Tensor,
        obs: tf.Tensor,
        next_obs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        done: tf.Tensor,
    ) -> Tuple[Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]], Dict]:
        with tf.GradientTape(persistent=True) as g:
            if self.auto_alpha:
                log_alpha = self.get_log_alpha(obs)
            else:
                log_alpha = tf.math.log(self.alpha)

            # Main outputs from computation graph
            mu, log_std, pi, logp_pi = self.actor(obs)
            q1 = self.critic1(obs, actions)
            q2 = self.critic2(obs, actions)

            # compose q with pi, for pi-learning
            q1_pi = self.critic1(obs, pi)
            q2_pi = self.critic2(obs, pi)

            # get actions and log probs of actions for next states, for Q-learning
            _, _, pi_next, logp_pi_next = self.actor(next_obs)

            # target q values, using actions from *current* policy
            target_q1 = self.target_critic1(next_obs, pi_next)
            target_q2 = self.target_critic2(next_obs, pi_next)

            # Min Double-Q:
            min_q_pi = tf.minimum(q1_pi, q2_pi)
            min_target_q = tf.minimum(target_q1, target_q2)

            # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
            if self.critic_cl is PopArtMlpCritic:
                q_backup = tf.stop_gradient(
                    self.critic1.normalize(
                        rewards
                        + self.gamma
                        * (1 - done)
                        * (
                            self.critic1.unnormalize(min_target_q, next_obs)
                            - tf.math.exp(log_alpha) * logp_pi_next
                        ),
                        obs,
                    )
                )
            else:
                q_backup = tf.stop_gradient(
                    rewards
                    + self.gamma
                    * (1 - done)
                    * (min_target_q - tf.math.exp(log_alpha) * logp_pi_next)
                )

            # Soft actor-critic losses
            pi_loss = tf.reduce_mean(tf.math.exp(log_alpha) * logp_pi - min_q_pi)
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
            value_loss = q1_loss + q2_loss

            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    log_alpha * tf.stop_gradient(logp_pi + self.target_entropy)
                )

            auxiliary_loss = self.get_auxiliary_loss(seq_idx)
            metrics = dict(
                pi_loss=pi_loss,
                q1_loss=q1_loss,
                q2_loss=q2_loss,
                q1=q1,
                q2=q2,
                logp_pi=logp_pi,
                reg_loss=auxiliary_loss,
                agem_violation=0,
            )

            pi_loss += auxiliary_loss
            value_loss += auxiliary_loss

        # Compute gradients
        actor_gradients = g.gradient(pi_loss, self.actor.trainable_variables)
        critic_gradients = g.gradient(value_loss, self.critic_variables)
        if self.auto_alpha:
            alpha_gradient = g.gradient(alpha_loss, self.all_log_alpha)
        else:
            alpha_gradient = None
        del g

        if self.use_popart:
            # Stats are shared between critic1 and critic2.
            # We keep them only in critic1.
            self.critic1.update_stats(q_backup, obs)

        gradients = (actor_gradients, critic_gradients, alpha_gradient)
        return gradients, metrics

    def apply_update(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: List[tf.Tensor],
    ) -> None:
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self.optimizer.apply_gradients(zip(critic_gradients, self.critic_variables))

        if self.auto_alpha:
            self.optimizer.apply_gradients([(alpha_gradient, self.all_log_alpha)])

        # Polyak averaging for target variables
        for v, target_v in zip(
            self.critic1.trainable_variables, self.target_critic1.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
        for v, target_v in zip(
            self.critic2.trainable_variables, self.target_critic2.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)

    def test_agent(self, deterministic, num_episodes) -> None:
        avg_success = []
        mode = "deterministic" if deterministic else "stochastic"
        for seq_idx, test_env in enumerate(self.test_envs):
            key_prefix = f"test/{mode}/{seq_idx}/{test_env.name}/"

            self.on_test_start(seq_idx)

            for j in range(num_episodes):
                obs, info = test_env.reset()
                done = False
                episode_return = 0
                episode_len = 0
                while not (done or (episode_len == self.max_episode_len)):
                    obs, reward, terminated, truncated, info = test_env.step(
                        self.get_action_test(tf.convert_to_tensor(obs), tf.constant(deterministic))
                    )
                    episode_return += reward
                    episode_len += 1
                self.logger.store(
                    {key_prefix + "return": episode_return, key_prefix + "ep_length": episode_len}
                )

            self.on_test_end(seq_idx)

            self.logger.log_tabular(key_prefix + "return", with_min_and_max=True)
            self.logger.log_tabular(key_prefix + "ep_length", average_only=True)
            env_success = test_env.pop_successes()
            avg_success += env_success
            self.logger.log_tabular(key_prefix + "success", np.mean(env_success))
        key = f"test/{mode}/average_success"
        self.logger.log_tabular(key, np.mean(avg_success))

    def _log_after_update(self, results):
        self.logger.store(
            {
                "train/q1_vals": results["q1"],
                "train/q2_vals": results["q2"],
                "train/log_pi": results["logp_pi"],
                "train/loss_pi": results["pi_loss"],
                "train/loss_q1": results["q1_loss"],
                "train/loss_q2": results["q2_loss"],
                "train/loss_reg": results["reg_loss"],
                "train/agem_violation": results["agem_violation"],
            }
        )

        for task_idx in range(self.num_tasks):
            if self.auto_alpha:
                self.logger.store(
                    {f"train/alpha/{task_idx}": float(tf.math.exp(self.all_log_alpha[task_idx][0]))}
                )
            if self.use_popart:
                self.logger.store(
                    {
                        f"train/popart_mean/{task_idx}": self.critic1.moment1[task_idx][0],
                        f"train/popart_std/{task_idx}": self.critic1.sigma[task_idx][0],
                    }
                )

    def _log_after_epoch(self, epoch, current_task_timestep, global_timestep, info):
        # Log info about epoch
        self.logger.log_tabular("epoch", epoch)
        self.logger.log_tabular("train/return", with_min_and_max=True)
        self.logger.log_tabular("train/ep_length", average_only=True)
        self.logger.log_tabular("total_env_steps", global_timestep + 1)
        self.logger.log_tabular("current_task_steps", current_task_timestep + 1)
        self.logger.log_tabular("train/q1_vals", with_min_and_max=True)
        self.logger.log_tabular("train/q2_vals", with_min_and_max=True)
        self.logger.log_tabular("train/log_pi", with_min_and_max=True)
        self.logger.log_tabular("train/loss_pi", average_only=True)
        self.logger.log_tabular("train/loss_q1", average_only=True)
        self.logger.log_tabular("train/loss_q2", average_only=True)
        for task_idx in range(self.num_tasks):
            if self.auto_alpha:
                self.logger.log_tabular(f"train/alpha/{task_idx}", average_only=True)
            if self.use_popart:
                self.logger.log_tabular(f"train/popart_mean/{task_idx}", average_only=True)
                self.logger.log_tabular(f"train/popart_std/{task_idx}", average_only=True)
        self.logger.log_tabular("train/loss_reg", average_only=True)
        self.logger.log_tabular("train/agem_violation", average_only=True)

        avg_success = np.mean(self.env.pop_successes())
        self.logger.log_tabular("train/success", avg_success)
        if "seq_idx" in info:
            self.logger.log_tabular("train/active_env", info["seq_idx"])

        self.logger.log_tabular("walltime", time.time() - self.start_time)
        self.logger.dump_tabular()

    def save_model(self, current_task_idx):
        dir_prefixes = []
        if current_task_idx == -1:
            dir_prefixes.append("./checkpoints")
        else:
            dir_prefixes.append(f"./checkpoints/task{current_task_idx}")
            if current_task_idx == self.num_tasks - 1:
                dir_prefixes.append("./checkpoints")

        for prefix in dir_prefixes:
            self.actor.save_weights(os.path.join(prefix, "actor"))
            self.critic1.save_weights(os.path.join(prefix, "critic1"))
            self.target_critic1.save_weights(os.path.join(prefix, "target_critic1"))
            self.critic2.save_weights(os.path.join(prefix, "critic2"))
            self.target_critic2.save_weights(os.path.join(prefix, "target_critic2"))

    def _handle_task_change(self, current_task_idx: int):
        self.on_task_start(current_task_idx)

        if self.reset_buffer_on_task_change:
            assert self.buffer_type == BufferType.FIFO
            self.replay_buffer = ReplayBuffer(
                obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size
            )
        if self.reset_critic_on_task_change:
            reset_weights(self.critic1, self.critic_cl, self.critic_kwargs)
            self.target_critic1.set_weights(self.critic1.get_weights())
            reset_weights(self.critic2, self.critic_cl, self.critic_kwargs)
            self.target_critic2.set_weights(self.critic2.get_weights())

        if self.reset_optimizer_on_task_change:
            reset_optimizer(self.optimizer)

        # Update variables list and update function in case model changed.
        # E.g: For VCL after the first task we set trainable=False for layer
        # normalization. We need to recompute the graph in order for TensorFlow
        # to notice this change.
        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)
        self.all_common_variables = (
            self.actor.common_variables
            + self.critic1.common_variables
            + self.critic2.common_variables
        )

    def run(self):
        """A method to run the SAC training, after the object has been created."""
        self.start_time = time.time()
        obs, info = self.env.reset()
        episode_return = 0
        episode_len = 0

        # Main loop: collect experience in env and update/log each epoch
        current_task_timestep = 0
        current_task_idx = -1
        # self.learn_on_batch = self.get_learn_on_batch(current_task_idx)

        for global_timestep in range(self.steps):
            # On task change
            if current_task_idx != getattr(self.env, "cur_seq_idx", -1):
                print("if statement 1")
                current_task_timestep = 0
                current_task_idx = getattr(self.env, "cur_seq_idx")
                self._handle_task_change(current_task_idx)

            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if current_task_timestep > self.start_steps or (
                self.agent_policy_exploration and current_task_idx > 0
            ):
                print("if not exploring")
                action = self.get_action(tf.convert_to_tensor(obs))
            else:
                action = self.env.action_space.sample()

            # Step the env
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            episode_return += reward
            episode_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            done = np.logical_or(terminated,truncated)
            done_to_store = done
            if episode_len == self.max_episode_len or truncated:  # updated for gymnasium
                done_to_store = False

            # Store experience to replay buffer
            self.replay_buffer.store(obs, action, reward, next_obs, done_to_store)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            obs = next_obs

            # End of trajectory handling
            if done or (episode_len == self.max_episode_len):
                self.logger.store({"train/return": episode_return, "train/ep_length": episode_len})
                episode_return, episode_len = 0, 0
                if global_timestep < self.steps - 1:  # This may not work with mujoco anymore
                    obs, info = self.env.reset()

            # Update handling
            if (
                current_task_timestep >= self.update_after
                and current_task_timestep % self.update_every == 0
            ):
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)

                    episodic_batch = self.get_episodic_batch(current_task_idx)

                    ### TODO LLVM ERROR COMES FROM HERE
                    results = self.learn_on_batch(
                        tf.convert_to_tensor(current_task_idx), batch, episodic_batch
                    )
                    self._log_after_update(results)

            if (
                self.env.name == "ContinualLearningEnv"
                and current_task_timestep + 1 == self.env.steps_per_env
            ):
                self.on_task_end(current_task_idx)

            # End of epoch wrap-up
            if ((global_timestep + 1) % self.log_every == 0) or (global_timestep + 1 == self.steps):
                epoch = (global_timestep + 1 + self.log_every - 1) // self.log_every

                # Save model
                if (epoch % self.save_freq_epochs == 0) or (global_timestep + 1 == self.steps):
                    self.save_model(current_task_idx)

                # Test the performance of stochastic and detemi version of the agent.
                self.test_agent(deterministic=False, num_episodes=self.num_test_eps_stochastic)
                self.test_agent(deterministic=True, num_episodes=self.num_test_eps_deterministic)

                self._log_after_epoch(epoch, current_task_timestep, global_timestep, info)

            current_task_timestep += 1
