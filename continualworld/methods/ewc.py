from typing import List, Tuple

import numpy as np
import tensorflow as tf

from continualworld.methods.regularization import Regularization_SAC


class EWC_SAC(Regularization_SAC):
    """EWC regularization method.

    https://arxiv.org/abs/1612.00796"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @tf.function
    def _get_grads(
        self,
        obs: tf.Tensor,
        next_obs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        done: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(persistent=True) as g:
            # Main outputs from computation graph
            mu, log_std, pi, logp_pi = self.actor(obs)
            std = tf.exp(log_std)

            q1 = self.critic1(obs, actions)
            q2 = self.critic2(obs, actions)

        # Compute diagonal of the Fisher matrix
        actor_mu_gs = g.jacobian(mu, self.actor_common_variables)
        actor_std_gs = g.jacobian(std, self.actor_common_variables)
        q1_gs = g.jacobian(q1, self.critic1.common_variables)
        q2_gs = g.jacobian(q2, self.critic2.common_variables)
        del g
        return actor_mu_gs, actor_std_gs, q1_gs, q2_gs, std

    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:
        actor_mu_gs, actor_std_gs, q1_gs, q2_gs, std = self._get_grads(**batch)

        reg_weights = []
        for mu_g, std_g in zip(actor_mu_gs, actor_std_gs):
            if mu_g is None and std_g is None:
                raise ValueError("Both mu and std gradients are None!")
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

        critic_coef = 1.0 if self.regularize_critic else 0.0
        for q_g in q1_gs:
            fisher = q_g ** 2
            reg_weights += [critic_coef * tf.reduce_mean(fisher, 0)]

        for q_g in q2_gs:
            fisher = q_g ** 2
            reg_weights += [critic_coef * tf.reduce_mean(fisher, 0)]

        return reg_weights
