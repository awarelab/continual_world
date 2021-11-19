from typing import List, Tuple

import tensorflow as tf

from continualworld.methods.regularization import Regularization_SAC


class MAS_SAC(Regularization_SAC):
    """MAS regularization method.

    https://arxiv.org/abs/1711.09601"""

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
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(persistent=True) as g:
            # Main outputs from computation graph
            mu, log_std, pi, logp_pi = self.actor(obs)

            # Get squared L2 norm per-example
            actor_norm = tf.reduce_sum(mu ** 2, -1) + tf.reduce_sum(log_std ** 2, -1)

            q1 = self.critic1(obs, actions)
            critic1_norm = q1 ** 2

            q2 = self.critic2(obs, actions)
            critic2_norm = q2 ** 2

        # Compute gradients for MAS
        actor_gs = g.jacobian(actor_norm, self.actor_common_variables)
        q1_gs = g.jacobian(critic1_norm, self.critic1.common_variables)
        q2_gs = g.jacobian(critic2_norm, self.critic2.common_variables)
        del g
        return actor_gs, q1_gs, q2_gs

    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:
        actor_gs, q1_gs, q2_gs = self._get_grads(**batch)

        reg_weights = []
        for g in actor_gs:
            reg_weights += [tf.reduce_mean(tf.abs(g), 0)]

        critic_coef = 1.0 if self.regularize_critic else 0.0
        for g in q1_gs:
            reg_weights += [critic_coef * tf.reduce_mean(tf.abs(g), 0)]

        for g in q2_gs:
            reg_weights += [critic_coef * tf.reduce_mean(tf.abs(g), 0)]

        return reg_weights
