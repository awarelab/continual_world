from typing import List

import tensorflow as tf

from continualworld.sac.replay_buffers import ReplayBuffer
from continualworld.sac.sac import SAC


class Regularization_SAC(SAC):
    def __init__(self, cl_reg_coef=1.0, regularize_critic=False, **vanilla_sac_kwargs):
        """Class for regularization methods.

        Args:
          cl_reg_coef: Regularization strength for continual learning methods.
            Valid for 'l2', 'ewc', 'mas' continual learning methods.
          regularize_critic: If True, both actor and critic are regularized; if False, only actor
            is regularized.
        """
        super().__init__(**vanilla_sac_kwargs)
        self.cl_reg_coef = cl_reg_coef
        self.regularize_critic = regularize_critic
        self.old_params = list(
            tf.Variable(tf.identity(param), trainable=False) for param in self.all_common_variables
        )

        self.actor_common_variables = self.actor.common_variables
        self.critic_common_variables = self.critic1.common_variables + self.critic2.common_variables

        self.reg_weights = list(
            tf.Variable(tf.zeros_like(param), trainable=False)
            for param in self.all_common_variables
        )

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        aux_loss = self._regularize(self.old_params)
        aux_loss_coef = tf.cond(seq_idx > 0, lambda: self.cl_reg_coef, lambda: 0.0)
        aux_loss *= aux_loss_coef

        return aux_loss

    def on_task_start(self, current_task_idx: int) -> None:
        if current_task_idx > 0:
            for old_param, new_param in zip(self.old_params, self.all_common_variables):
                old_param.assign(new_param)
            self._update_reg_weights(self.replay_buffer)

    def _merge_weights(self, new_weights: List[tf.Variable]) -> None:
        """Merge the parameter importance weights for current task with the importance weights
        of previous tasks."""
        if self.reg_weights is not None:
            merged_weights = list(
                old_reg + new_reg for old_reg, new_reg in zip(self.reg_weights, new_weights)
            )
        else:
            merged_weights = new_weights

        for old_weight, new_weight in zip(self.reg_weights, merged_weights):
            old_weight.assign(new_weight)

    def _update_reg_weights(
        self, replay_buffer: ReplayBuffer, batches_num: int = 10, batch_size: int = 256
    ) -> None:
        """Calculate importance weights representing how important each weight is for the current
        task."""
        all_weights = []
        for batch_idx in range(batches_num):
            batch = replay_buffer.sample_batch(batch_size)
            all_weights += [self._get_importance_weights(**batch)]

        mean_weights = []
        for weights in zip(*all_weights):
            mean_weights += [tf.reduce_mean(tf.stack(weights, 0), 0)]

        self._merge_weights(mean_weights)

    def _regularize(self, old_params: List[tf.Tensor]) -> tf.Tensor:
        """Calculate the regularization loss based on previous parameters and parameter weights."""
        reg_loss = tf.zeros([])
        for new_param, old_param, weight in zip(
            self.all_common_variables, old_params, self.reg_weights
        ):
            diffs = (new_param - old_param) ** 2
            weighted_diffs = weight * diffs
            reg_loss += tf.reduce_sum(weighted_diffs)
        return reg_loss

    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:
        raise NotImplementedError
