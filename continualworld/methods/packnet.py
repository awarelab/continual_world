from typing import Dict, List, Tuple

import tensorflow as tf

from continualworld.sac.sac import SAC
from continualworld.utils.utils import reset_optimizer


class PackNet_SAC(SAC):
    def __init__(
        self, regularize_critic: bool = False, retrain_steps: int = 0, **vanilla_sac_kwargs
    ) -> None:
        """PackNet method. See https://arxiv.org/abs/1711.05769 .

        Args:
          regularize_critic: If True, both actor and critic are regularized; if False, only actor
            is regularized.
          retrain_steps: Number of retrain steps after network pruning, which occurs after
            each task.
        """
        super().__init__(**vanilla_sac_kwargs)
        self.regularize_critic = regularize_critic
        self.retrain_steps = retrain_steps

        packnet_models = [self.actor]
        if self.regularize_critic:
            packnet_models.extend([self.critic1, self.critic2])

        self.owner = {}
        self.saved_variables = {}
        self.current_view = tf.Variable(-1, trainable=False)
        self.managed_variable_refs = set()
        for model in packnet_models:
            if model.num_heads == 1:
                variables_to_manage = model.trainable_variables
            else:
                # If there are more heads, do not touch them with PackNet.
                variables_to_manage = model.core.trainable_variables
            for v in variables_to_manage:
                self.managed_variable_refs.add(v.ref())
                if "kernel" in v.name:
                    self.owner[v.ref()] = tf.Variable(
                        tf.zeros_like(v, dtype=tf.int32), trainable=False
                    )
                    self.saved_variables[v.ref()] = tf.Variable(tf.zeros_like(v), trainable=False)
        self.freeze_biases_and_normalization = tf.Variable(False, trainable=False)

    def adjust_gradients(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: List[tf.Tensor],
        current_task_idx: int,
        metrics: dict,
        episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        actor_gradients = self._adjust_gradients_list(
            actor_gradients, self.actor.trainable_variables, tf.convert_to_tensor(current_task_idx)
        )
        if self.regularize_critic:
            critic_gradients = self._adjust_gradients_list(
                critic_gradients, self.critic_variables, tf.convert_to_tensor(current_task_idx)
            )
        return actor_gradients, critic_gradients, alpha_gradient

    def on_test_start(self, seq_idx: tf.Tensor) -> None:
        self._set_view(seq_idx)

    def on_test_end(self, seq_idx: tf.Tensor) -> None:
        self._set_view(-1)

    def on_task_end(self, current_task_idx: int) -> None:
        if current_task_idx < self.env.num_envs - 1:
            if current_task_idx == 0:
                self._set_freeze_biases_and_normalization(True)

            # Each task gets equal share of 'kernel' weights.
            num_tasks_left = self.env.num_envs - current_task_idx - 1
            prune_perc = num_tasks_left / (num_tasks_left + 1)
            self._prune(prune_perc, current_task_idx)

            reset_optimizer(self.optimizer)

            for _ in range(self.retrain_steps):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                self.learn_on_batch(tf.convert_to_tensor(current_task_idx), batch)

            reset_optimizer(self.optimizer)

    @tf.function
    def _adjust_gradients_list(
        self, grads: List[tf.Tensor], variables: List[tf.Variable], seq_idx: int
    ) -> List[tf.Tensor]:
        """Computes PackNet adjustment to the gradients to be used in gradient step.

        Args:
          grads: original gradients
          variables: variables corresponding to the original gradients
          seq_idx: number of the task we are currently in

        Returns:
          List[tf.Tensor]: adjusted gradients
        """
        res = []
        assert len(grads) == len(variables)
        for g, v in zip(grads, variables):
            if v.ref() in self.managed_variable_refs:
                if "kernel" in v.name:
                    res.append(g * tf.cast(self.owner[v.ref()] == seq_idx, tf.float32))
                else:
                    res.append(
                        g * (1.0 - tf.cast(self.freeze_biases_and_normalization, tf.float32))
                    )
            else:
                res.append(g)
        return res

    def _prune(self, prune_perc: float, seq_idx: int) -> None:
        """Prune given percentage of weights previously used by a given task.

        Args:
          prune_perc: percentage to prune
          seq_idx: number of the task to prune weights from
        """
        for ref, owner in self.owner.items():
            v = ref.deref()
            vals = v[owner == seq_idx]
            vals = tf.sort(tf.abs(vals))
            threshold_index = tf.cast(tf.cast(tf.shape(vals)[0], tf.float32) * prune_perc, tf.int32)
            threshold = vals[threshold_index]
            keep_mask = (tf.abs(v) > threshold) | (owner != seq_idx)
            v.assign(v * tf.cast(keep_mask, tf.float32))
            owner.assign(
                owner * tf.cast(keep_mask, tf.int32) + (seq_idx + 1) * tf.cast(~keep_mask, tf.int32)
            )

    def _set_view(self, seq_idx: int) -> None:
        """Bring back the version of the models from a moment corresponding to a given task.

        Args:
          seq_idx: Number of a task. If this value is N >= 0, then the weights corresponding to
            tasks 0, 1, ..., N will be set to their real values, and the rest will be set to 0. This
            is used to do inference on task N.
            If seq_idx is -1, all weights will be set to their current values. This mode is used in
            training.
        """
        if seq_idx == -1:
            for ref, saved_variable in self.saved_variables.items():
                v = ref.deref()
                v.assign(saved_variable)
        else:
            for ref, owner in self.owner.items():
                v = ref.deref()
                self.saved_variables[ref].assign(v)
                v.assign(v * tf.cast(owner <= seq_idx, tf.float32))
        self.current_view.assign(seq_idx)

    def _set_freeze_biases_and_normalization(self, value: bool) -> None:
        self.freeze_biases_and_normalization.assign(value)
