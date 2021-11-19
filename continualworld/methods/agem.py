from typing import Dict, List, Optional, Tuple

import tensorflow as tf

from continualworld.sac.replay_buffers import EpisodicMemory
from continualworld.sac.sac import SAC


class AGEM_SAC(SAC):
    def __init__(
        self, episodic_mem_per_task: int = 0, episodic_batch_size: int = 0, **vanilla_sac_kwargs
    ):
        """AGEM method. See https://arxiv.org/abs/1812.00420 .

        Args:
          episodic_mem_per_task: Number of examples to keep in additional memory per task.
            Valid for 'agem' continual learning method.
          episodic_batch_size: Minibatch size to compute additional loss in 'agem' continual
            learning method.
        """
        super().__init__(**vanilla_sac_kwargs)
        self.episodic_mem_per_task = episodic_mem_per_task
        self.episodic_batch_size = episodic_batch_size

        episodic_mem_size = self.episodic_mem_per_task * self.env.num_envs
        self.episodic_memory = EpisodicMemory(
            obs_dim=self.obs_dim, act_dim=self.act_dim, size=episodic_mem_size
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
        if current_task_idx > 0:
            (ref_actor_gradients, ref_critic_gradients, _), _ = self.get_gradients(
                seq_idx=tf.constant(-1), **episodic_batch
            )

            dot_prod = 0.0
            ref_squared_norm = 0.0
            for new_gradient, ref_gradient in zip(actor_gradients, ref_actor_gradients):
                dot_prod += tf.reduce_sum(new_gradient * ref_gradient)
                ref_squared_norm += tf.reduce_sum(ref_gradient * ref_gradient)

            for new_gradient, ref_gradient in zip(critic_gradients, ref_critic_gradients):
                dot_prod += tf.reduce_sum(new_gradient * ref_gradient)
                ref_squared_norm += tf.reduce_sum(ref_gradient * ref_gradient)

            violation = tf.cond(dot_prod >= 0, lambda: 0, lambda: 1)

            actor_gradients = self._project_gradients(
                actor_gradients, ref_actor_gradients, dot_prod, ref_squared_norm
            )
            critic_gradients = self._project_gradients(
                critic_gradients, ref_critic_gradients, dot_prod, ref_squared_norm
            )

            metrics["agem_violation"] = violation

        return actor_gradients, critic_gradients, alpha_gradient

    def on_task_start(self, current_task_idx: int) -> None:
        if current_task_idx > 0:
            new_episodic_mem = self.replay_buffer.sample_batch(self.episodic_mem_per_task)
            self.episodic_memory.store_multiple(**new_episodic_mem)

    def get_episodic_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        if current_task_idx > 0:
            return self.episodic_memory.sample_batch(self.episodic_batch_size)
        return None

    def _project_gradients(
        self,
        new_gradients: List[tf.Tensor],
        ref_gradients: List[tf.Tensor],
        dot_prod: tf.Tensor,
        ref_squared_norm: tf.Tensor,
    ) -> List[tf.Tensor]:
        projected_grads = []
        for new_gradient, ref_gradient in zip(new_gradients, ref_gradients):
            projected_grads += [
                tf.cond(
                    dot_prod >= 0,
                    lambda: new_gradient,
                    lambda: new_gradient - (dot_prod / ref_squared_norm * ref_gradient),
                )
            ]
        return projected_grads
