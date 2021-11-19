import tensorflow as tf

from continualworld.methods.regularization import Regularization_SAC
from continualworld.sac.replay_buffers import ReplayBuffer


class L2_SAC(Regularization_SAC):
    """L2 regularization method."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _update_reg_weights(
        self, replay_buffer: ReplayBuffer, batches_num: int = 10, batch_size: int = 256
    ) -> None:
        if self.regularize_critic:
            new_weights = list(tf.ones_like(param) for param in self.all_common_variables)
        else:
            new_weights = list(tf.ones_like(param) for param in self.actor_common_variables) + list(
                tf.zeros_like(param) for param in self.critic_common_variables
            )

        self._merge_weights(new_weights)
