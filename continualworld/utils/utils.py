import argparse
import random
import string
from datetime import datetime
from typing import Callable, Dict, Optional, Type, Union

import gym
import numpy as np
import tensorflow as tf


def set_seed(seed: int, env: Optional[gym.Env] = None) -> None:
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    if env:
        env.action_space.seed(seed)


def get_activation_from_str(name: str) -> Callable:
    if name == "tanh":
        return tf.tanh
    if name == "relu":
        return tf.nn.relu
    if name == "elu":
        return tf.nn.elu
    if name == "lrelu":
        return tf.nn.leaky_relu
    assert False, "Bad activation function name!"


# https://stackoverflow.com/a/43357954/6365092
def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def sci2int(v: str) -> int:
    # Scientific Notation cannot be converted directly to int,
    # so here's a workaround
    return int(float(v))


def float_or_str(v: Union[float, str]) -> Union[float, str]:
    # If it is possible, convert to float. Otherwise leave str as it is.
    try:
        float_v = float(v)
        return float_v
    except ValueError:
        return v


def reset_optimizer(optimizer: tf.keras.optimizers.Optimizer) -> None:
    # Skip the first variable, its step count.
    for var in optimizer.variables()[1:]:
        var.assign(tf.zeros_like(var))


def reset_weights(
    model: tf.keras.Model, model_cl: Type[tf.keras.Model], model_kwargs: Dict
) -> None:
    """Re-initialize randomly weights of the model.

    Args:
        model: model to re-initialize weights
        model_cl: model class that matches the class of the model argument
        model_kwargs: kwargs that need to be passed to model_cl
    """
    dummy_model = model_cl(**model_kwargs)
    model.set_weights(dummy_model.get_weights())


def get_readable_timestamp() -> str:
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def get_random_string(n: int = 6) -> str:
    return "".join(
        random.choice(string.ascii_lowercase + string.ascii_uppercase + string.digits)
        for _ in range(n)
    )
