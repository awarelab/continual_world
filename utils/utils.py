import argparse
import tensorflow as tf


def get_activation_from_str(name):
  # We want to have string in configs so they are nicely saved by neptune.
  if name == 'tanh':
    return tf.tanh
  if name == 'relu':
    return tf.nn.relu
  if name == 'elu':
    return tf.nn.elu
  if name == 'lrelu':
    return tf.nn.leaky_relu
  assert False, 'Bad activation function name!'

# https://stackoverflow.com/a/43357954/6365092
def str2bool(v):
  if isinstance(v, bool):
   return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def sci2int(v):
    # Scientific Notation cannot be converted directly to int,
    # so here's a workaround
    return int(float(v))

def reset_optimizer(optimizer):
  # Skip the first variable, its step count.
  for var in optimizer.variables()[1:]:
    var.assign(tf.zeros_like(var))

def reset_weights(model, model_cl, model_kwargs):
  dummy_model = model_cl(**model_kwargs)
  model.set_weights(dummy_model.get_weights())

def get_script_command(run_kind):
  assert run_kind in ['cl', 'mt', 'single']
  return 'python3 run_{}.py'.format(run_kind)
