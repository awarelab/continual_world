import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from tensorflow.keras import Input, Model
from tensorflow.python.ops import nn

from envs import MW_OBS_LEN, MW_ACT_LEN
from spinup.models import apply_squashing_func, gaussian_likelihood, _choose_head

EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class VclDense(tfk.layers.Layer):
    def __init__(self, input_dim, output_dim, activation=None, enable_kl=True, num_heads=1):
      super().__init__()

      logvar_init = tf.constant_initializer(-6.)
      w_init = tfk.initializers.GlorotUniform()
      b_init = tf.zeros_initializer()

      self.input_dim = input_dim
      self.output_dim = output_dim
      self.num_heads = num_heads

      self.posterior_w_mean = tf.Variable(
          initial_value=w_init(shape=(input_dim, output_dim), dtype='float32'),
          trainable=True,
      )
      self.posterior_w_logvar = tf.Variable(
          initial_value=logvar_init(shape=(input_dim, output_dim), dtype='float32'),
          trainable=True,
      )

      self.posterior_b_mean = tf.Variable(
          initial_value=b_init(shape=(output_dim,), dtype='float32'), trainable=True
      )
      self.posterior_b_logvar = tf.Variable(
          initial_value=logvar_init(shape=(output_dim,), dtype='float32'), trainable=True
      )

      self.prior_w_mean = tf.Variable(
          tf.zeros_like(self.posterior_w_mean), trainable=False)
      self.prior_w_logvar = tf.Variable(
          tf.zeros_like(self.posterior_w_logvar), trainable=False)

      self.prior_b_mean = tf.Variable(
          tf.zeros_like(self.posterior_b_mean), trainable=False)
      self.prior_b_logvar = tf.Variable(
          tf.zeros_like(self.posterior_b_logvar), trainable=False)

      self.activation = activation
      self.enable_kl = enable_kl


    def call(self, inputs):
      eps_w = tf.random.normal((self.input_dim, self.output_dim), 0, 1, dtype=tf.float32)
      eps_b = tf.random.normal((1, self.output_dim), 0, 1, dtype=tf.float32)

      weights = eps_w * tf.exp(0.5 * self.posterior_w_logvar) + self.posterior_w_mean
      biases = eps_b * tf.exp(0.5 * self.posterior_b_logvar) + self.posterior_b_mean
      output = tf.matmul(inputs, weights) + biases

      if self.activation is not None:
          output = self.activation(output)

      return output

class VclLayerNormalization(tfk.layers.Layer):

  def __init__(self, output_dim, epsilon=1e-3):
    super().__init__()

    self.output_dim = output_dim
    self.epsilon = epsilon

    logvar_init = tf.constant_initializer(-6.)

    self.posterior_beta_mean = tf.Variable(
        initial_value=tf.zeros(output_dim, dtype='float32'), trainable=True
    )
    self.posterior_beta_logvar = tf.Variable(
        initial_value=logvar_init(output_dim, dtype='float32'), trainable=True
    )

    self.posterior_gamma_mean = tf.Variable(
        initial_value=tf.ones(output_dim, dtype='float32'), trainable=True
    )
    self.posterior_gamma_logvar = tf.Variable(
        initial_value=logvar_init(output_dim, dtype='float32'), trainable=True
    )

    self.prior_beta_mean = tf.Variable(
        tf.zeros_like(self.posterior_beta_mean), trainable=False)
    self.prior_beta_logvar = tf.Variable(
        tf.zeros_like(self.posterior_beta_logvar), trainable=False)

    self.prior_gamma_mean = tf.Variable(
        tf.zeros_like(self.posterior_gamma_mean), trainable=False)
    self.prior_gamma_logvar = tf.Variable(
        tf.zeros_like(self.posterior_gamma_logvar), trainable=False)

  def call(self, inputs):
    mean, variance = nn.moments(inputs, -1, keep_dims=True)
    eps_gamma = tf.random.normal((self.output_dim,))
    eps_beta = tf.random.normal((self.output_dim,))

    gamma = eps_gamma * tf.exp(0.5 * self.posterior_gamma_logvar) + self.posterior_gamma_mean
    beta = eps_beta * tf.exp(0.5 * self.posterior_beta_logvar) + self.posterior_beta_mean

    # Compute layer normalization using the batch_normalization function.
    outputs = nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=self.epsilon)

    return outputs

def variational_mlp(
        input_dim, hidden_sizes, activation,
        use_layer_norm=False, variational_ln=False):
  model = tf.keras.Sequential()
  model.add(Input(shape=(input_dim,)))
  model.add(VclDense(input_dim, hidden_sizes[0]))
  if use_layer_norm:
    if variational_ln:
      model.add(VclLayerNormalization(hidden_sizes[0]))
    else:
      model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Activation(tf.nn.tanh))
  else:
    model.add(tf.keras.layers.Activation(activation))
  for layer_idx in range(1, len(hidden_sizes)):
    prev_size, next_size = hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]
    model.add(VclDense(prev_size, next_size, activation=activation))
  return model

class VclMlpActor(Model):
  def __init__(self, input_dim, action_space, hidden_sizes=(256, 256),
               activation=tf.tanh, use_layer_norm=False, variational_ln=False,
               num_heads=1, hide_task_id=False):
    super(VclMlpActor, self).__init__()

    self.num_heads = num_heads
    self.hide_task_id = hide_task_id

    if self.hide_task_id:
      input_dim = MW_OBS_LEN

    self.core = variational_mlp(
            input_dim, hidden_sizes,
            activation, use_layer_norm=use_layer_norm,
            variational_ln=variational_ln)

    self.head_mu = tf.keras.Sequential([
      Input(shape=(hidden_sizes[-1],)),
      VclDense(hidden_sizes[-1], action_space.shape[0] * num_heads, num_heads=num_heads)
    ])
    self.head_log_std = tf.keras.Sequential([
      Input(shape=(hidden_sizes[-1],)),
      VclDense(hidden_sizes[-1], action_space.shape[0] * num_heads, num_heads=num_heads)
    ])
    self.action_space = action_space

  @property
  def common_variables(self):
    return (self.core.trainable_variables
            + self.head_mu.trainable_variables
            + self.head_log_std.trainable_variables)

  def call(self, x, samples_num=1):
    input_x = x
    full_obs = x
    if self.hide_task_id:
      input_x = input_x[:, :MW_OBS_LEN]
    mus, pis = [], []

    for sample_idx in range(samples_num):
      x = self.core(input_x)
      mu = self.head_mu(x)
      log_std = self.head_log_std(x)

      if self.num_heads > 1:
        mu = _choose_head(mu, full_obs, self.num_heads)
        log_std = _choose_head(log_std, full_obs, self.num_heads)

      log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
      std = tf.exp(log_std)
      pi = mu + tf.random.normal(tf.shape(input=mu)) * std
      logp_pi = gaussian_likelihood(pi, mu, log_std)

      mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

      # Make sure actions are in correct range
      action_scale = self.action_space.high[0]
      mu *= action_scale
      pi *= action_scale

      mus += [mu]
      pis += [pi]

    mu = tf.reduce_mean(tf.stack(mus), 0)
    pi = tf.reduce_mean(tf.stack(pis), 0)

    return mu, log_std, pi, logp_pi


def kl_divergence(posterior_mean, posterior_logvar, prior_mean, prior_logvar):
  numel = tf.cast(tf.size(posterior_mean), tf.float32)
  const_term = -0.5 * numel
  log_std_diff = 0.5 * tf.reduce_sum(prior_logvar - posterior_logvar)

  posterior_var = tf.exp(posterior_logvar)
  prior_var = tf.exp(prior_logvar)

  mu_diff_term = 0.5 * tf.reduce_sum((posterior_var + (posterior_mean  - prior_mean) ** 2) / (prior_var))
  kl = const_term + log_std_diff + mu_diff_term
  return kl


class VclHelper:
  def __init__(self, actor, critic1, critic2, regularize_critic):
    assert regularize_critic is False, "VCL critic reg not supported"
    self.reg_layers = (actor.core.layers
                       + actor.head_mu.layers
                       + actor.head_log_std.layers)

  def update_prior(self):
    for layer in self.reg_layers:
      if isinstance(layer, (VclDense, VclLayerNormalization)):
        self.update_layer(layer)
      if isinstance(layer, tf.keras.layers.LayerNormalization):
        layer.trainable = False

  def update_layer(self, layer):
    if isinstance(layer, VclDense) and layer.num_heads == 1:
      layer.prior_w_mean.assign(layer.posterior_w_mean)
      layer.prior_w_logvar.assign(layer.posterior_w_logvar)

      layer.prior_b_mean.assign(layer.posterior_b_mean)
      layer.prior_b_logvar.assign(layer.posterior_b_logvar)
    elif isinstance(layer, VclLayerNormalization):
      layer.prior_beta_mean.assign(layer.posterior_beta_mean)
      layer.prior_beta_logvar.assign(layer.posterior_beta_logvar)

      layer.prior_gamma_mean.assign(layer.posterior_gamma_mean)
      layer.prior_gamma_logvar.assign(layer.posterior_gamma_logvar)


  def regularize(self, seq_idx, regularize_last_layer):
    kl_loss = 0.
    for layer in self.reg_layers:
      if isinstance(layer, VclDense):
        if layer.num_heads > 1:  # Last layer
          if not regularize_last_layer:
            continue
          input_dim = tf.shape(layer.posterior_w_mean)[0]
          new_shape = (input_dim, -1, layer.num_heads)
          kl_loss += kl_divergence(
            tf.reshape(layer.posterior_w_mean, new_shape)[:, :, seq_idx],
            tf.reshape(layer.posterior_w_logvar, new_shape)[:, :, seq_idx],
            tf.reshape(layer.prior_w_mean, new_shape)[:, :, seq_idx],
            tf.reshape(layer.prior_w_logvar, new_shape)[:, :, seq_idx]
          )
          kl_loss += kl_divergence(
            tf.reshape(layer.posterior_b_mean, (-1, layer.num_heads))[:, seq_idx],
            tf.reshape(layer.posterior_b_logvar, (-1, layer.num_heads))[:, seq_idx],
            tf.reshape(layer.prior_b_mean, (-1, layer.num_heads))[:, seq_idx],
            tf.reshape(layer.prior_b_logvar, (-1, layer.num_heads))[:, seq_idx],
          )
        else:
          kl_loss += kl_divergence(
             layer.posterior_w_mean, layer.posterior_w_logvar,
             layer.prior_w_mean, layer.prior_w_logvar)
          kl_loss += kl_divergence(
             layer.posterior_b_mean, layer.posterior_b_logvar,
             layer.prior_b_mean, layer.prior_b_logvar)
      if isinstance(layer, VclLayerNormalization):
        kl_loss += kl_divergence(
            layer.posterior_gamma_mean, layer.posterior_gamma_logvar,
            layer.prior_gamma_mean, layer.prior_gamma_logvar)
        kl_loss += kl_divergence(
            layer.posterior_beta_mean, layer.posterior_beta_logvar,
            layer.prior_beta_mean, layer.prior_beta_logvar)

    return kl_loss
