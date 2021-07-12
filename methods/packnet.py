import tensorflow as tf


class PackNetHelper:
  def __init__(self, models):
    self.owner = {}
    self.saved_variables = {}
    self.current_view = tf.Variable(-1, trainable=False)
    self.managed_variable_refs = set()
    for model in models:
      if model.num_heads == 1:
        variables_to_manage = model.trainable_variables
      else:
        # If there are more heads, do not touch them with PackNet.
        variables_to_manage = model.core.trainable_variables
      for v in variables_to_manage:
        self.managed_variable_refs.add(v.ref())
        if 'kernel' in v.name:
          self.owner[v.ref()] = tf.Variable(
            tf.zeros_like(v, dtype=tf.int32), trainable=False)
          self.saved_variables[v.ref()] = tf.Variable(
            tf.zeros_like(v), trainable=False)
    self.freeze_biases_and_normalization = tf.Variable(False, trainable=False)

  @tf.function
  def adjust_gradients(self, grads, variables, seq_idx):
    res = []
    assert len(grads) == len(variables)
    for g, v in zip(grads, variables):
      if v.ref() in self.managed_variable_refs:
        if 'kernel' in v.name:
          res.append(g * tf.cast(self.owner[v.ref()] == seq_idx, tf.float32))
        else:
          res.append(
            g * (1. - tf.cast(self.freeze_biases_and_normalization, tf.float32)))
      else:
        res.append(g)
    return res

  def prune(self, prune_perc, seq_idx):
    for ref, owner in self.owner.items():
      v = ref.deref()
      vals = v[owner == seq_idx]
      vals = tf.sort(tf.abs(vals))
      threshold_index = tf.cast(
        tf.cast(tf.shape(vals)[0], tf.float32) * prune_perc, tf.int32)
      threshold = vals[threshold_index]
      keep_mask = (tf.abs(v) > threshold) | (owner != seq_idx)
      v.assign(v * tf.cast(keep_mask, tf.float32))
      owner.assign(owner * tf.cast(keep_mask, tf.int32) +
                   (seq_idx + 1) * tf.cast(~keep_mask, tf.int32))

  def set_view(self, seq_idx):
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

  def set_freeze_biases_and_normalization(self, value):
    self.freeze_biases_and_normalization.assign(value)
