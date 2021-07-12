from copy import deepcopy

CL_DEFAULTS = dict(
  tasks=None,
  task_list=None,
  seed=0,
  steps_per_task=int(1e6),
  replay_size=int(1e6),
  batch_size=128,
  hidden_sizes=[256, 256, 256, 256],
  buffer_type='fifo',
  reset_buffer_on_task_change=True,
  reset_optimizer_on_task_change=True,
  reset_critic_on_task_change=False,
  activation='lrelu',
  use_layer_norm=True,
  scale_reward=False,
  div_by_return=False,
  lr=1e-3,
  alpha='auto',
  use_popart=False,
  cl_method=None,
  packnet_retrain_steps=0,
  regularize_critic=False,
  cl_reg_coef=0.,
  vcl_first_task_kl=True,
  vcl_variational_ln=False,
  episodic_mem_per_task=0,
  episodic_batch_size=0,
  randomization='random_init_all',
  multihead_archs=True,
  hide_task_id=True,
  clipnorm=None,
  gamma=0.99,
  target_output_std=0.089,
  packnet_fake_num_tasks=None,
  agent_policy_exploration=False,
  critic_reg_coef=1.,
)

MT_DEFAULTS = dict(
  tasks=None,
  task_list=None,
  seed=0,
  steps_per_task=int(1e6),
  replay_size=int(1e6),
  batch_size=128,
  hidden_sizes=[256, 256, 256, 256],
  activation='lrelu',
  use_layer_norm=True,
  scale_reward=False,
  div_by_return=False,
  lr=1e-3,
  alpha='auto',
  use_popart=True,
  randomization='random_init_all',
  multihead_archs=True,
  hide_task_id=False,
  gamma=0.99,
  target_output_std=0.089,
)

SINGLE_DEFAULTS = dict(
  seed=0,
  steps=int(1e6),
  replay_size=int(1e6),
  batch_size=128,
  hidden_sizes=[256, 256, 256, 256],
  activation='lrelu',
  use_layer_norm=True,
  lr=1e-3,
  alpha='auto',
  use_popart=False,
  randomization='random_init_all',
  gamma=0.99,
  target_output_std=0.089,
)


def combine_config_with_defaults(config, run_kind):
  if run_kind == 'cl':
    res = deepcopy(CL_DEFAULTS)
  elif run_kind == 'mt':
    res = deepcopy(MT_DEFAULTS)
  elif run_kind == 'single':
    res = deepcopy(SINGLE_DEFAULTS)
  else:
    assert False, 'bad run_kind!'

  for k, v in config.items():
    res[k] = v

  return res
