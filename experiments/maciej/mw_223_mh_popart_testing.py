from mrunner.helpers.specification_helper import create_experiments_helper
from defaults import combine_config_with_defaults

run_kind = 'mt'

config = {
  'steps_per_task': int(2e3),
  'replay_size': int(1e6),
  'batch_size': 256,
  'hidden_sizes': [256, 256, 256, 256],
  'activation': 'lrelu',
  'use_layer_norm': True,
  'lr': 3e-4,
  'alpha': 0.4,
  'scale_reward': True,
  'div_by_return': False,
  'tasks': 'NEW_EASY5_V0',
}
config = combine_config_with_defaults(config, run_kind)

params_grid = {
  'seed': list(range(5)),
  'multihead_archs': [True, False],
  'hide_task_id': [True, False],
  'use_popart': [True, False]
  # TODO: 1xx
}
name = globals()['script'][:-3]

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script='python3 run_mt.py',
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
