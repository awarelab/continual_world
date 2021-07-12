from mrunner.helpers.specification_helper import create_experiments_helper

config = {
  'steps_per_task': int(2e6),
  'replay_size': int(1e6),
  'batch_size': 256,
  'hidden_sizes': [256, 256, 256, 256],
  'activation': 'lrelu',
  'use_layer_norm': True,
  'lr': 3e-4,
  'scale_reward': False,
  'div_by_return': False,
}

params_grid = {
  'seed': list(range(10)),
  'alpha': ['auto', 0.6, 0.8],
  'tasks': ['MT10'],
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
