from mrunner.helpers.specification_helper import create_experiments_helper

config = {
  'steps_per_task': int(2e6),
  'replay_size': int(1e6),
  'hidden_sizes': [256, 256, 256, 256],
  'activation': 'lrelu',
  'use_layer_norm': True,
  'tasks': 'NEW_EASY5_V0',
  'div_by_return': False,
}

params_grid = {
  'seed': list(range(5)),
  'scale_reward': [False, True],
  'batch_size': [128, 256, 512],
  'lr': [3e-4, 1e-3, 3e-3],
  'alpha': [0.1, 0.2, 0.4],
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
