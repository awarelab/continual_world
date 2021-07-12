from mrunner.helpers.specification_helper import create_experiments_helper

config = {
  'steps_per_task': int(2e6),
  'replay_size': int(1e6),
  'batch_size': 256,
}

params_grid = {
  'seed': list(range(5)),
  'tasks': ['NEW_EASY5_V0', 'NEW_EASY5_REV_V0'],
  'hidden_sizes': [
    [100, 100],
    [256, 256],
    [400, 400],
    [800, 800],
    [100, 100, 100],
    [256, 256, 256],
    [400, 400, 400],
    [800, 800, 800],
    [100, 100, 100, 100],
    [256, 256, 256, 256],
  ],
  'use_layer_norm': [False, True],
  'activation': ['tanh', 'relu', 'elu', 'lrelu'],
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
