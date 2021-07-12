from mrunner.helpers.specification_helper import create_experiments_helper

config = {
  'steps_per_task': 2000000,
  'replay_size': 1000000,
  'batch_size': 256,
  'hidden_sizes': [256, 256],
}

params_grid = {
  'seed': list(range(10)),
  'tasks': ['MT10', 'EASY4_V0', 'NEW_EASY5_V0', 'MICHAL10_V0'],
}
name = globals()['script'][:-3]

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script='python3 run_cl.py',
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)