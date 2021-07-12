from mrunner.helpers.specification_helper import create_experiments_helper

config = {
  'steps_per_task': int(2e6),
  'replay_size': int(1e6),
  'batch_size': 256,
  'buffer_type': 'fifo',
  'reset_buffer_on_task_change': True,
  'reset_optimizer_on_task_change': False,
  'activation': 'lrelu',
  'hidden_sizes': [256, 256, 256, 256],
  'use_layer_norm': True,
  'cl_regularization': 'ewc',
  'regularize_qf': False,
}

params_grid = {
  'cl_reg_coef': [5e-1, 1e0, 5e0, 1e1, 2e1, 5e1],
  'tasks': ['NEW_EASY5_V0', 'NEW_EASY5_REV_V0', 'MT10'],
  'seed': list(range(5)),
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
