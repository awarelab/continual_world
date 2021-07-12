from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]
config = {
  'steps_per_task': int(1e6),
  'randomization': 'random_init_all',
  'hide_task_id': True,
  'multihead_archs': True,
  'tasks': 'PRETTY_MASHUP_ORD_1',
}
config = combine_config_with_defaults(config, run_kind)

params_grid = {
  'seed': list(range(10)),
  'batch_size': [128, 256, 512],
  'lr': [3e-4, 1e-4, 3e-5, 1e-3],
  'gamma': [0.95, 0.99, 0.995],
  'target_output_std': [0.089, 0.03, 0.3],
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
