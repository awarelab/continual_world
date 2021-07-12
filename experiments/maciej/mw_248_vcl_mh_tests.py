from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'alpha': 'auto',
  'steps_per_task': int(2e3),
  'cl_method': 'vcl',
  'cl_reg_coef': 5e-5,
  'task_list': ['hammer-v1', 'stick-pull-v1'],
}
config = combine_config_with_defaults(config, run_kind)

params_grid = {
  'seed': list(range(1)),
  'multihead_archs': [False, True],
  'hide_task_id': [False, True],
  'vcl_first_task_kl': [True, False],
  'vcl_variational_ln': [True, False],
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
