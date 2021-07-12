from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'use_layer_norm': True,
  'regularize_critic': False,
  'alpha': 'auto',
  'tasks': 'NEW_EASY5_V0',
}
config = combine_config_with_defaults(config, run_kind)

params_grid = [
   {
       'seed': list(range(5)),
       'cl_method': ['mas'],
       'cl_reg_coef': [1., 10., 20., 50., 100., 200.],
       'reset_critic_on_task_change': [True, False],
   },
   {
       'seed': list(range(5)),
       'cl_method': ['l2'],
       'cl_reg_coef': [1e-1, 1., 10., 100., 200., 500.],
       'reset_critic_on_task_change': [True, False],
   },
]

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)