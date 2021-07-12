from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from task_lists import task_seq_to_task_list
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'regularize_critic': False,
  'tasks': 'NEW_EASY5_V0',
  'hide_task_id': True,
  'multihead_archs': True,
}
config = combine_config_with_defaults(config, run_kind)

params_grid = [
  {
    'seed': list(range(5)),
    'cl_method': [None],
  },
  {
    'seed': list(range(5)),
    'cl_method': ['l2'],
    'cl_reg_coef': [50., 100., 200.],
  },
  {
    'seed': list(range(5)),
    'cl_method': ['ewc'],
    'cl_reg_coef': [1000., 2500., 5000.],
  },
  {
    'seed': list(range(5)),
    'cl_method': ['mas'],
    'cl_reg_coef': [10., 20., 50.],
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
