from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]
config = {
}
config = combine_config_with_defaults(config, run_kind)

params_grid = [
  {
    'seed': list(range(10)),
    'tasks': ['NEW_EASY5_V0', 'MT10', 'MICHAL10_V0'],
    'cl_method': [None],
    'reset_critic_on_task_change': [True],
  },
  {
    'seed': list(range(10)),
    'tasks': ['NEW_EASY5_V0', 'MT10', 'MICHAL10_V0'],
    'cl_method': ['packnet'],
    'packnet_retrain_steps': [100000],
    'reset_critic_on_task_change': [True],
  },
  {
    'seed': list(range(10)),
    'tasks': ['NEW_EASY5_V0', 'MT10', 'MICHAL10_V0'],
    'cl_method': ['l2'],
    'cl_reg_coef': [100.],
    'reset_critic_on_task_change': [True],
  },
  {
    'seed': list(range(10)),
    'tasks': ['NEW_EASY5_V0', 'MT10', 'MICHAL10_V0'],
    'cl_method': ['ewc'],
    'cl_reg_coef': [2500.],
    'reset_critic_on_task_change': [True],
  },
  {
    'seed': list(range(10)),
    'tasks': ['NEW_EASY5_V0', 'MT10', 'MICHAL10_V0'],
    'cl_method': ['mas'],
    'cl_reg_coef': [200.],
    'reset_critic_on_task_change': [False, True],
  },
  {
    'seed': list(range(10)),
    'tasks': ['NEW_EASY5_V0', 'MT10', 'MICHAL10_V0'],
    'cl_method': ['vcl'],
    'cl_reg_coef': [0.001],
    'vcl_first_task_kl': [False],
    'vcl_variational_ln': [False],
    'reset_critic_on_task_change': [False, True],
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
