from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from task_lists import task_seq_to_task_list
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'regularize_critic': False,
}
config = combine_config_with_defaults(config, run_kind)

params_grid = {
  'seed': list(range(5)),
  'hide_task_id': [True],
  'multihead_archs': [True],
  'cl_method': [None],
  'tasks': ['MT10', 'MICHAL10_V0'],
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)