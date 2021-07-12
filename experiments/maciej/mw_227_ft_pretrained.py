from mrunner.helpers.specification_helper import create_experiments_helper

from task_lists import task_seq_to_task_list
from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'single'
name = globals()['script'][:-3]

config = {
    'pretrained_tag': 'mw_226_single_hard'
}
config = combine_config_with_defaults(config, run_kind)

task_list = task_seq_to_task_list['HARDEST10_V0']

params_grid = {
  'seed': list(range(30, 31)),
  'task': task_list,
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
