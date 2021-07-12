from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from task_lists import task_seq_to_task_list
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'hide_task_id': True,
  'multihead_archs': True,
}
config = combine_config_with_defaults(config, run_kind)

task_lists = [
    ["window-close-v1", "door-open-v1", "button-press-topdown-v1"],
    ["window-close-v1", "button-press-topdown-v1", "door-open-v1"],
]

params_grid = {
  'seed': list(range(20)),
  'task_list': task_lists
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
