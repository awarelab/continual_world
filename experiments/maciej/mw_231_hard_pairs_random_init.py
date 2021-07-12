from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from task_lists import task_seq_to_task_list
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'randomization': 'random_init_all',
  'reset_critic_on_task_change': False,
}
config = combine_config_with_defaults(config, run_kind)

# All 100 possibilities, with diagonal
hard10 = task_seq_to_task_list['HARDEST10_V0']
task_lists = [[first, second] for first in hard10 for second in hard10]

params_grid = {
  'seed': list(range(20)),
  'task_list': task_lists,
  'hide_task_id': [True],
  'multihead_archs': [True],
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
