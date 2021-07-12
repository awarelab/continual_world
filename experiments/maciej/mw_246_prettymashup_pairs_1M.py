from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from task_lists import task_seq_to_task_list
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'randomization': 'random_init_all',
  'reset_critic_on_task_change': False,
  'hide_task_id': True,
  'multihead_archs': True,
  'steps_per_task': int(1e6),
}
config = combine_config_with_defaults(config, run_kind)

# All 100 possibilities, with diagonal
tasks = ['hammer-v1', 'push-wall-v1', 'faucet-close-v1',
         'push-back-v1', 'stick-pull-v1',
         'handle-press-side-v1', 'push-v1', 'shelf-place-v1',
         'window-close-v1', 'peg-unplug-side-v1']
pairs = [[first, second] for first in tasks for second in tasks]

params_grid = {
      'seed': list(range(10)),
      'task_list': pairs,
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
