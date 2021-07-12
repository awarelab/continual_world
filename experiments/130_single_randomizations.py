from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'single'
name = globals()['script'][:-3]

config = {}
config = combine_config_with_defaults(config, run_kind)

params_grid = {
  'seed': list(range(5)),
  'task': list(range(10)),
  'randomization': ['deterministic', 'random_init_all', 'random_init_fixed20',
                    'random_init_small_box'],
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
