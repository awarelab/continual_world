from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'alpha': 0.4,
  'steps_per_task': int(2e3),
  'cl_method': 'agem',
  'episodic_mem_per_task': 10000,
  'episodic_batch_size': 256,
}
config = combine_config_with_defaults(config, run_kind)

params_grid = {
  'seed': list(range(2)),
  'tasks': ['NEW_EASY5_V0', 'MT10'],

}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)