from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]
config = {
  'tasks': 'DOUBLE_PMO1',
}
config = combine_config_with_defaults(config, run_kind)

params_grid = [
  {
    'seed': list(range(20)),
    'cl_method': ['packnet'],
    'packnet_retrain_steps': [100000],
    'clipnorm': [1e-5, 2e-5, 4e-5, 6e-5, 1e-4],
    'packnet_fake_num_tasks': [40],
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
