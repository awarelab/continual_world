from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'mt'
name = globals()['script'][:-3]
config = {
  'tasks': 'PRETTY_MASHUP_ORD_1',
}
config = combine_config_with_defaults(config, run_kind)

params_grid = {
  'seed': list(range(10)),
  'use_popart': [False, True],
  'batch_size': [128, 256, 512],
  'lr': [3e-4, 1e-4, 3e-5, 1e-3],
}


experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
