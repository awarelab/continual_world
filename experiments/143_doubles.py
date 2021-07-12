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
    'seed': list(range(20)),
    'tasks': ['DOUBLE_MASHUP_V0', 'DOUBLE_MT10'],
    'hide_task_id': [True],
    'multihead_archs': [True],
    'steps_per_task': [int(1e6)],
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
