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
    'cl_method': ['agem'],
    'regularize_critic': [True],
    'episodic_mem_per_task': [100000, 500000],
    'episodic_batch_size': [128, 256, 512, 1024],
    'batch_size': [128, 256, 512],
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
