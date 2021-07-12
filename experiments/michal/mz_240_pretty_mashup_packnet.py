from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from utils.utils import get_script_command

run_kind = 'cl'
name = globals()['script'][:-3]
config = {
  'steps_per_task': int(1e6),
  'randomization': 'random_init_all',
  'hide_task_id': True,
  'multihead_archs': True,
}
config = combine_config_with_defaults(config, run_kind)

params_grid = {
  'seed': list(range(10)),
  'cl_method': ['packnet'],
  'packnet_retrain_steps': [100000],
  'tasks': ['PRETTY_MASHUP_ORD_1', 'PRETTY_MASHUP_ORD_2',
            'PRETTY_MASHUP_ORD_3'],
  'clipnorm': [None, 1e-3, 1e-4, 1e-5]
}

experiments_list = create_experiments_helper(
  experiment_name=name,
  project_name='pmtest/continual-learning',
  script=get_script_command(run_kind),
  python_path='.',
  tags=[name],
  base_config=config,
  params_grid=params_grid)
