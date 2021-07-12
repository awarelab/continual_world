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
    'seed': list(range(10)),
    'cl_method': [None],
  },
  {
    'seed': list(range(10)),
    'cl_method': ['l2'],
    'cl_reg_coef': [1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
  },
  {
    'seed': list(range(10)),
    'cl_method': ['ewc'],
    'cl_reg_coef': [1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
  },
  {
    'seed': list(range(10)),
    'cl_method': ['mas'],
    'cl_reg_coef': [1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
  },
  {
    'seed': list(range(10)),
    'cl_method': ['agem'],
    'regularize_critic': [True],
    'episodic_mem_per_task': [10000, 100000, 500000],
    'episodic_batch_size': [128, 256, 512],
  },
  {
    'seed': list(range(10)),
    'cl_method': ['vcl'],
    'cl_reg_coef': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.],
    'vcl_first_task_kl': [False, True],
    'vcl_variational_ln': [False],
  },
  {
    'seed': list(range(10)),
    'cl_method': ['packnet'],
    'packnet_retrain_steps': [50000, 100000, 200000],
    'clipnorm': [None, 1., 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
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