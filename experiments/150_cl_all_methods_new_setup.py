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

params_grid = [
  {
    'seed': list(range(10)),
    'cl_method': [None],
    'tasks': ['MASHUP_V0', 'MT10', 'HARDEST10_V0', 'PRETTY_HARD_V0',
              'HARDEST10_HARD_ORDER', 'MASHUP_RND_ORD_1'],
  },
  # {
  #   'seed': list(range(10)),
  #   'cl_method': ['packnet'],
  #   'packnet_retrain_steps': [100000],
  # },
  {
    'seed': list(range(10)),
    'cl_method': ['l2'],
    'cl_reg_coef': [100.],
    'tasks': ['MASHUP_V0', 'MT10', 'HARDEST10_V0', 'PRETTY_HARD_V0',
              'HARDEST10_HARD_ORDER', 'MASHUP_RND_ORD_1'],
  },
  {
    'seed': list(range(10)),
    'cl_method': ['ewc'],
    'cl_reg_coef': [2500.],
    'tasks': ['MASHUP_V0', 'MT10', 'HARDEST10_V0', 'PRETTY_HARD_V0',
              'HARDEST10_HARD_ORDER', 'MASHUP_RND_ORD_1'],
  },
  {
    'seed': list(range(10)),
    'cl_method': ['mas'],
    'cl_reg_coef': [200.],
    'tasks': ['MASHUP_V0', 'MT10', 'HARDEST10_V0', 'PRETTY_HARD_V0',
              'HARDEST10_HARD_ORDER', 'MASHUP_RND_ORD_1'],
  },
  # {
  #   'seed': list(range(10)),
  #   'cl_method': ['vcl'],
  #   'cl_reg_coef': [5e-4],
  #   'vcl_first_task_kl': [False],
  #   'vcl_variational_ln': [False],
  # },
  {
    'seed': list(range(10)),
    'cl_method': ['agem'],
    'regularize_critic': [True],
    'episodic_mem_per_task': [10000],
    'episodic_batch_size': [256],
    'tasks': ['MASHUP_V0', 'MT10', 'HARDEST10_V0', 'PRETTY_HARD_V0',
              'HARDEST10_HARD_ORDER', 'MASHUP_RND_ORD_1'],
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
