from mrunner.helpers.specification_helper import create_experiments_helper

from defaults import combine_config_with_defaults
from task_lists import task_seq_to_task_list
from utils.utils import get_script_command

# Min ['handle-press-side-v1', 'basketball-v1', 'faucet-close-v1', 'push-wall-v1', 'coffee-pull-v1', 'hammer-v1', 'stick-push-v1', 'bin-picking-v1', 'shelf-place-v1', 'sweep-v1'] -0.66


run_kind = 'cl'
name = globals()['script'][:-3]

config = {
  'randomization': 'random_init_all',
  'reset_critic_on_task_change': False,
  'hide_task_id': True,
  'multihead_archs': True,
  'steps_per_task': int(1e6),
  'task_list': ['stick-push-v1', 'shelf-place-v1', 'hammer-v1',
      'coffee-pull-v1', 'basketball-v1', 'handle-press-side-v1', 'sweep-v1',
      'push-wall-v1', 'faucet-close-v1', 'bin-picking-v1']
}
config = combine_config_with_defaults(config, run_kind)

# Max: 1.2
params_grid = [
    {
      'cl_method': [None],
      'seed': list(range(10)),
    },
    {
      'cl_method': ['ewc'],
      'seed': list(range(10)),
      'cl_reg_coef': [2500.],
    },
    {
      'seed': list(range(10)),
      'cl_method': ['mas'],
      'cl_reg_coef': [20.],
    },
    {
      'seed': list(range(10)),
      'cl_method': ['agem'],
      'regularize_critic': [True],
      'episodic_mem_per_task': [10000],
      'episodic_batch_size': [256],
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
