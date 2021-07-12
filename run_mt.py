import argparse
from envs import get_single_env, get_mt_env
from spinup.models import PopArtMlpCritic, MlpCritic
from spinup.utils.logx import EpochLogger
from spinup.sac import sac
from task_lists import task_seq_to_task_list
from utils.utils import get_activation_from_str, sci2int, str2bool


def main(logger, tasks, task_list, seed, steps_per_task, replay_size, batch_size,
         hidden_sizes, activation, use_layer_norm, scale_reward, div_by_return,
         lr, alpha, use_popart, randomization, multihead_archs, hide_task_id,
         gamma, target_output_std):
  assert (tasks is None) != (task_list is None)
  if tasks is not None:
    tasks = task_seq_to_task_list[tasks]
  else:
    tasks = task_list

  train_env = get_mt_env(tasks, steps_per_task, scale_reward, div_by_return,
                         randomization=randomization)
  # Consider normalizing test envs in the future.
  num_tasks = len(tasks)
  test_envs = [get_single_env(task, one_hot_idx=i, one_hot_len=num_tasks,
                              randomization=randomization) for
               i, task in enumerate(tasks)]
  steps = steps_per_task * len(tasks)

  num_heads = num_tasks if multihead_archs else 1
  actor_kwargs = dict(hidden_sizes=hidden_sizes,
                      activation=get_activation_from_str(activation),
                      use_layer_norm=use_layer_norm, num_heads=num_heads,
                      hide_task_id=hide_task_id)
  critic_kwargs = dict(hidden_sizes=hidden_sizes,
                       activation=get_activation_from_str(activation),
                       use_layer_norm=use_layer_norm, num_heads=num_heads,
                       hide_task_id=hide_task_id)
  if use_popart:
    assert multihead_archs, 'PopArt works only in the multi-head setup'
    critic_cl = PopArtMlpCritic
  else:
    critic_cl = MlpCritic

  sac(train_env, test_envs, logger, seed=seed, steps=steps, replay_size=replay_size,
      batch_size=batch_size, actor_kwargs=actor_kwargs, critic_cl=critic_cl,
      critic_kwargs=critic_kwargs, reset_buffer_on_task_change=False,
      lr=lr, alpha=alpha, gamma=gamma, target_output_std=target_output_std)

def get_parser():
  parser = argparse.ArgumentParser(description='Continual World')
  task_group = parser.add_mutually_exclusive_group()
  task_group.add_argument(
      '--tasks', type=str, choices=task_seq_to_task_list.keys(),
      default=None,
      help='Name of the sequence you want to run')
  task_group.add_argument(
      '--task_list', nargs='+', default=None,
      help='List of tasks you want to run, by name or by the MetaWorld index')
  parser.add_argument(
      '--logger_output', type=str, nargs="+", choices=['neptune', 'tensorboard', 'tsv'],
      help='Types of logger used.')
  parser.add_argument(
      '--seed', type=int,
      help='Random seed used for running the experiments')
  parser.add_argument(
      '--steps_per_task', type=sci2int, default=int(1e6),
      help='Numer of steps per task')
  parser.add_argument(
      '--replay_size', type=sci2int, default=int(1e6),
      help='Size of the replay buffer')
  parser.add_argument(
      '--batch_size', type=int, default=128,
      help='Number of samples in each mini-batch sampled by SAC')
  parser.add_argument(
      '--hidden_sizes', type=int, nargs="+",
      default=[256, 256, 256, 256],
      help="Hidden layers sizes in the base network")
  parser.add_argument(
    '--activation', type=str, default='lrelu')
  parser.add_argument(
    '--use_layer_norm', type=str2bool, default=True)
  parser.add_argument(
    '--scale_reward', type=str2bool, default=False)
  parser.add_argument(
    '--div_by_return', type=str2bool, default=False)
  parser.add_argument(
    '--lr', type=float, default=1e-3)
  parser.add_argument(
    '--alpha', default='auto')
  parser.add_argument(
    '--use_popart', type=str2bool, default=True)
  parser.add_argument(
    '--randomization', type=str, default='random_init_all')
  parser.add_argument(
    '--multihead_archs', type=str2bool, default=True)
  parser.add_argument(
    '--hide_task_id', type=str2bool, default=False)
  parser.add_argument(
    '--gamma', type=float, default=0.99)
  parser.add_argument(
    '--target_output_std', type=float, default=0.089)
  return parser.parse_args()


if __name__ == '__main__':
  args = vars(get_parser())
  logger = EpochLogger(args['logger_output'], config=args)
  del args['logger_output']
  main(logger, **args)
