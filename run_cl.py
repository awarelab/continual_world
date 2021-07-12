import argparse
from envs import get_single_env, get_cl_env
from methods.vcl import VclMlpActor
from spinup.models import MlpActor, MlpCritic, PopArtMlpCritic
from spinup.utils.logx import EpochLogger
from spinup.sac import sac
from task_lists import task_seq_to_task_list
from utils.utils import get_activation_from_str, sci2int, str2bool


def main(
    logger, tasks, task_list, seed, steps_per_task, replay_size, batch_size,
    hidden_sizes, buffer_type, reset_buffer_on_task_change,
    reset_optimizer_on_task_change, activation, use_layer_norm, scale_reward,
    div_by_return, lr, alpha, use_popart, cl_method, packnet_retrain_steps,
    regularize_critic, cl_reg_coef, vcl_first_task_kl, vcl_variational_ln,
    episodic_mem_per_task, episodic_batch_size, reset_critic_on_task_change,
    randomization, multihead_archs, hide_task_id, clipnorm, gamma,
    target_output_std, packnet_fake_num_tasks, agent_policy_exploration,
    critic_reg_coef):

  assert (tasks is None) != (task_list is None)
  if tasks is not None:
    tasks = task_seq_to_task_list[tasks]
  else:
    tasks = task_list
  train_env = get_cl_env(tasks, steps_per_task, scale_reward, div_by_return,
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

  if cl_method == 'vcl':
    actor_cl = VclMlpActor
    actor_kwargs['variational_ln'] = vcl_variational_ln
  else:
    actor_cl = MlpActor

  sac(train_env, test_envs, logger,
      seed=seed, steps=steps, replay_size=replay_size,
      batch_size=batch_size, actor_cl=actor_cl, actor_kwargs=actor_kwargs,
      critic_cl=critic_cl, critic_kwargs=critic_kwargs, buffer_type=buffer_type,
      reset_buffer_on_task_change=reset_buffer_on_task_change,
      reset_optimizer_on_task_change=reset_optimizer_on_task_change,
      lr=lr, alpha=alpha, cl_method=cl_method, cl_reg_coef=cl_reg_coef,
      packnet_retrain_steps=packnet_retrain_steps,
      regularize_critic=regularize_critic, vcl_first_task_kl=vcl_first_task_kl,
      episodic_mem_per_task=episodic_mem_per_task,
      episodic_batch_size=episodic_batch_size,
      reset_critic_on_task_change=reset_critic_on_task_change,
      clipnorm=clipnorm, gamma=gamma, target_output_std=target_output_std,
      packnet_fake_num_tasks=packnet_fake_num_tasks,
      agent_policy_exploration=agent_policy_exploration,
      critic_reg_coef=critic_reg_coef)

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
      '--buffer_type', type=str, default='fifo',
      choices=['fifo', 'reservoir'],
      help='Strategy of inserting examples into the buffer'
  )

  parser.add_argument(
      '--reset_buffer_on_task_change', type=str2bool, default=True)
  parser.add_argument(
      '--reset_optimizer_on_task_change', type=str2bool, default=True)
  parser.add_argument(
      '--reset_critic_on_task_change', type=str2bool, default=False)
  parser.add_argument(
      '--activation', type=str, default='lrelu')
  parser.add_argument(
      '--use_layer_norm', type=str2bool, default=True)
  parser.add_argument(
      '--scale_reward', type=str2bool, default=False)
  parser.add_argument(
      '--div_by_return', type=str2bool, default=False)
  parser.add_argument(
      '--lr', type=float,
      default=1e-3)
  parser.add_argument(
      '--alpha', default='auto')
  parser.add_argument(
      '--use_popart', type=str2bool, default=False)
  parser.add_argument(
      '--cl_method', type=str,
      choices=[None, 'l2', 'ewc', 'mas', 'vcl', 'packnet', 'agem'],
      default=None)
  parser.add_argument(
      '--packnet_retrain_steps', type=int, default=0)
  parser.add_argument(
      '--regularize_critic', type=str2bool, default=False)
  parser.add_argument(
      '--cl_reg_coef', type=float, default=0.)
  parser.add_argument(
      '--vcl_first_task_kl', type=str2bool, default=True)
  parser.add_argument(
      '--vcl_variational_ln', type=str2bool, default=False)
  parser.add_argument(
      '--episodic_mem_per_task', type=int, default=0)
  parser.add_argument(
      '--episodic_batch_size', type=int, default=0)
  parser.add_argument(
      '--randomization', type=str, default='random_init_all')
  parser.add_argument(
      '--multihead_archs', type=str2bool, default=True)
  parser.add_argument(
      '--hide_task_id', type=str2bool, default=True)
  parser.add_argument(
      '--clipnorm', type=float, default=None)
  parser.add_argument(
      '--gamma', type=float, default=0.99)
  parser.add_argument(
      '--target_output_std', type=float, default=0.089)
  parser.add_argument(
      '--packnet_fake_num_tasks', type=int, default=None)
  parser.add_argument(
      '--agent_policy_exploration', type=str2bool, default=False)
  parser.add_argument(
      '--critic_reg_coef', type=float, default=1.)
  return parser.parse_args()


if __name__ == '__main__':
  args = vars(get_parser())
  logger = EpochLogger(args['logger_output'], config=args)
  del args['logger_output']
  main(logger, **args)
