import argparse
from envs import get_single_env
from spinup.models import MlpCritic, PopArtMlpCritic
from spinup.utils.logx import EpochLogger
from spinup.sac import sac
from utils.utils import get_activation_from_str, sci2int, str2bool


def main(logger, task, seed, steps, replay_size, batch_size, hidden_sizes,
         activation, use_layer_norm, lr, alpha, use_popart, randomization,
         gamma, target_output_std):
  actor_kwargs = dict(hidden_sizes=hidden_sizes,
                      activation=get_activation_from_str(activation),
                      use_layer_norm=use_layer_norm)
  critic_kwargs = dict(hidden_sizes=hidden_sizes,
                       activation=get_activation_from_str(activation),
                       use_layer_norm=use_layer_norm)
  if use_popart:
    critic_cl = PopArtMlpCritic
  else:
    critic_cl = MlpCritic

  # Keep in mind that for now we do not normalize single envs here!
  sac(get_single_env(task, randomization=randomization),
      [get_single_env(task, randomization=randomization)], logger, seed=seed,
      steps=steps, replay_size=replay_size, batch_size=batch_size,
      actor_kwargs=actor_kwargs, critic_cl=critic_cl,
      critic_kwargs=critic_kwargs, lr=lr, alpha=alpha, gamma=gamma,
      target_output_std=target_output_std)

def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--task', type=str)
  parser.add_argument(
      '--logger_output', type=str, nargs="+", choices=['neptune', 'tensorboard', 'tsv'],
      help='Types of logger used.')
  parser.add_argument(
    '--seed', type=int, default=0)
  parser.add_argument(
    '--steps', type=sci2int, default=int(1e6))
  parser.add_argument(
    '--replay_size', type=sci2int, default=int(1e6))
  parser.add_argument(
    '--batch_size', type=int, default=128)
  parser.add_argument(
    '--hidden_sizes', type=int, nargs="+",
    default=[256, 256, 256, 256])
  parser.add_argument(
    '--activation', type=str, default='lrelu')
  parser.add_argument(
    '--use_layer_norm', type=str2bool, default=True)
  parser.add_argument(
    '--lr', type=float, default=1e-3)
  parser.add_argument(
    '--alpha', default='auto')
  parser.add_argument(
    '--use_popart', type=str2bool, default=False)
  parser.add_argument(
    '--randomization', type=str, default='random_init_all')
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
