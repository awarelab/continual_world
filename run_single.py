from typing import Callable, Iterable

from continualworld.envs import get_single_env, get_task_name
from continualworld.sac.sac import SAC
from continualworld.sac.utils.logx import EpochLogger
from continualworld.utils.utils import get_activation_from_str
from input_args import single_parse_args


def main(
    logger: EpochLogger,
    task: str,
    seed: int,
    steps: int,
    log_every: int,
    replay_size: int,
    batch_size: int,
    hidden_sizes: Iterable[int],
    activation: Callable,
    use_layer_norm: bool,
    lr: float,
    gamma: float,
    alpha: str,
    target_output_std: float,
):
    actor_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        use_layer_norm=use_layer_norm,
    )
    critic_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        use_layer_norm=use_layer_norm,
    )

    sac = SAC(
        get_single_env(task),
        [get_single_env(task)],
        logger,
        seed=seed,
        steps=steps,
        log_every=log_every,
        replay_size=replay_size,
        batch_size=batch_size,
        actor_kwargs=actor_kwargs,
        critic_kwargs=critic_kwargs,
        lr=lr,
        alpha=alpha,
        gamma=gamma,
        target_output_std=target_output_std,
    )
    sac.run()


if __name__ == "__main__":
    args = vars(single_parse_args())
    args["task"] = get_task_name(args["task"])
    logger = EpochLogger(args["logger_output"], config=args, group_id=args["group_id"])
    del args["group_id"]
    del args["logger_output"]
    main(logger, **args)
