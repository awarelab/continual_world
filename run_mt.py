from typing import Callable, Iterable, List

from continualworld.envs import get_mt_env, get_single_env
from continualworld.sac.models import MlpCritic, PopArtMlpCritic
from continualworld.sac.sac import SAC
from continualworld.sac.utils.logx import EpochLogger
from continualworld.tasks import TASK_SEQS
from continualworld.utils.utils import get_activation_from_str
from input_args import mt_parse_args


def main(
    logger: EpochLogger,
    tasks: str,
    task_list: List[str],
    seed: int,
    steps_per_task: int,
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
    use_popart: bool,
    popart_beta: float,
    multihead_archs: bool,
    hide_task_id: bool,
):
    assert (tasks is None) != (task_list is None)
    if tasks is not None:
        tasks = TASK_SEQS[tasks]
    else:
        tasks = task_list

    train_env = get_mt_env(tasks, steps_per_task)
    # Consider normalizing test envs in the future.
    num_tasks = len(tasks)
    test_envs = [
        get_single_env(task, one_hot_idx=i, one_hot_len=num_tasks) for i, task in enumerate(tasks)
    ]
    steps = steps_per_task * len(tasks)

    num_heads = num_tasks if multihead_archs else 1
    actor_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        use_layer_norm=use_layer_norm,
        num_heads=num_heads,
        hide_task_id=hide_task_id,
    )
    critic_kwargs = dict(
        hidden_sizes=hidden_sizes,
        activation=get_activation_from_str(activation),
        use_layer_norm=use_layer_norm,
        num_heads=num_heads,
        hide_task_id=hide_task_id,
    )
    if use_popart:
        assert multihead_archs, "PopArt works only in the multi-head setup"
        critic_cl = PopArtMlpCritic
        critic_kwargs["beta"] = popart_beta
    else:
        critic_cl = MlpCritic

    sac = SAC(
        train_env,
        test_envs,
        logger,
        seed=seed,
        steps=steps,
        log_every=log_every,
        replay_size=replay_size,
        batch_size=batch_size,
        actor_kwargs=actor_kwargs,
        critic_cl=critic_cl,
        critic_kwargs=critic_kwargs,
        reset_buffer_on_task_change=False,
        lr=lr,
        alpha=alpha,
        gamma=gamma,
        target_output_std=target_output_std,
    )
    sac.run()


if __name__ == "__main__":
    args = vars(mt_parse_args())
    logger = EpochLogger(args["logger_output"], config=args, group_id=args["group_id"])
    del args["group_id"]
    del args["logger_output"]
    main(logger, **args)
