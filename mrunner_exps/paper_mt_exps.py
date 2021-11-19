from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "mt",
    "tasks": "CW10",
    "steps_per_task": 2_000_000,
    "lr": 1e-4,
    "logger_output": ["tsv", "neptune"],
}
config = combine_config_with_defaults(config)

params_grid = {
    "seed": list(range(20)),
    "use_popart": [False, True],
}

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="pmtest/continual-learning",
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name, "v6"],
    base_config=config,
    params_grid=params_grid,
)
