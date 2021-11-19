from mrunner.helpers.specification_helper import create_experiments_helper

from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "cl",
    "steps_per_task": 2000,
    "tasks": "CW20",
    "cl_method": "ewc",
    "logger_output": ["tsv", "neptune"],
}
config = combine_config_with_defaults(config)

params_grid = {
    "seed": list(range(2)),
}

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="michalzajac/testtest",
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    base_config=config,
    params_grid=params_grid,
)
