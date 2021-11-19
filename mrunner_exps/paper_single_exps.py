from mrunner.helpers.specification_helper import create_experiments_helper

from continualworld.tasks import TASK_SEQS
from mrunner_utils import combine_config_with_defaults

name = globals()["script"][:-3]
config = {
    "run_kind": "single",
    "logger_output": ["tsv", "neptune"],
}
config = combine_config_with_defaults(config)

params_grid = {
    "seed": list(range(100)),
    "task": TASK_SEQS["CW10"],
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
