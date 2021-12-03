import json
import os
import warnings
from collections import defaultdict
from glob import glob
from typing import List

import pandas as pd

METHODS_ORDER = [
    "finetuning",
    "l2",
    "ewc",
    "mas",
    "vcl",
    "packnet",
    "reservoir",
    "agem",
    "mtl",
    "mtl_popart",
]


def get_data_for_single_run(path, with_config_info=True):
    data_file = os.path.join(path, "progress.tsv")
    data = pd.read_csv(data_file, sep="\t")

    experiment_id = os.path.basename(os.path.normpath(path))
    data["experiment_id"] = experiment_id

    if with_config_info:
        config_file = os.path.join(path, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)

        for k, v in config.items():
            if isinstance(v, list):
                v = str(v)
            data[k] = v

    # Some postprocessing
    data["x"] = data["total_env_steps"]
    if "buffer_type" in data.columns:
        data.loc[data["buffer_type"] == "reservoir", "cl_method"] = "reservoir"
    if "cl_method" in data.columns:
        data["cl_method"] = data["cl_method"].fillna("finetuning")

    return data


def get_task_balanced_baseline_data(runs_data: List[pd.DataFrame]) -> List[pd.DataFrame]:
    # In our code for bootstrapping, we have a technical assumption
    # that we need the same number of runs for every baseline task.

    task_to_runs = defaultdict(list)
    for d in runs_data:
        task = d.task[0]
        task_to_runs[task].append(d)

    runs_per_task = min([len(l) for l in task_to_runs.values()])
    res = []
    for task, runs in task_to_runs.items():
        if len(runs) > runs_per_task:
            warnings.warn(f"Number of runs for baseline task {task} cut from {len(runs)} to {runs_per_task}!")
        res.extend(runs[:runs_per_task])

    return res


def get_data_for_runs(path, kind, with_config_info=True):
    run_paths = [p for p in glob(os.path.join(path, "*")) if os.path.isdir(p)]
    data = [get_data_for_single_run(p, with_config_info) for p in run_paths]
    if kind == "single":
        data = get_task_balanced_baseline_data(data)
    data = pd.concat(data)

    if kind == "mtl":
        data["cl_method"] = ""
        data.loc[data["use_popart"] == False, "cl_method"] = "mtl"
        data.loc[data["use_popart"] == True, "cl_method"] = "mtl_popart"

    return data


def get_individual_task_success_columns(df):
    return df.columns[df.columns.str.contains(f"test/stochastic/.*/success", regex=True)]


def get_task_num_to_name(df):
    task_num_to_name = {}

    for success_col in get_individual_task_success_columns(df):
        col_split = success_col.split("/")
        index = int(col_split[2])
        name = col_split[3]
        assert (
            index not in task_num_to_name.keys()
        ), "Multiple tasks with the same index - probably mixed experiments with different task sequences!"
        task_num_to_name[index] = name

    return task_num_to_name
