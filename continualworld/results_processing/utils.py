import json
import os
from glob import glob

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


def get_data_for_runs(path, with_config_info=True, mtl=False):
    run_paths = [p for p in glob(os.path.join(path, "*")) if os.path.isdir(p)]
    data = [get_data_for_single_run(p, with_config_info) for p in run_paths]
    data = pd.concat(data)

    if mtl:
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
