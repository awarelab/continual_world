import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from continualworld.results_processing.utils import get_task_num_to_name

DEV_MODE = True

PALETTE = sns.color_palette("deep")
sns.set_palette(PALETTE)

RENAMER = {
    "cl_method=finetuning": "Fine-tuning",
    "cl_method=ewc": "EWC",
    "cl_method=mas": "MAS",
    "cl_method=l2": "L2",
    "cl_method=vcl": "VCL",
    "cl_method=agem": "A-GEM",
    "cl_method=packnet": "PackNet",
    "cl_method=reservoir": "Perfect Memory",
    "cl_method=mtl": "Multi-Task",
    "cl_method=mtl_popart": "Multi-Task + PopArt",
}


# m_cols = [col for col in first_half.columns if '/success' in col]
# METRIC_RENAMER = {m: m.strip('test/stochastic/').strip('/success') for m in m_cols}


def apply_filters(exp_df, filters_list):
    filtered = []
    for filter_dict in filters_list:
        indices = [True] * len(exp_df)
        for key, val in filter_dict.items():
            indices = indices & (exp_df[key] == val)
        filtered += [exp_df[indices]]
    return pd.concat(filtered)


def get_steps_per_task(data):
    cl_data = data[~data.cl_method.str.contains("mtl")]
    return int(cl_data.steps_per_task.unique())


def plot_task_separators(data, steps_per_task, ax=None, special_at_end=False, only_special=False):
    if ax == None:
        ax = plt
    left = int(data["x"].min()) // steps_per_task * steps_per_task
    right = int(data["x"].max())
    middle = (left + right) // 2
    special = right if special_at_end else middle
    iter_right_end = right + 1 if special_at_end else right
    for x in range(left, iter_right_end, steps_per_task):
        if x == special:
            ax.plot([x, x], [0.0, 1], "-.", linewidth=1.5, color="gray")
        elif not only_special:
            ax.plot([x, x], [0.0, 1], "--", linewidth=0.5, color="gray")


def smoothen_long_seq(data, smooth_col="average_success", window=5):
    for exp_id in data["experiment_id"].unique():
        has_active_env = (
            "train/active_env" in data.columns
            and len(data.loc[data["experiment_id"] == exp_id, "train/active_env"].dropna()) > 0
        )
        if has_active_env:
            for active_env in data["train/active_env"].unique():
                filtered_indices = (data["experiment_id"] == exp_id) & (
                    data["train/active_env"] == active_env
                )
                data.loc[filtered_indices, smooth_col] = (
                    data.loc[filtered_indices, smooth_col].rolling(window, min_periods=1).mean()
                )
        else:
            # multi-task experiments
            filtered_indices = data["experiment_id"] == exp_id
            data.loc[filtered_indices, smooth_col] = (
                data.loc[filtered_indices, smooth_col].rolling(window, min_periods=1).mean()
            )
    return data


def plot_long_seq(
    data,
    ax,
    method_name,
    lineplot_kwargs=None,
    manual_average=True,
    smoothen=False,
    title="Average Success",
):
    if lineplot_kwargs is None:
        lineplot_kwargs = {}

    data = data.copy()
    if manual_average:
        cols = data.columns[data.columns.str.contains(f"test/stochastic/.*/success", regex=True)]
        data["average_success"] = data[cols].mean(axis=1)
        avg_success = data
    else:
        avg_success = data.rename(columns={"test/stochastic/average_success": "average_success"})
    if smoothen:
        avg_success = smoothen_long_seq(avg_success)

    avg_success["Average Success"] = avg_success["average_success"]
    avg_success["Method"] = avg_success["identifier"]
    avg_success = avg_success.replace({"Method": RENAMER})

    style = "half" if "half" in data.columns else None
    sns.lineplot(
        data=avg_success,
        ax=ax,
        x="x",
        y="Average Success",
        hue="Method",
        style=style,
        **lineplot_kwargs,
    )

    plot_task_separators(data, steps_per_task=get_steps_per_task(data), ax=ax)
    ax.set_title(f"{title}, {method_name}")


def plot_current(data, ax, method_name, lineplot_kwargs=None, smoothen=False):
    if lineplot_kwargs is None:
        lineplot_kwargs = {}

    data = data.copy()
    for env in sorted(data["train/active_env"].dropna().unique()):
        env = int(env)
        env_indices = data["train/active_env"] == env
        current_col = data.columns[data.columns.str.contains(f"test/stochastic/{env}/.*/success")][
            0
        ]
        data.loc[env_indices, "current_success"] = data.loc[env_indices, current_col]

    if smoothen:
        data = smoothen_long_seq(data, smooth_col="current_success")

    data["Current Task Success"] = data["current_success"]
    data["Method"] = data["identifier"]
    data = data.replace({"Method": RENAMER})

    style = "half" if "half" in data.columns else None
    sns.lineplot(
        data=data,
        ax=ax,
        x="x",
        y="Current Task Success",
        hue="Method",
        style=style,
        **lineplot_kwargs,
    )

    # ax.xaxis.set_ticks_position('none')
    plot_task_separators(data, steps_per_task=get_steps_per_task(data), ax=ax)
    ax.set_title(f"Current Task Success, {method_name}")


def plot_individual(data, ax, method_name, lineplot_kwargs=None, smoothen=False):
    if lineplot_kwargs is None:
        lineplot_kwargs = {}

    data = data.copy()
    data_cols = data.columns[data.columns.str.contains(f"test/stochastic/.*/success", regex=True)]
    if smoothen:
        for col in data_cols:
            data = smoothen_long_seq(data, smooth_col=col)
    sorted_cols = []
    min_env = int(data["train/active_env"].dropna().min())
    max_env = int(data["train/active_env"].dropna().max())
    for idx in range(min_env, max_env + 1):
        sorted_cols.append(next(col for col in data_cols if f"/{idx}/" in col))

    steps_per_task = get_steps_per_task(data)
    data = data.melt(id_vars=["x", "experiment_id"], value_vars=sorted_cols)

    data["Task Success"] = data["value"]
    data["Task"] = data["variable"]
    # data = data.replace({'Task': METRIC_RENAMER})

    sns.lineplot(data=data, ax=ax, x="x", y="Task Success", hue="Task", **lineplot_kwargs)

    plot_task_separators(data, steps_per_task=steps_per_task, ax=ax)
    ax.set_title(f"Task Success, {method_name}")


def plot_forward_transfer(
    data, ax, method_name, baseline_data, lineplot_kwargs=None, normalize=True, smoothen=False
):
    steps_per_task = get_steps_per_task(data)

    if lineplot_kwargs == None:
        lineplot_kwargs = {}

    task_num_to_name = get_task_num_to_name(data)

    data = data.copy()
    long_baseline = []
    for env in sorted(data["train/active_env"].unique()):
        if np.isnan(env):
            continue
        env = int(env)
        env_name = task_num_to_name[env]

        # baseline
        current_baseline = baseline_data[baseline_data["task"] == env_name].copy()
        current_baseline["current_success"] = current_baseline[
            f"test/stochastic/0/{env_name}/success"
        ]
        current_baseline["x"] += env * steps_per_task
        current_baseline["train/active_env"] = env
        long_baseline += [current_baseline]

        # current task: update data with 'current_succes' column
        env_indices = data["train/active_env"] == env
        current_col = data.columns[
            data.columns.str.contains(f"test/stochastic/{env}/.*/success", regex=True)
        ][0]
        data.loc[env_indices, "current_success"] = data.loc[env_indices, current_col]

    # 10 taskow x 100 seedow = 1000
    long_baseline = pd.concat(long_baseline)

    if smoothen:
        data = smoothen_long_seq(data, smooth_col="current_success")
        long_baseline = smoothen_long_seq(long_baseline, smooth_col="current_success")

    long_baseline_mean = long_baseline.groupby(["x"]).mean()["current_success"].reset_index()
    transfer_mean = data.groupby(["x"]).mean()["current_success"].reset_index()
    avg_data = long_baseline_mean.merge(transfer_mean, on="x", suffixes=("_baseline", ""))
    avg_data["diff"] = avg_data["current_success"] - avg_data["current_success_baseline"]

    long_baseline["run_type"] = "Reference"
    data["run_type"] = method_name
    plot_data = pd.concat([data, long_baseline])

    plot_data["Current Task Success"] = plot_data["current_success"]
    sns.lineplot(
        data=plot_data, x="x", y="Current Task Success", hue="run_type", ax=ax, **lineplot_kwargs
    )

    ax.fill_between(
        avg_data["x"],
        avg_data["current_success_baseline"],
        avg_data["current_success"],
        where=avg_data["diff"] > 0,
        color="green",
        alpha=0.15,
    )
    ax.fill_between(
        avg_data["x"],
        avg_data["current_success"],
        avg_data["current_success_baseline"],
        where=avg_data["diff"] < 0,
        color="red",
        alpha=0.15,
    )

    # hack: remove legend title
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)

    plot_task_separators(data, steps_per_task=get_steps_per_task(data), ax=ax)
    ax.set_title(f"Forward Transfer, {method_name}")


def plot_all_methods(
    data,
    plot_fn,
    use_mtl="no",
    one_png=True,
    plot_methods_collectively=True,
    plot_methods_separately=True,
    plot_fn_kwargs=None,
    lineplot_kwargs_fn=None,
    smoothen=False,
    disable_ci=False,
    output_file=None,
):
    assert use_mtl in ["no", "only_collective", "all"]

    if plot_fn_kwargs is None:
        plot_fn_kwargs = {}

    if lineplot_kwargs_fn is None:
        lineplot_kwargs_fn = lambda _: {}

    data = data.copy()

    if use_mtl == "no":
        data = data[~data.cl_method.str.contains("mtl")]

    group_by = ["identifier"]
    if "half" in data.columns:
        group_by.append("half")

    data_groupby = data.groupby(group_by, sort=False)
    if use_mtl == "only_collective":
        data_groupby = [(k, d) for (k, d) in data_groupby if "cl_method=mtl" not in k]

    if one_png:
        num_plots = len(data_groupby) * plot_methods_separately + plot_methods_collectively
        fig, axes = plt.subplots(num_plots)
        if num_plots == 1:
            axes = [axes]
        fig.set_size_inches(22.4, 4 * num_plots)
        if plot_methods_collectively:
            plot_fn(data, axes[0], "all methods", smoothen=smoothen, **plot_fn_kwargs)
            axes[0].get_xaxis().set_label_text("")
            axes[0].grid(False, axis="x")
            axes[0].legend(loc="center right", bbox_to_anchor=(1.1, 0.5))
            axes = axes[1:]
        if plot_methods_separately:
            for i, ((groupby_values, data_chunk), ax) in enumerate(zip(data_groupby, axes)):
                lineplot_kwargs = lineplot_kwargs_fn(i)
                plot_fn(
                    data_chunk,
                    ax,
                    RENAMER[groupby_values],
                    lineplot_kwargs=lineplot_kwargs,
                    smoothen=smoothen,
                    **plot_fn_kwargs,
                )
                ax.get_xaxis().set_label_text("")
                ax.grid(False, axis="x")
                if lineplot_kwargs.get("legend", True):
                    ax.legend(loc="center right", bbox_to_anchor=(1.1, 0.5))

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)
    else:
        assert False
        # Some work needs to be done if we want to bring back this option.
        # CI = 'sd'
        # for i, (groupby_values, data_chunk) in enumerate(data_groupby):
        #     fig, ax = plt.subplots(1)
        #     fig.set_size_inches(28 * HACK_SCALING, 4 * HACK_SCALING)
        #     plot_fn(data_chunk, ax, RENAMER[groupby_values],
        #             lineplot_kwargs=lineplot_kwargs_fn(i))
        #     plt.show()


def visualize_sequence(
    data,
    mtl_data,
    baseline_data,
    group_by=None,
    show_avg=False,
    show_individual=False,
    show_current=False,
    show_ft=False,
    show_forgetting=False,
    order=None,
    separate_current=False,
    smoothen=False,
    plot_methods_collectively=True,
    plot_methods_separately=True,
    output_dir=None,
    use_ci=True,
):
    font = {"family": "sans-serif", "weight": "normal", "size": 34}

    sns.set_context("paper", font_scale=1.5)
    sns.set_style("whitegrid")

    matplotlib.rc("font", **font)
    matplotlib.rc("lines", linewidth=2.0)

    data = pd.concat([data, mtl_data])
    if group_by is None:
        group_by = ["cl_method"]

    # Sorting
    if order is None:
        data = data.sort_values(by=group_by + ["experiment_id", "x"])
    else:
        key, vals = order
        new_data = []
        for val in vals:
            new_data.append(data[data[key] == val])
        new_data.append(data[~(data[key].isin(vals))])
        data = pd.concat(new_data)

    data["identifier"] = data.apply(
        lambda x: ", ".join(f"{col}={str(x[col])}" for col in group_by), axis=1
    )

    if show_avg:
        lineplot_kwargs_fn = lambda i: {"palette": [PALETTE[i]], "ci": 95 if use_ci else None}
        output_file = os.path.join(output_dir, "average_performance.png") if output_dir else None
        plot_all_methods(
            data,
            plot_long_seq,
            use_mtl="only_collective",
            plot_methods_collectively=plot_methods_collectively,
            plot_methods_separately=plot_methods_separately,
            lineplot_kwargs_fn=lineplot_kwargs_fn,
            smoothen=smoothen,
            output_file=output_file,
        )

    if show_current:
        lineplot_kwargs_fn = lambda i: {"palette": [PALETTE[i]], "ci": 95 if use_ci else None}
        output_file = os.path.join(output_dir, "current_task.png") if output_dir else None
        plot_all_methods(
            data,
            plot_current,
            lineplot_kwargs_fn=lineplot_kwargs_fn,
            plot_methods_collectively=plot_methods_collectively,
            plot_methods_separately=plot_methods_separately,
            smoothen=smoothen,
            output_file=output_file,
        )

    if show_individual:
        lineplot_kwargs_fn = lambda _: {"legend": False, "ci": 95 if use_ci else None}
        output_file = os.path.join(output_dir, "individual_tasks.png") if output_dir else None
        plot_all_methods(
            data,
            plot_individual,
            lineplot_kwargs_fn=lineplot_kwargs_fn,
            plot_methods_collectively=False,
            plot_methods_separately=plot_methods_separately,
            smoothen=smoothen,
            disable_ci=True,
            output_file=output_file,
        )

    if show_ft:
        lineplot_kwargs_fn = lambda i: {"palette": [PALETTE[i], (0.6, 0.6, 0.6)], "ci": None}
        output_file = os.path.join(output_dir, "forward_transfer.png") if output_dir else None
        plot_all_methods(
            data,
            plot_forward_transfer,
            lineplot_kwargs_fn=lineplot_kwargs_fn,
            plot_methods_collectively=False,
            plot_methods_separately=plot_methods_separately,
            smoothen=smoothen,
            plot_fn_kwargs={"baseline_data": baseline_data},
            output_file=output_file,
        )
