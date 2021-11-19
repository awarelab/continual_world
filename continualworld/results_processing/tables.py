from collections import defaultdict

import numpy as np
import pandas as pd

from continualworld.results_processing.utils import (
    get_individual_task_success_columns,
    get_task_num_to_name,
)


class BootstrapCI:
    def __init__(self, X, Y, num_bootstrap, confidence, statistics, ranges, seed):
        """Vanilla bootstrap CIs. This approach has some flaws, but is flexible.
        Short info about arguments (no proper docstring)
        X = volume num_seeds x T
        Y = volume num_seeds x T
        num_bootstrap = number of bootstrap samples (e.g. 4000)
        confidence = 0.9
        statistics = {'ft': fun, ....}
        num_bootstrap = number of bootstrap samples
        ranges = dict{"[0]": range(0, 1), ...}
        """
        self._X = X
        self._Y = Y
        self._num_bootstrap = num_bootstrap
        self._ranges = ranges
        self._statistics = statistics
        self._confidence = confidence
        self._significance = (1.0 - confidence) / 2.0
        if seed is not None:
            np.random.seed(seed)
        volumeX = np.expand_dims(self._X, axis=0)
        volumeY = np.expand_dims(self._Y, axis=0)
        self._original_data_metrics = self._data_statistics(volumeX, volumeY)

    @property
    def original_data_metrics(self):
        return self._original_data_metrics

    def _data_statistics(self, volumeX, volumeY):
        statistics = defaultdict(dict)
        for name, statistic in self._statistics.items():
            for range_name, range_ in self._ranges.items():
                metric = self._apply_statistics(volumeX, volumeY, statistic, range_)
                statistics[name].update({range_name: metric.item()})
        return statistics

    def _bootstrap(self, volumeX, volumeY):
        """X is a volume num_seedsX x T, Y is a volume num_seedsY x T."""
        output = {}
        ii = np.random.choice(len(volumeX), (self._num_bootstrap, len(volumeX)))
        output["X"] = volumeX[ii]
        jj = np.random.choice(len(volumeY), (self._num_bootstrap, len(volumeY)))
        output["Y"] = volumeY[jj]
        return output

    def _apply_statistics(self, volumeX, volumeY, fun, range_):
        # apply fun over seeds dimension, axis=1
        return np.mean([fun(volumeX[..., i], volumeY[..., i], 1).flatten() for i in range_], axis=0)

    def ci(self):
        """
        returns {"statistic0": {"range0": (lb, ub), "range1": (lb, ub), ...}, ...}
        """
        # bootstrap_result = {"statistic 0": {"slice 0": (lb, ub), "slice 1": (lb, ub), ...}, ...}
        bootstrap_result = defaultdict(dict)
        # boostrap_sample = {"X": list(volumeX), "Y": list(volumeY)}
        # where volumeX = num_seedsX x T, volumeY = num_seedsY x T
        bootstrap_sample = self._bootstrap(self._X, self._Y)
        volumeX = bootstrap_sample["X"]
        volumeY = bootstrap_sample["Y"]
        for name, statistic in self._statistics.items():
            for range_name, range_ in self._ranges.items():
                metric = self._apply_statistics(volumeX, volumeY, statistic, range_)
                lb = np.quantile(metric, self._significance, interpolation="midpoint")
                ub = np.quantile(metric, 1 - self._significance, interpolation="midpoint")
                bootstrap_result[name].update({range_name: (lb, ub)})
        return bootstrap_result


def ft(x, y, a):
    return np.mean(x, axis=a) - np.mean(y, axis=a)


def normalized_ft(x, y, a):
    return (np.mean(x, axis=a) - np.mean(y, axis=a)) / (1 - np.mean(y, axis=a))


statistics = {
    "ft": ft,
    "normalized_ft": normalized_ft,
}


def calculate_data_at_the_end(data, index_columns):
    data1 = data.copy()
    data1 = data1.sort_values(by=["x"])

    data_at_the_end = data1

    data_at_the_end = data_at_the_end.groupby(index_columns).tail(5)
    data_at_the_end = data_at_the_end.groupby(index_columns).mean()

    columns = get_individual_task_success_columns(data)
    data_at_the_end = data_at_the_end[columns]

    return data_at_the_end


# by PM, + cosmetic changes
def calculate_forgetting_individual(data, columns_to_add=["cl_method"]):
    index_columns = columns_to_add + ["experiment_id"]

    data_at_the_end = calculate_data_at_the_end(data, index_columns)

    data = data.copy()
    partial_data = []
    for env in sorted(data["train/active_env"].unique()):
        #         if np.isnan(env):
        #             continue
        env = int(env)
        env_indices = data["train/active_env"] == env
        current_col = data.columns[data.columns.str.contains(f"test/stochastic/{env}/.*/success")][
            0
        ]
        env_data = data.loc[env_indices, [current_col, "x", "train/active_env"] + index_columns]
        env_data = env_data.sort_values(by=["x"])
        env_data = env_data.groupby(index_columns).tail(5)
        env_data = env_data.groupby(index_columns).mean()
        partial_data.append(env_data)

    # success rate at the end of training given env
    success_rate_of_active_env = (
        pd.concat(partial_data).groupby(index_columns).max()
    )  # max is just to flatten
    # Drop uninformative leftovers
    success_rate_of_active_env = success_rate_of_active_env.drop(["x", "train/active_env"], axis=1)

    individual_forgetting = success_rate_of_active_env - data_at_the_end

    return data_at_the_end, success_rate_of_active_env, individual_forgetting


def calculate_forward_transfer(data, baseline_data, normalize=True):
    data = data.copy()

    task_num_to_name = get_task_num_to_name(data)
    steps_per_task = int(data.steps_per_task.unique())

    long_baseline = []
    for env in sorted(data["train/active_env"].unique()):
        #         if np.isnan(env): continue
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

    long_baseline = pd.concat(long_baseline)

    # correct for double seeds
    #     unique_exps = data.groupby(['experiment_id', 'seed'], as_index=False).size()
    #     target_size = 500 if cw10 else 1000
    #     unique_exps = unique_exps[unique_exps['size'] == target_size].drop_duplicates(subset='seed', keep="last")['experiment_id']
    #     data = data[data['experiment_id'].isin(unique_exps)].reset_index()
    # display(data.groupby(['experiment_id', 'seed'], as_index=False).size())

    data = (
        data.drop("x", axis=1)
        .groupby(["train/active_env", "experiment_id"])["current_success"]
        .mean()
        .reset_index()
    )
    long_baseline = (
        long_baseline.drop("x", axis=1)
        .groupby(["train/active_env", "experiment_id"])["current_success"]
        .mean()
        .reset_index()
    )

    # ugly
    X = data.pivot(
        index="experiment_id", columns="train/active_env", values="current_success"
    ).to_numpy()
    Y = long_baseline.pivot(
        index="experiment_id", columns="train/active_env", values="current_success"
    ).reset_index(drop=True)
    Y = pd.DataFrame({c: Y[c].dropna().values for c in Y.columns}).to_numpy()

    T = X.shape[1]

    ranges = {f"[{i}]": range(i, i + 1) for i in range(T)}
    ranges.update(
        {
            f"[{0}:{T // 2}]": range(0, T // 2),
            f"[{T // 2}:{T}]": range(T // 2, T),
            f"[{0}:{T}]": range(0, T),
        }
    )

    BCI = BootstrapCI(
        X=X, Y=Y, num_bootstrap=4000, confidence=0.9, statistics=statistics, ranges=ranges, seed=0
    )
    CIs = BCI.ci()

    ci_result = defaultdict(list)

    for env in sorted(data["train/active_env"].unique()):
        ci_result["train/active_env"].append(env)

        for name in statistics.keys():
            lb, ub = CIs[name][f"[{int(env)}]"]
            m = BCI.original_data_metrics[name][f"[{int(env)}]"]

            ci_result[f"lower_bound_{name}"].append(lb)
            ci_result[f"upper_bound_{name}"].append(ub)
            ci_result[f"CI_{name}"].append(f"{m:.2f} [{lb:.2f}, {ub:.2f}]")

            lbfh, ubfh = CIs[name][f"[{0}:{T // 2}]"]
            lbsh, ubsh = CIs[name][f"[{T // 2}:{T}]"]
            lbt, ubt = CIs[name][f"[{0}:{T}]"]
            mfh = BCI.original_data_metrics[name][f"[{0}:{T // 2}]"]
            msh = BCI.original_data_metrics[name][f"[{T // 2}:{T}]"]
            mt = BCI.original_data_metrics[name][f"[{0}:{T}]"]

            ci_result[f"lb_first_half_{name}"].append(lbfh)
            ci_result[f"ub_first_half_{name}"].append(ubfh)
            ci_result[f"CI_first_half_{name}"].append(f"{mfh:.2f} [{lbfh:.2f}, {ubfh:.2f}]")

            ci_result[f"lb_second_half_{name}"].append(lbsh)
            ci_result[f"ub_second_half_{name}"].append(ubsh)
            ci_result[f"CI_second_half_{name}"].append(f"{msh:.2f} [{lbsh:.2f}, {ubsh:.2f}]")

            ci_result[f"lb_total_{name}"].append(lbt)
            ci_result[f"ub_total_{name}"].append(ubt)
            ci_result[f"CI_total_{name}"].append(f"{mt:.2f} [{lbt:.2f}, {ubt:.2f}]")
            ci_result[f"total_{name}"].append(mt)

    ci_result = pd.DataFrame(ci_result)

    # We have all the data inside - best place for confidence interval analysis :)
    data = data.merge(long_baseline, on="train/active_env", suffixes=("", "_baseline"))
    data = data.groupby("train/active_env").mean()
    data = data.merge(ci_result, on="train/active_env")

    data["ft"] = data["current_success"] - data["current_success_baseline"]
    data["normalized_ft"] = data["ft"] / (1 - data["current_success_baseline"])
    #     for name in statistics.keys():
    #         data[f'{name}_CI'] = data[f'{name}'].map('{:.2f}'.format) + " " + data[f'CI_{name}']

    return data


def compute_mean_and_ci(data, metric_name, columns, group_by=["cl_method"]):
    data = data.copy()
    metric = data[columns].mean(axis=1)
    metric = metric.groupby(group_by)

    result = []
    for cl_method, data_chunk in metric:
        x = np.expand_dims(data_chunk, axis=1)
        y = np.asarray([0])
        BCI = BootstrapCI(
            X=x,
            Y=y,
            num_bootstrap=4000,
            confidence=0.9,
            statistics={"mean": lambda x, y, a: np.mean(x, axis=1)},
            ranges={0: [0]},
            seed=0,
        )
        ci = BCI.ci()["mean"]
        result.append(
            pd.DataFrame(
                [[cl_method, x.mean(), ci[0][0], ci[0][1]]],
                columns=["cl_method", metric_name, rf"lb_{metric_name}", rf"ub_{metric_name}"],
            )
        )
    result = pd.concat(result)

    result.set_index(group_by, inplace=True)

    return result


def calculate_metrics(data, mtl_data, baseline_data, group_by=["cl_method"], methods_order=None):
    if methods_order is None:
        methods_order = []
    data_at_the_end, _, individual_forgetting = calculate_forgetting_individual(
        data, columns_to_add=group_by
    )

    #     active_envs_columns = [test_columns_by_index[int(env)] for env in sorted(data['train/active_env'].dropna().unique())]
    active_envs_columns = get_individual_task_success_columns(data)

    average_performance = compute_mean_and_ci(data_at_the_end, "performance", active_envs_columns)
    average_forgetting = compute_mean_and_ci(
        individual_forgetting, "forgetting", active_envs_columns
    )

    forward_transfer = []
    for cl_method, data_chunk in data.groupby(group_by):
        ft_res = calculate_forward_transfer(data_chunk, baseline_data)
        cols = ["total_normalized_ft", "lb_total_normalized_ft", "ub_total_normalized_ft"]
        ft_res = ft_res[cols][:1]
        ft_res["cl_method"] = cl_method
        forward_transfer.append(ft_res)
    forward_transfer = pd.concat(forward_transfer)
    forward_transfer.set_index(group_by, inplace=True)

    result = pd.concat([average_performance, average_forgetting, forward_transfer], axis=1)

    # Compute performance metric for MTL
    mtl_data_at_the_end = calculate_data_at_the_end(
        mtl_data, index_columns=group_by + ["experiment_id"]
    )
    #     mtl_active_envs_columns = [test_columns_by_index[int(env)] for env in list(range(10))]
    mtl_active_envs_columns = get_individual_task_success_columns(mtl_data)
    mtl_ap = compute_mean_and_ci(mtl_data_at_the_end, "performance", mtl_active_envs_columns)

    result = pd.concat([result, mtl_ap])

    methods_order = [m for m in methods_order if m in result.index]
    result = pd.concat([result.loc[methods_order], result.loc[~result.index.isin(methods_order)]])

    return result
