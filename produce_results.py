import argparse
import os

from continualworld.results_processing.plots import visualize_sequence
from continualworld.results_processing.tables import calculate_metrics
from continualworld.results_processing.utils import METHODS_ORDER, get_data_for_runs
from continualworld.utils.utils import get_readable_timestamp, str2bool


def main(args: argparse.Namespace) -> None:
    cl_data = get_data_for_runs(args.cl_logs)
    mtl_data = get_data_for_runs(args.mtl_logs, mtl=True)
    baseline_data = get_data_for_runs(args.baseline_logs)

    timestamp = get_readable_timestamp()
    output_dir = os.path.join(args.output_path, f"report_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    table = calculate_metrics(cl_data, mtl_data, baseline_data)
    table_path = os.path.join(output_dir, "results.csv")
    table.to_csv(table_path)

    visualize_sequence(
        cl_data,
        mtl_data,
        baseline_data,
        group_by=["cl_method"],
        show_avg=True,
        show_current=True,
        show_individual=True,
        show_ft=True,
        order=("cl_method", METHODS_ORDER),
        smoothen=False,
        output_dir=output_dir,
        use_ci=args.use_ci,
    )

    print(f"Report saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cl_logs", type=str)
    parser.add_argument("--mtl_logs", type=str)
    parser.add_argument("--baseline_logs", type=str)
    parser.add_argument("--use_ci", type=str2bool, default=True)
    parser.add_argument("--output_path", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
