# This script searches for a given OpenML (Weka) measure which sklearn measure
# is the same
import argparse
import openml
import sklearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=6)  # Arbitrary, in this case Letter
    parser.add_argument('--flow_id', type=int, default=8353)  # Arbitrary, in this case Pipeline with SVC
    parser.add_argument('--num_runs', type=int, default=100)  # Number of runs to inspect
    parser.add_argument('--openml_measure', type=str, default='f_measure')
    parser.add_argument('--sklearn_metric', type=str, default='f1_score')

    args_ = parser.parse_args()

    return args_


SKLEARN_AVERAGE_FUNCTIONS = ['micro', 'macro', 'weighted']


def run_script():
    args = parse_args()
    runs = openml.runs.list_runs(size=args.num_runs,
                                 flow=[args.flow_id],
                                 task=[args.task_id])
    if len(runs) != args.num_runs:
        raise ValueError('Obtained too few runs: %d' % len(runs))

    # ideally, we would like to use the openml scores per fold (to remove bias from averaging),
    # but these are not always calculated for all measures
    differences = {avg_fn: list() for avg_fn in SKLEARN_AVERAGE_FUNCTIONS}
    for run_id in runs:
        run = openml.runs.get_run(run_id)
        openml_score = run.evaluations[args.openml_measure]
        for avg_fn in differences:
            kwargs = {'average': avg_fn}
            score = run.get_metric_fn(getattr(sklearn.metrics, args.sklearn_metric), kwargs)
            score_avg = sum(score) / len(score)
            difference = openml_score - score_avg
            differences[avg_fn].append(difference)

    differences_squared = {avg_fn: sum(map(lambda x: x*x, diffs)) for avg_fn, diffs in differences.items()}
    differences_max = {avg_fn: max(diffs) for avg_fn, diffs in differences.items()}

    # lower is better in both cases
    print('squared', differences_squared)
    print('max', differences_max)


if __name__ == '__main__':
    run_script()
