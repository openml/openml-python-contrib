import arff
import argparse
import logging
import openml
import openmlcontrib
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_id', type=str, default='OpenML100')
    parser.add_argument('--raise_missing_task', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--drop_missing', action='store_true')
    parser.add_argument('--impute_nan_value', type=float, default=-1.0)
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml')

    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    args = parse_args()
    study = openml.study.get_study(args.study_id, 'tasks')
    df = openmlcontrib.meta.get_tasks_qualities_as_dataframe(task_ids=study.tasks, 
                                                             normalize=args.normalize, 
                                                             impute_nan_value=args.impute_nan_value,
                                                             drop_missing=args.drop_missing, 
                                                             raise_missing_task=args.raise_missing_task)
    df = df.reset_index()
    df = df.rename(columns={'index': 'task_id'})

    arff_dict = openmlcontrib.meta.dataframe_to_arff(df, 'meta-features', None)
    output_file = os.path.join(args.output_directory, 'metafeatures.arff')
    os.makedirs(args.output_directory, exist_ok=True)
    with open(output_file, 'w') as fp:
        arff.dump(arff_dict, fp)
    logging.info('Saved result to: %s' % output_file)
