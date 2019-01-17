import arff
import argparse
import logging
import json
import matplotlib.pyplot as plt
import numpy as np
import openmlcontrib
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.linear_model
import sklearn.ensemble
import os
import typing


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--performances_path', type=str,
                        default=os.path.expanduser('~') + '/projects/sklearn-bot/data/svc.arff')
    parser.add_argument('--metafeatures_path', type=str,
                        default=os.path.expanduser('~') + '/projects/sklearn-bot/data/metafeatures.arff')
    parser.add_argument('--output_directory', type=str,
                        default=os.path.expanduser('~') + '/experiments/meta-models')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--spearman_k', type=int, default=100)
    parser.add_argument('--cv_iterations', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=16)
    parser.add_argument('--random_seed', type=int, default=42)
    args_ = parser.parse_args()
    return args_


def get_dataset_metadata(dataset_path: str):
    """
    Many datasets outputted by the sklearn bot have a first comment line with
    important meta-data
    """
    with open(dataset_path) as fp:
        first_line = fp.readline()
        if first_line[0] != '%':
            raise ValueError('arff data file should start with comment for meta-data')
    meta_data = json.loads(first_line[1:])
    return meta_data


def evaluate_fold(model: sklearn.base.RegressorMixin, X_tr: np.ndarray,
                  y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                  use_k: int) -> typing.Tuple[float, float]:
    """
    Evaluates a meta or aggregate model that has been trained on data from
    different tasks
    """
    new_model = sklearn.base.clone(model)
    new_model.fit(X_tr, y_tr)
    experiments = {
        'tr': (X_tr, y_tr),
        'te': (X_te, y_te),
    }
    spearman_score = dict()
    for exp_type, (X, y) in experiments.items():
        y_hat = new_model.predict(X)
        rand_indices = np.random.randint(len(X), size=use_k)
        spearman_score[exp_type] = scipy.stats.pearsonr(y[rand_indices], y_hat[rand_indices])[0]
    return spearman_score['te'], spearman_score['tr']


def cross_validate_surrogate(model: sklearn.base.RegressorMixin, data: np.ndarray,
                             targets: np.ndarray, n_folds: int,
                             use_k: int) -> typing.Tuple[float, float]:
    """
    Cross-validates a surrogate model that is trained on some data from the same
    tasks
    """
    kf = sklearn.model_selection.KFold(n_splits=n_folds, random_state=42, shuffle=True)
    splits = kf.split(data)

    spearman_scores_te = list()
    spearman_scores_tr = list()
    for tr_idx, te_idx in splits:
        X_tr, y_tr = data[tr_idx], targets[tr_idx]
        X_te, y_te = data[te_idx], targets[te_idx]
        spearm_te, spearm_tr = evaluate_fold(model, X_tr, y_tr, X_te, y_te, use_k)
        spearman_scores_te.append(spearm_te)
        spearman_scores_tr.append(spearm_tr)

    return float(np.mean(spearman_scores_te)), float(np.mean(spearman_scores_tr))


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.info('Started %s' % os.path.basename(__file__))

    # some naming declarations
    spearman_name = 'spearmanr_%d' % args.spearman_k

    # data loading and management
    with open(args.performances_path, 'r') as fp:
        arff_performances = arff.load(fp)
    performances = openmlcontrib.meta.arff_to_dataframe(arff_performances, None)
    param_columns = get_dataset_metadata(args.performances_path)['col_parameters']
    with open(args.metafeatures_path, 'r') as fp:
        arff_metafeatures = arff.load(fp)
    # impute missing meta-features with -1 value
    metafeatures = openmlcontrib.meta.arff_to_dataframe(arff_metafeatures, None).set_index('task_id').fillna(-1)
    # join with meta-features frame, and remove tasks without meta-features
    performances = performances.join(metafeatures, on='task_id', how='inner')

    performances = performances.fillna(-1)
    # sklearn objects
    surrogate_model = sklearn.ensemble.RandomForestRegressor(n_estimators=args.n_estimators,
                                                             random_state=args.random_seed)

    # determine relevant tasks
    all_tasks = performances['task_id'].unique()

    results = []
    for idx, task_id in enumerate(all_tasks):
        logging.info('Processing task %d (%d/%d)' % (task_id, idx+1, len(all_tasks)))
        frame_task = performances.loc[performances['task_id'] == task_id]
        frame_others = performances.loc[performances['task_id'] != task_id]
        assert(frame_task.shape[0] > 100)

        #######################
        # SURROGATES          #
        #######################
        # random forest
        spearm_te, spearm_tr = cross_validate_surrogate(surrogate_model,
                                                        pd.get_dummies(frame_task[param_columns]).values,
                                                        frame_task[args.scoring].values,
                                                        args.cv_iterations,
                                                        args.spearman_k)
        results.append({'task_id': task_id, 'strategy': 'RF_surrogate', 'x_order': 31, 'set': 'train-obs', spearman_name: spearm_tr})
        results.append({'task_id': task_id, 'strategy': 'RF_surrogate', 'x_order': 30, 'set': 'test', spearman_name: spearm_te})

        #######################
        # AGGREGATES          #
        #######################
        # random forest
        spearm_te, spearm_tr = evaluate_fold(surrogate_model,
                                             pd.get_dummies(frame_others[param_columns]).values,
                                             frame_others[args.scoring].values,
                                             pd.get_dummies(frame_task[param_columns]).values,
                                             frame_task[args.scoring].values,
                                             args.spearman_k)
        results.append({'task_id': task_id, 'strategy': 'RF_aggregate', 'x_order': 11, 'set': 'test', spearman_name: spearm_te})
        results.append({'task_id': task_id, 'strategy': 'RF_aggregate', 'x_order': 10, 'set': 'train-tasks', spearman_name: spearm_tr})

        ############################
        # META-MODELS              #
        ############################

        # random forest
        columns = list(param_columns) + list(metafeatures.columns.values)
        spearm_te, spearm_tr = evaluate_fold(surrogate_model,
                                             pd.get_dummies(frame_others[columns]).values,
                                             frame_others[args.scoring].values,
                                             pd.get_dummies(frame_task[columns]).values,
                                             frame_task[args.scoring].values,
                                             args.spearman_k)
        results.append({'task_id': task_id, 'strategy': 'RF_meta', 'x_order': 21, 'set': 'test', spearman_name: spearm_te})
        results.append({'task_id': task_id, 'strategy': 'RF_meta', 'x_order': 20, 'set': 'train-tasks', spearman_name: spearm_tr})

    # x_order is used to trick seaborn plot into using the right order
    # general order: first random forest models, then quadratic models
    # secondary order: first aggregates, then meta-models, then surrogates
    # tertiary order: first train-tasks, then test, then test-obs
    result_frame = pd.DataFrame(results).sort_values(['x_order'])

    os.makedirs(args.output_directory, exist_ok=True)
    result_frame.to_csv(os.path.join(args.output_directory, 'results.csv'))

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.boxplot(x="strategy", y=spearman_name, hue="set", data=result_frame, ax=ax)
    plt.savefig(os.path.join(args.output_directory, '%s.png' % spearman_name))


if __name__ == '__main__':
    run(parse_args())
