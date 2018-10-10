import ConfigSpace
import ConfigSpace.hyperparameters
import logging
import numpy as np
import openml
import openmlcontrib
import os
import pandas as pd
import pickle
import sklearn
import typing
import warnings


def _merge_setup_dict_and_evaluation_dicts(
        setups: typing.Dict[int, openml.setups.OpenMLSetup],
        flow: openml.flows.OpenMLFlow,
        configuration_space: ConfigSpace.ConfigurationSpace,
        evaluations: typing.Dict[str, typing.Dict[int, openml.evaluations.OpenMLEvaluation]]) \
        -> typing.Dict[int, typing.Dict]:
    # returns a dict, mapping from setup id to a dict containing all
    # hyperparameters and evaluation measures
    setup_evaluations = dict()
    for measure in evaluations:
        # evaluations[measure] is a dict mapping from run id to evaluation
        # we can assume that all results are on the same task, so setup is the determining key
        # we will reindex this setup_evaluations[measure] to map from a setup id to evaluation measure
        setup_keys = [eval.setup_id for eval in evaluations[measure].values()]
        task_keys = [eval.task_id for eval in evaluations[measure].values()]
        if len(set(task_keys)) != 1:
            # this should never happen
            raise KeyError('Found multiple task keys in the result set for measure %s' % measure)
        if set(setup_keys) != set(setups.keys()):
            # this should also never happen, and hints at either a bug in setup
            # listing or evaluation listing not complete
            raise KeyError('Setup keys do not align for measure %s' % measure)
        setup_evaluations[measure] = {eval.setup_id: eval for eval in evaluations[measure].values()}
        if len(setup_evaluations[measure]) != len(evaluations[measure]):
            raise KeyError('Lengths of reindexed dict does not comply with old length. ')

    result = dict()
    for setup in setups.values():
        if setup.flow_id != flow.flow_id:
            # this should never happen
            raise ValueError('Setup and flow do not align.')
        setup_dict = openmlcontrib.setups.setup_to_parameter_dict(setup=setup,
                                                                  flow=flow,
                                                                  map_library_names=True,
                                                                  configuration_space=configuration_space)
        for measure in evaluations:
            setup_dict[measure] = setup_evaluations[measure][setup.setup_id].value
        result[setup.setup_id] = setup_dict
    return result


def get_task_flow_results_as_dataframe(task_id: int, flow_id: int,
                                       num_runs: int, raise_few_runs: bool,
                                       configuration_space: ConfigSpace.ConfigurationSpace,
                                       evaluation_measures: typing.List[str],
                                       cache_directory: typing.Union[str, None]) -> pd.DataFrame:
    """
    Obtains a number of runs from a given flow on a given task, and returns a
    (relevant) set of parameters and performance measures. Makes solely use of
    listing functions.

    Parameters
    ----------
    task_id: int
        The task id
    flow_id:
        The flow id
    num_runs: int
        Maximum on the number of runs per task
    configuration_space: ConfigurationSpace
        Determines valid parameters and ranges. These will be returned as
        column names
    evaluation_measures: List[str]
        A list with the evaluation measure to obtain
    cache_directory: optional, str
        Directory where cache files can be stored to or obtained from
    raise_few_runs: bool
        Raises an error if not enough runs are found according to the
        `num_runs` argument

    Returns
    -------
    df : pd.DataFrame
        a dataframe with as columns the union of the config_space
        hyperparameters and the evaluation measures, and num_runs rows.
    """
    for measure in evaluation_measures:
        if measure in configuration_space.get_hyperparameters():
            raise ValueError('measure shadows name in hyperparameter list: %s' % measure)
    # both cache paths will be set to a value if cache_directory is not None
    evaluations_cache_path = dict()
    setups_cache_path = None

    # decides the files where the cache will be stored
    if cache_directory is not None:
        cache_flow_task = os.path.join(cache_directory, str(flow_id), str(task_id))
        os.makedirs(cache_flow_task, exist_ok=True)

        for measure in evaluation_measures:
            evaluations_cache_path[measure] = os.path.join(cache_flow_task,
                                                           'evaluations_%s_%d.pkl' % (measure, num_runs))
        setups_cache_path = os.path.join(cache_flow_task, 'setups_%d.pkl' % num_runs)

    # downloads (and caches, if allowed) the evaluations for all measures.
    evaluations = dict()
    setup_ids = set()  # list maintaining all used setup ids
    for measure in evaluation_measures:
        if cache_directory is None or not os.path.isfile(evaluations_cache_path[measure]):
            # downloads (and caches, if allowed) num_runs random evaluations
            evals_current_measure = openml.evaluations.list_evaluations(measure,
                                                                        size=num_runs,
                                                                        task=[task_id],
                                                                        flow=[flow_id])
            if len(evals_current_measure) < num_runs and raise_few_runs:
                raise ValueError('Not enough evaluations for measure: %s. '
                                 'Required: %d, Got: %d' % (measure, num_runs,
                                                            len(evals_current_measure)))
            if cache_directory is not None and len(evals_current_measure) == num_runs:
                # important to only store cache if enough runs were obtained
                with open(evaluations_cache_path[measure], 'wb') as fp:
                    pickle.dump(evals_current_measure, fp)
            evaluations[measure] = evals_current_measure
        else:
            # obtains the evaluations from cache
            with open(evaluations_cache_path[measure], 'rb') as fp:
                evaluations[measure] = pickle.load(fp)
        if len(evaluations[measure]) == 0:
            raise ValueError('No results on Task %d measure %s according to these criteria' % (task_id, measure))
        for eval in evaluations[measure].values():
            setup_ids.add(eval.setup_id)

    # downloads (and caches, if allowed) the setups that belong to the evaluations
    if cache_directory is None or not os.path.isfile(setups_cache_path):
        setups = openmlcontrib.setups.obtain_setups_by_ids(setup_ids=list(setup_ids))

        if cache_directory is not None and len(evaluations) == num_runs:
            # important to only store cache if enough runs were obtained
            with open(setups_cache_path, 'wb') as fp:
                pickle.dump(setups, fp)
    else:
        # obtains the setups from cache
        with open(setups_cache_path, 'rb') as fp:
            setups = pickle.load(fp)

    # download flows. Note that only one flow is allowed, per definition
    flows = dict()
    for setup in setups.values():
        if flow_id not in flows:
            flows[setup.flow_id] = openml.flows.get_flow(setup.flow_id)
    if len(flows) != 1:
        # This should never happen.
        raise ValueError('Expected exactly one flow. Got %d' % len(flows))

    # initiates the dataframe object
    relevant_parameters = configuration_space.get_hyperparameter_names()
    all_columns = list(relevant_parameters) + evaluation_measures
    df = pd.DataFrame(columns=all_columns)

    # initiates all records. Note that we need to check them one by one before
    # we can add them to the dataframe
    setups_merged = _merge_setup_dict_and_evaluation_dicts(setups,
                                                           flows[flow_id],
                                                           configuration_space,
                                                           evaluations)
    # adds the applicable setups to the dataframe
    for setup_id, setup_merged in setups_merged.items():
        # the setups dict still contains the setup objects
        current_setup = setups[setup_id]
        if openmlcontrib.setups.setup_in_config_space(current_setup,
                                                      flows[current_setup.flow_id],
                                                      configuration_space):
            df = df.append(setup_merged, ignore_index=True)
        else:
            warnings.warn('Setup does not comply to configuration space: %s ' % setup_id)

    all_numeric_columns = list(evaluation_measures)
    for param in configuration_space.get_hyperparameters():
        if isinstance(param, ConfigSpace.hyperparameters.NumericalHyperparameter):
            all_numeric_columns.append(param.name)

    df[all_numeric_columns] = df[all_numeric_columns].apply(pd.to_numeric)

    if df.shape[0] > num_runs:
        # this should never happen
        raise ValueError('Too many runs. Expected %d got %d' % (num_runs, df.shape[0]))
    exp_params = len(relevant_parameters) + len(evaluation_measures)
    if df.shape[1] != exp_params:
        # this should never happen
        raise ValueError('Wrong number of attributes. Expected %d got %d' % (exp_params, df.shape[1]))

    df = df.reindex(sorted(df.columns), axis=1)

    return df


def get_task_flow_results_per_fold_as_dataframe(task_id: int, flow_id: int,
                                                num_runs: int, raise_few_runs: bool,
                                                configuration_space: ConfigSpace.ConfigurationSpace,
                                                evaluation_measures: typing.List[str]) -> pd.DataFrame:
    """
    Obtains a number of runs from a given flow on a given task, and returns a
    (relevant) set of parameters and performance measures. Because the per-fold
    performance information is not available in the listing functions, it is
    slower than the `openmlcontrib.meta.get_task_flow_results_as_dataframe`,
    however it can rely on openml-pythons native cache mechanism.

    Parameters
    ----------
    task_id: int
        The task id
    flow_id:
        The flow id
    num_runs: int
        Maximum on the number of runs per task
    configuration_space: ConfigurationSpace
        Determines valid parameters and ranges. These will be returned as
        column names
    evaluation_measures: List[str]
        A list with the evaluation measure to obtain
    raise_few_runs: bool
        Raises an error if not enough runs are found according to the
        `num_runs` argument

    Returns
    -------
    df : pd.DataFrame
        a dataframe with as columns the union of the config_space
        hyperparameters and the evaluation measures, and num_runs rows.
    """
    for measure in evaluation_measures:
        if measure in configuration_space.get_hyperparameters():
            raise ValueError('measure shadows name in hyperparameter list: %s' % measure)

    # book keeping for task parameters (num_repeats and num_folds)
    task = openml.tasks.get_task(task_id)
    repeats = int(task.estimation_procedure['parameters']['number_repeats']) \
        if 'number_repeats' in task.estimation_procedure['parameters'] else 1
    folds = int(task.estimation_procedure['parameters']['number_folds']) \
        if 'number_folds' in task.estimation_procedure['parameters'] else 1

    if repeats * folds <= 1:
        raise ValueError('Can only obtain per fold frame if the task defines '
                         'multiple repeats or folds. Instead, you are adviced '
                         'to use get_task_flow_results_as_dataframe().')

    # obtain all runs
    runs = openml.runs.list_runs(size=num_runs, task=[task_id], flow=[flow_id])
    if len(runs) < num_runs and raise_few_runs:
        raise ValueError('Not enough evaluations. Required: %d, '
                         'Got: %d' % (num_runs, len(runs)))
    all_records = list()
    for run_dict in runs.values():
        setup = openml.setups.get_setup(run_dict['setup_id'])
        flow = openml.flows.get_flow(setup.flow_id)
        if openmlcontrib.setups.setup_in_config_space(setup,
                                                      flow,
                                                      configuration_space):
            setup = openmlcontrib.setups.setup_to_parameter_dict(setup=setup,
                                                                 flow=flow,
                                                                 map_library_names=True,
                                                                 configuration_space=configuration_space)
            run = openml.runs.get_run(run_dict['run_id'])
            if run.fold_evaluations is None or len(run.fold_evaluations) == 0:
                warnings.warn('Skipping run that is not processed yet: %d' % run.run_id)
                continue

            for repeat_nr in range(repeats):
                for fold_nr in range(folds):
                    current_record = dict(setup)
                    current_record['repeat_nr'] = repeat_nr
                    current_record['fold_nr'] = fold_nr
                    for measure in evaluation_measures:
                        current_record[measure] = run.fold_evaluations[measure][repeat_nr][fold_nr]
                    all_records.append(current_record)

    # initiates the dataframe object
    relevant_parameters = configuration_space.get_hyperparameter_names()
    all_columns = list(relevant_parameters) + evaluation_measures + ['repeat_nr', 'fold_nr']
    df = pd.DataFrame(columns=all_columns, data=all_records)
    return df


def get_task_result_as_dataframe(task_ids: typing.List[int], flow_id: int,
                                 num_runs: int, per_fold: bool, raise_few_runs: bool,
                                 configuration_space: ConfigSpace.ConfigurationSpace,
                                 evaluation_measures: typing.List[str],
                                 normalize: bool,
                                 cache_directory: str) -> pd.DataFrame:
    """
    Obtains a number of runs from a given flow on a set of tasks, and returns a
    (relevant) set of parameters and performance measures. As backend, it uses
    either `get_task_flow_results_as_dataframe` (fast, one result per run) or
    `get_task_flow_results_perfold_as_dataframe` (slow, but results per fold).

    Parameters
    ----------
    task_ids: List[int]
        The task ids
    flow_id:
        The flow id
    num_runs: int
        Maximum on the number of runs per task
    per_fold: bool
        Whether to obtain all results per repeat and per fold (slower)
    raise_few_runs: bool
        Raises an error if not enough runs are found according to the
        `num_runs` argument
    configuration_space: ConfigurationSpace
        Determines valid parameters and ranges. These will be returned as
        column names
    evaluation_measures: List[str]
        A list with the evaluation measure to obtain
    normalize: bool
        Whether to normalize the measures per task to interval [0,1]
    cache_directory: optional, str
        Directory where cache files can be stored to or obtained from. Only
        relevant when per_fold is True

    Returns
    -------
    df : pd.DataFrame
        a dataframe with as columns the union of the config_space
        hyperparameters and the evaluation measures, and num_runs rows.
    """
    setup_data_all = None
    scaler = sklearn.preprocessing.MinMaxScaler()
    for task_id in task_ids:
        logging.info('Currently processing task %d' % task_id)
        try:
            if per_fold:
                setup_data = get_task_flow_results_as_dataframe(task_id=task_id,
                                                                flow_id=flow_id,
                                                                num_runs=num_runs,
                                                                raise_few_runs=False,
                                                                configuration_space=configuration_space,
                                                                evaluation_measures=evaluation_measures,
                                                                cache_directory=cache_directory)

            else:
                setup_data = get_task_flow_results_per_fold_as_dataframe(task_id=task_id,
                                                                         flow_id=flow_id,
                                                                         num_runs=num_runs,
                                                                         raise_few_runs=False,
                                                                         configuration_space=configuration_space,
                                                                         evaluation_measures=evaluation_measures)
        except ValueError as e:
            warnings.warn('Problem in Task %d: %s' % (task_id, str(e)))
            continue
        setup_data['task_id'] = task_id
        if setup_data_all is None:
            setup_data_all = setup_data
        else:
            if list(setup_data.columns.values) != list(setup_data_all.columns.values):
                raise ValueError('Columns per task result do not match')
            if normalize:
                for measure in evaluation_measures:
                    setup_data[[measure]] = scaler.fit_transform(setup_data[[measure]])

            setup_data_all = pd.concat((setup_data_all, setup_data))
    return setup_data_all


def dataframe_to_arff(dataframe, relation, description):
    attributes = []
    for idx, column_name in enumerate(dataframe.columns.values):
        if np.issubdtype(dataframe[column_name].dtype, np.number):
            attributes.append((column_name, 'NUMERIC'))
        else:
            values = dataframe[column_name].unique()
            attributes.append((column_name, [str(value) for value in values]))

    arff_dict = dict()
    arff_dict['data'] = dataframe.values
    arff_dict['attributes'] = attributes
    arff_dict['description'] = description
    arff_dict['relation'] = relation

    return arff_dict


def arff_to_dataframe(liacarff):
    num_keys = {'numeric', 'real'}
    expected_keys = {'data', 'attributes', 'description', 'relation'}
    if liacarff.keys() != expected_keys:
        raise ValueError('liacarff object does not contain correct keys.')
    data_ = np.array(liacarff['data'])
    arff_dict = {
        col_name: pd.Series(data_[:, idx],
                            dtype=np.float64
                            if str(col_type).lower() in num_keys
                            else np.dtype(object))
        for idx, (col_name, col_type) in enumerate(liacarff['attributes'])
    }
    return pd.DataFrame(arff_dict)
