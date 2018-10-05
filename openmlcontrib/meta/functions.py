import ConfigSpace
import ConfigSpace.hyperparameters
import numpy as np
import openml
import openmlcontrib
import pandas as pd
import pickle
import os
import typing


def get_task_flow_results_as_dataframe(task_id: int, flow_id: int, num_runs: int, raise_few_runs: bool,
                                       configuration_space: ConfigSpace.ConfigurationSpace, parameter_field: str,
                                       evaluation_measure: str, cache_directory: typing.Union[str, None]):
    """
    Obtains a number of runs from a given flow on a given task, and returns a (relevant) set of parameters

    Parameters
    ----------
    task_id: int
        The task id
    flow_id:
        The flow id
    num_runs: int
        Maximum on the number of runs per task
    configuration_space: ConfigurationSpace
        Determines valid parameters and ranges. These will be returned as column names
    parameter_field: str
        the key field in the parameter object that should be selected. Use full_name in order to avoid collisions; a
        good alternative is the use of parameter_name
    evaluation_measure:
        Evaluation measure to obtain
    cache_directory: str or None
        Directory where cache files can be stored to or obtained from
    raise_few_runs: bool
        Raises an error if not enough runs are found according to the `num_runs` argument

    Returns
    -------
    dataframe : a pandas dataframe
    """

    if 'y' in configuration_space.get_hyperparameters():
        raise ValueError('y is a column name for the evaluation measure')

    if cache_directory is not None:
        cache_flow_task = os.path.join(cache_directory, str(flow_id), str(task_id))
        os.makedirs(cache_flow_task, exist_ok=True)

        evaluations_cache_path = os.path.join(cache_flow_task, 'evaluations_%s_%d.pkl' % (evaluation_measure, num_runs))
        setups_cache_path = os.path.join(cache_flow_task, 'setups_%d.pkl' % num_runs)

    if cache_directory is None or not os.path.isfile(evaluations_cache_path):
        # downloads (and caches, if allowed) num_runs random evaluations
        evaluations = openml.evaluations.list_evaluations(evaluation_measure,
                                                          size=num_runs, task=[task_id], flow=[flow_id])
        if len(evaluations) < num_runs and raise_few_runs:
            raise ValueError('Not enough evaluations. Required: %d, Got: %d' % (num_runs, len(evaluations)))
        if cache_directory is not None and len(evaluations) == num_runs:
            with open(evaluations_cache_path, 'wb') as fp:
                pickle.dump(evaluations, fp)
    else:
        # obtains the evaluations from cache
        with open(evaluations_cache_path, 'rb') as fp:
            evaluations = pickle.load(fp)

    if cache_directory is None or not os.path.isfile(setups_cache_path):
        # downloads (and caches, if allowed) the setups that belong to the evaluations
        setup_ids = []
        for run_id, evaluation in evaluations.items():
            setup_ids.append(evaluation.setup_id)
        setups = openmlcontrib.setups.obtain_setups_by_ids(setup_ids=setup_ids)

        if cache_directory is not None:
            with open(setups_cache_path, 'wb') as fp:
                pickle.dump(setups, fp)
    else:
        # obtains the setups from cache
        with open(setups_cache_path, 'rb') as fp:
            setups = pickle.load(fp)

    relevant_parameters = configuration_space.get_hyperparameter_names()
    all_columns = list(relevant_parameters)
    all_columns.append('y')
    dataframe = pd.DataFrame(columns=all_columns)

    for run_id, evaluation in evaluations.items():
        current_setup = setups[evaluation.setup_id]
        if openmlcontrib.setups.setup_in_config_space(current_setup, configuration_space):
            current_setup_as_dict = openmlcontrib.setups.setup_to_parameter_dict(current_setup,
                                                                                 parameter_field,
                                                                                 configuration_space)
            current_setup_as_dict['y'] = evaluation.value
            dataframe = dataframe.append(current_setup_as_dict, ignore_index=True)
        else:
            # sometimes, a numeric param can contain string values.
            # TODO: determine what to do with these. Raise Value, add or skip
            print('skipping setup', evaluation.setup_id)

    all_numeric_columns = list(['y'])
    for param in configuration_space.get_hyperparameters():
        if isinstance(param, ConfigSpace.hyperparameters.NumericalHyperparameter):
            all_numeric_columns.append(param.name)

    dataframe[all_numeric_columns] = dataframe[all_numeric_columns].apply(pd.to_numeric)

    if dataframe.shape[0] > num_runs:
        raise ValueError()
    if dataframe.shape[1] != len(relevant_parameters) + 1:  # plus 1 for y data
        raise ValueError()

    dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)

    return dataframe


def dataframe_to_arff(dataframe, relation, description):
    attributes = []
    for idx, column_name in enumerate(dataframe.columns.values):
        if np.issubdtype(dataframe[column_name].dtype, np.number):
            attributes.append((column_name, 'NUMERIC'))
        else:
            values = dataframe[column_name].unique()
            attributes.append((column_name, [str(value) for value in values]))

    arff_dict = dict()
    arff_dict['data'] = dataframe.as_matrix()
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
    arff_dict = {col_name: pd.Series(data_[:, idx], dtype=np.float64 if str(col_type).lower() in num_keys else object)
                 for idx, (col_name, col_type) in enumerate(liacarff['attributes'])}
    return pd.DataFrame(arff_dict)
