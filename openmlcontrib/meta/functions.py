import json
import openml
import openmlcontrib
import pandas as pd
import pickle
import os
import typing


def get_task_flow_results_as_dataframe(task_id: int, flow_id: int, num_runs: int,
                                       relevant_parameters: typing.Dict[str, str],
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
    relevant_parameters: dict
        Set with the parameter names. These will be returned as column names
    evaluation_measure:
        Evaluation measure to obtain
    cache_directory: str or None
        Directory where cache files can be stored to or obtained from

    Returns
    -------
    dataframe : a pandas dataframe
    """

    if 'y' in relevant_parameters.keys():
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
        if len(evaluations) < num_runs:
            raise ValueError('Not enough evaluations. Required: %d, Got: %d' % (num_runs, len(evaluations)))
        if cache_directory is not None:
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

    setup_parameters = {}

    for setup_id, setup in setups.items():
        setup_parameters[setup_id] = openmlcontrib.setups.setup_to_parameter_dict(setup,
                                                                                  'parameter_name',
                                                                                  set(relevant_parameters.keys()))

    all_columns = list(relevant_parameters.keys())
    all_columns.append('y')
    dataframe = pd.DataFrame(columns=all_columns)

    def complies_to_definition(evaluation):
        currentXy = {}
        legalConfig = True
        for idx, param in enumerate(relevant_parameters):
            value = json.loads(setup_parameters[evaluation.setup_id][param])
            if relevant_parameters[param] == 'numeric':
                if not (isinstance(value, int) or isinstance(value, float)):
                    legalConfig = False

            currentXy[param] = value

        currentXy['y'] = evaluation.value
        return currentXy, legalConfig

    for run_id, evaluation in evaluations.items():
        currentXy, legalConfig = complies_to_definition(evaluation)

        if legalConfig:
            dataframe = dataframe.append(currentXy, ignore_index=True)
        else:
            # sometimes, a numeric param can contain string values.
            # TODO: determine what to do with these. Raise Value, add or skip
            print('skipping', currentXy)

    all_numeric_columns = list(['y'])
    for parameter, datatype in relevant_parameters.items():
        if datatype == 'numeric':
            all_numeric_columns.append(parameter)

    dataframe[all_numeric_columns] = dataframe[all_numeric_columns].apply(pd.to_numeric)

    if dataframe.shape[0] > num_runs:
        raise ValueError()
    if dataframe.shape[1] != len(relevant_parameters) + 1:  # plus 1 for y data
        raise ValueError()

    dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)

    return dataframe
