import ConfigSpace
import numpy as np
import typing


def get_active_hyperparameters(configuration_space: ConfigSpace.ConfigurationSpace,
                               key_values: typing.Dict[str, typing.Union[str,int,float]]) -> typing.Set[str]:
    """
    Returns a set of hyperparameters that are considered active

    Parameters
    ----------
    configuration_space: ConfigSpace.ConfigurationSpace
        The configuration space (needed for determining dependencies)

    key_values: dict
        A dict with key values

    Returns
    ----------
    active_parameters: set
        A set with all hyperparameter names that are considered active
    """
    vector = np.ndarray((len(configuration_space._hyperparameters),), dtype=np.float)
    for hyperparameter in configuration_space.get_hyperparameters():
        name = hyperparameter.name
        vector[configuration_space._hyperparameter_idx[name]] = hyperparameter._inverse_transform(key_values[name])
    active_parameters = set()

    for hp_name, hyperparameter in configuration_space._hyperparameters.items():
        conditions = configuration_space._parent_conditions_of[hyperparameter.name]

        active = True
        for condition in conditions:
            parent_vector_idx = condition.get_parents_vector()
            if any([vector[i] != vector[i] for i in parent_vector_idx]):
                active = False
                break

            else:
                if not condition.evaluate_vector(vector):
                    active = False
                    break

        if active:
            active_parameters.add(hp_name)
    return active_parameters
