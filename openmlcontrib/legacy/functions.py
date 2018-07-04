import ConfigSpace
import numpy as np
import typing


def interpret_hyperparameter_as_string(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> bool:
    """
    Whether the ConfigSpace hyperparameter values should be interpreted in string format. This happens for categorical
    features (even when the values are all numeric) and sometimes for UnParameterized or Constants.

    Parameters
    ----------
    hyperparameter: ConfigSpace.hyperparameters.Hyperparameter
        The hyperparameter to inspect

    Returns
    ----------
    bool
        Whether the hyperparameter values should be interpreted as strings
    """
    if isinstance(hyperparameter, ConfigSpace.CategoricalHyperparameter):
        return True
    if isinstance(hyperparameter, ConfigSpace.UnParametrizedHyperparameter):
        if isinstance(hyperparameter.value, str):
            return True
    return False


def get_active_hyperparameters(configuration_space: ConfigSpace.ConfigurationSpace,
                               key_values: typing.Dict[str, typing.Union[str, int, float, bool]]) -> typing.Set[str]:
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
        # note that the openml-python json sometimes accidentally mistakes strings for bools or numeric values
        value_to_search = str(key_values[name]) if interpret_hyperparameter_as_string(hyperparameter) \
            else key_values[name]
        value = hyperparameter._inverse_transform(value_to_search)
        vector[configuration_space._hyperparameter_idx[name]] = value
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
