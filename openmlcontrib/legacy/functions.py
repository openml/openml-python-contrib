import ConfigSpace
import numpy as np


def is_integer_hyperparameter(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> bool:
    """
    Checks whether hyperparameter is one of the following: Integer hyperparameter,
    Constant Hyperparameter with integer value or Unparameterized Hyperparameter
    with integer value

    Parameters
    ----------
    hyperparameter: ConfigSpace.hyperparameters.Hyperparameter
        The hyperparameter to check

    Returns
    -------
    bool
        True iff the hyperparameter complies with the definition above, false
        otherwise
    """
    if isinstance(hyperparameter, ConfigSpace.hyperparameters.IntegerHyperparameter):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant) \
            and isinstance(hyperparameter.value, int):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter) \
            and isinstance(hyperparameter.value, int):
        return True
    return False


def is_boolean_hyperparameter(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> bool:
    """
    Checks whether hyperparameter is one of the following: Categorical
    hyperparameter with only boolean values, Constant Hyperparameter with
    boolean value or Unparameterized Hyperparameter with boolean value

    Parameters
    ----------
    hyperparameter: ConfigSpace.hyperparameters.Hyperparameter
        The hyperparameter to check

    Returns
    -------
    bool
        True iff the hyperparameter complies with the definition above, false
        otherwise
    """
    if isinstance(hyperparameter, ConfigSpace.hyperparameters.CategoricalHyperparameter) \
            and np.all([isinstance(choice, bool) for choice in hyperparameter.choices]):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant) \
            and isinstance(hyperparameter.value, bool):
        return True
    elif isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter) \
            and isinstance(hyperparameter.value, bool):
        return True
    return False
