import ConfigSpace


def is_numeric_hyperparameter(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter) -> bool:
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
