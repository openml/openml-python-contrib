import ConfigSpace


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
