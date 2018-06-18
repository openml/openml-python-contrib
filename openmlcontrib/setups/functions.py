import collections
import ConfigSpace
import copy
import json
import openml
import typing


def filter_setup_list(setupid_setup, param_name, min=None, max=None, allowed_values=None):
    """
    Removes setups from a dict of setups if the parameter value
    does not comply with a given value. Important: Use with
    caution, as it does not handle duplicate names for various
    modules well. Will be updated in a non-backward compatible 
    manner in a later version. 
    
     Parameters
        ----------
        setupid_setup : dict of OpenMLSetup
            As obtained from the openml.setups.list_setups fn
        
        param_name : str
            the name of the parameter which values should be restricted
        
        min : int
            setups with values below this threshold will be removed
        
        max : int
            setups with values above this threshold will be removed
        
        allowed_values : list
            list of allowed values
        
        Returns
        -------
        model : dict of OpenMLSetup
            a dict, with the setups that did not comply removed 
    """
    allowed = dict()

    for setupid, setup in setupid_setup.items():

        paramname_id = {v.parameter_name: k for k, v in setup.parameters.items()}

        if param_name not in paramname_id.keys():
            raise ValueError('Setup %d does not contain param %s' %(setupid, param_name))
        param_value = json.loads(setup.parameters[paramname_id[param_name]].value)

        # print(param_value, min, max, allowed_values)

        if min is not None:
            if param_value < min:
                continue

        if max is not None:
            if param_value > max:
                continue

        if allowed_values is not None:
            if param_value not in allowed_values:
                continue

        allowed[setupid] = copy.deepcopy(setup)

    return allowed


def obtain_setups_by_ids(setup_ids, require_all=True, limit=250):
    """
    Obtains a list of setups by id. Because the

    Parameters
    ----------
    setup_ids : list[int]
        list of setups to obtain

    require_all : bool
        if set to True, the list of requested setups is required to
        be complete (and an error is thrown if not)

    limit : int
        The number of setups to obtain per call
        (lower numbers avoid long URLs, but take longer)

    Returns
    -------
    The dict of setups
    """
    if not isinstance(setup_ids, collections.Iterable):
        raise ValueError()
    for val in setup_ids:
        if not isinstance(val, int):
            raise ValueError()

    setups = {}
    offset = 0
    setup_ids = list(setup_ids)
    while True:
        setups_batch = openml.setups.list_setups(setup=setup_ids[offset:offset+limit])
        setups.update(setups_batch)

        offset += limit
        if offset >= len(setup_ids):
            break
    if require_all:
        missing = set(setup_ids) - set(setups.keys())
        if set(setup_ids) != set(setups.keys()):
            raise ValueError('Did not get all setup ids. Missing: %s' % missing)

    return setups


def setup_to_configuration(setup, config_space):
    """
    Turns an OpenML setup object into a Configuration object.
    Throws an error if not possible

    Parameters
    ----------
    setup : OpenMLSetup
        the setup object

    config_space : ConfigurationSpace
        The configuration space

    Returns
    -------
    The Configuration object
    """
    if not isinstance(setup, openml.setups.OpenMLSetup):
        raise TypeError('setup should be of type: openml.setups.OpenMLSetup')
    if not isinstance(config_space, ConfigSpace.ConfigurationSpace):
        raise TypeError('config_space should be of type: ConfigSpace.ConfigurationSpace')

    name_values = dict()
    name_inputid = {param.parameter_name: id for id, param in setup.parameters.items()}
    for hyperparameter in config_space.get_hyperparameters():
        name = hyperparameter.name
        if name not in name_inputid.keys():
            raise KeyError('Setup does not contain parameter: %s' % hyperparameter.name)
        value = setup.parameters[name_inputid[hyperparameter.name]].value
        # TODO: take into account hyperparameter conditionals. i.e.,
        # the libsvm_svc degree is only relevant when the poly kernel is selected
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.UniformIntegerHyperparameter):
            name_values[name] = int(value)
        elif isinstance(hyperparameter, ConfigSpace.hyperparameters.NumericalHyperparameter):
            name_values[name] = float(value)
        else:
            val = json.loads(value)
            if isinstance(val, bool):
                val = str(val)
            name_values[name] = val

    return ConfigSpace.Configuration(config_space, name_values)


def setup_in_config_space(setup, config_space):
    """
    Checks whether a given setup is within the boundaries of a config space

    Parameters
    ----------
    setup : OpenMLSetup
        the setup object

    config_space : ConfigurationSpace
        The configuration space

    Returns
    -------
    Whether this setup is within the boundaries of a config space
    """
    try:
        setup_to_configuration(setup, config_space)
        return True
    except ValueError:
        return False


def filter_setup_list_by_config_space(setups, config_space):
    """
    Removes all setups that do not comply to the config space

    Parameters
    ----------
    setups : OpenMLSetup
        the setup object

    config_space : ConfigurationSpace
        The configuration space

    Returns
    -------
    A dict mapping from setup id to setup, all complying to the config space
    """
    if not isinstance(setups, dict):
        raise TypeError('setups should be of type: dict')

    setups_remain = {}
    for sid, setup in setups.items():
        if setup_in_config_space(setup, config_space):
            setups_remain[sid] = setup
    return setups_remain


def setup_to_parameter_dict(setup: openml.setups.OpenMLSetup,
                            parameter_field: str,
                            relevant_parameters: typing.Set[str]):
    """
    Transforms a setup into a dict, containing the relevant parameters as key / value pair

    Parameters
    ----------
    setup : OpenMLSetup
        the OpenML setup object

    parameter_field : str
        the key field in the parameter object that should be selected. Use full_name in order to avoid collisions; a
        good alternative is the use of parameter_name

    relevant_parameters : set
        The parameters that are expected to become the keys of the dict (others are neglected). Which field is used
        depends on the value of parameter_field

    Returns
    -------
    A dict mapping from parameter name to value

    """
    hyperparameters = {}
    for pid, hyperparameter in setup.parameters.items():
        name = getattr(hyperparameter, parameter_field)
        value = hyperparameter.value
        if name not in relevant_parameters:
            continue

        if name in hyperparameters:
            # duplicate parameter name, this can happen due to sub-flows.
            # when this happens, we need to fix
            raise KeyError('Duplicate hyperparameter: %s' % name)

        hyperparameters[name] = json.loads(value)

    missing_parameters = relevant_parameters - hyperparameters.keys()
    if len(missing_parameters) > 0:
        raise ValueError('Setup %d does not comply to relevant parameters set. Missing: %s' % (setup.setup_id,
                                                                                               str(missing_parameters)))
    return hyperparameters
