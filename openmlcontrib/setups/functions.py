import copy
import json
import openml


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


def setup_complies_to_fixed_parameters(setup, fixed_parameters, keyfield):
    """
    Given an OpenML Setup, it checks whether several hyperparameter values are
    the same as the ones given in a dictionary of fixed_parameters

    Parameters
    ----------
    setup : OpenMLSetup
        The setup whose hyperparameters should be checked

    fixed_parameters : dict[mixed, mixed]
        A dict mapping from an attribute name in setup (determined by
        argument keyfield, typically the parameter_name) to the required value

    keyfield : mixed
        The required key field in the hyperparameter
        (typically parameter_name)

    Returns
    -------
    True iff the setup has all values complying to the fixed parameters,
    False otherwise
    """
    if not isinstance(setup, openml.setups.OpenMLSetup):
        raise ValueError('setup should be of type OpenMLSetup')
    if not isinstance(fixed_parameters, dict):
        raise ValueError('fixed_parameters should be of type dict')

    if len(fixed_parameters) == 0:
        return True

    setup_parameters = {getattr(setup.parameters[param_id], keyfield): setup.parameters[param_id].value for param_id in
                        setup.parameters}
    for parameter in fixed_parameters.keys():
        if parameter not in setup_parameters.keys():
            raise ValueError('Fixed parameter %s not in setup parameter for setup %d' % (parameter, setup.setup_id))
        value_online = openml.flows.flow_to_sklearn(setup_parameters[parameter])
        value_request = fixed_parameters[parameter]
        if value_online != value_request:
            return False
    return True


def obtain_setups_by_id(flow_id, setup_ids, fixed_parameters=None, keyfield=None):
    """
    Obtains a list of setups by id. Because the

    Parameters
    ----------
    flow_id : int
        The flow to which the setups belong

    setup_ids : list[int]
        list of setups to obtain

    fixed_parameters : dict[mixed, mixed] (optional)
        A dict mapping from an attribute name in setup (determined by
        argument keyfield, typically the parameter_name) to the required value

    keyfield : mixed (optional)
        The required key field in the hyperparameter
        (typically parameter_name)

    Returns
    -------
    The dict of setups
    """
    if not isinstance(flow_id, int):
        raise ValueError()
    if not isinstance(setup_ids, list):
        raise ValueError()
    if (keyfield is None) != (fixed_parameters is None):  # xor
        raise ValueError('Setting fixed_parameters either or keyfield is unexpected')

    setups = {}
    offset = 0
    limit = 250
    setup_ids = list(setup_ids)
    while True:
        setups_batch = openml.setups.list_setups(flow=flow_id, setup=setup_ids[offset:offset+limit])
        if fixed_parameters is None:
            setups.update(setups_batch)
        else:
            for setup_id in setups_batch.keys():
                if setup_complies_to_fixed_parameters(setups_batch[setup_id], keyfield, fixed_parameters):
                    setups[setup_id] = setups_batch[setup_id]

        offset += limit
        if len(setups_batch) < limit:
            break
    return setups
