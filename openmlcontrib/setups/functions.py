import copy
import json


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
