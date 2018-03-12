
def filter_listing(list, property_name, allowed_values=None):
    """
    Removes items from the result of a listing fn if a property
    does not comply with a given value. 

     Parameters
        ----------
        setupid_setup : dict of [OpenMLRun, OpenMLDataset, OpenMLFlow, 
            OpenMLTask] as obtained from the openml listing fn

        property_name : str
            the name of the property which values should be restricted

        allowed_values : list
            list of allowed values

        Returns
        -------
        model : dict of OpenML objects
            a dict, with the objects that did not comply removed 
    """
    allowed = dict()
    if not isinstance(allowed_values, list):
        raise ValueError('allowed values should be a list')

    for id, object in list.items():
        if not hasattr(object, property_name):
            raise ValueError('OpenML object does not have property: %d' %property_name)

        if object.property_name in allowed_values:
            allowed[id] = object

    return allowed
