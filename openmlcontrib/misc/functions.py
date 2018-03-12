
def filter_listing(listing, property_name, allowed_values, dict_representation=True):
    """
    Removes items from the result of a listing fn if a property
    does not comply with a given value. 

     Parameters
        ----------
        setupid_setup : dict of dicts or objects, representing 
            openml objects as obtained from an openml listing fn 

        property_name : str
            the name of the property which values should be restricted

        allowed_values : list
            list of allowed values
        
        dict_representation : bool
            wether the individual items are represented as dicts 
            or objects

        Returns
        -------
        model : dict of dicts or objects
            a dict, with the objects that did not comply removed 
    """
    allowed = dict()
    if not isinstance(allowed_values, list):
        raise ValueError('allowed values should be a list')

    for id, object in listing.items():
        if dict_representation:
            if property_name not in object:
                raise ValueError('dict does not have property: %s' %property_name)

            if object[property_name] in allowed_values:
                allowed[id] = object
        else:
            if not hasattr(object, property_name):
                raise ValueError('dict does not have property: %s' % property_name)

            if getattr(object, property_name) in allowed_values:
                allowed[id] = object

    return allowed
