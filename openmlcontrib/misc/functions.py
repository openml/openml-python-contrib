
def filter_listing(listing, property_name, allowed_values=None):
    """
    Removes items from the result of a listing fn if a property
    does not comply with a given value. 

     Parameters
        ----------
        setupid_setup : dict of dicts, representing openml objects
            as obtained from an openml listing fn 

        property_name : str
            the name of the property which values should be restricted

        allowed_values : list
            list of allowed values

        Returns
        -------
        model : dict of dicts
            a dict, with the objects that did not comply removed 
    """
    allowed = dict()
    if not isinstance(allowed_values, list):
        raise ValueError('allowed values should be a list')

    for id, object in listing.items():
        if property_name not in object:
            raise ValueError('dict does not have property: %s' %property_name)

        if object[property_name] in allowed_values:
            allowed[id] = object

    return allowed
