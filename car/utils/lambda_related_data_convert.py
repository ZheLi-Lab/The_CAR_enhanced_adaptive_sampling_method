import numpy as np

def check_iflambda_in_mbar(lambda_dict, mbar_lambda_dict):
    """
    Convert lambda_dict and mbar_lambda_dict to vectors and check if the state vector
    derived from lambda_dict is in the state vectors derived from mbar_lambda_dict.

    Parameters:
    - lambda_dict (dict): Dictionary containing lambda values.
        Keys are lambda types, and values are corresponding lambda values.
    - mbar_lambda_dict (dict): Dictionary containing multiple lambda values.
        Similar structure to lambda_dict but contains multiple states.

    Returns:
    - bool: True if the state vector derived from lambda_dict is in the state vectors
        derived from mbar_lambda_dict, False otherwise.
    """
    # Convert lambda_dict to a column vector
    lambda_vector = np.array([lambda_dict['lambda_restraints'][0],
                             lambda_dict['lambda_electrostatics'][0],
                             lambda_dict['lambda_sterics'][0]])

    # Convert mbar_lambda_dict to a 3*N matrix
    mbar_matrix = np.array([mbar_lambda_dict['lambda_restraints'],
                            mbar_lambda_dict['lambda_electrostatics'],
                            mbar_lambda_dict['lambda_sterics']])

    # Check if lambda_vector is in mbar_matrix
    is_in_matrix = np.any(np.all(mbar_matrix == lambda_vector.reshape(3, 1), axis=0))

    return is_in_matrix

def is_monotonic(lst):
    """
    Check if a list of floats is monotonically increasing or decreasing.

    Parameters
    ----------
    - lst (list): The list of floats to check.

    Returns:
    --------
    - bool: True if the list is monotonic, False otherwise.
    """

    increasing = all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))
    decreasing = all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))

    return increasing or decreasing

def from_mbar_lambda_dict_to_lambda_lst(mbar_lambda_dict):
    """
    Converts MBAR lambda dictionary to a list of lambda values based on specified conditions.

    Parameters
    ---------
    - mbar_lambda_dict (dict): The lambda values of MBAR.
        The keys are 'lambda_restraints', 'lambda_electrostatics', and 'lambda_sterics'.
        The values are three respective lambda lists.

    Returns
    -------
    - lambda_lst (list): A list of lambda values based on specified conditions.
    - lambda_type (str): The type of the lambda values. It can be 'electrostatics' or 'sterics'

    Raises
    ------
    - ValueError: If 'lambda_restraints', 'lambda_electrostatics', and 'lambda_sterics'
    lists are not of equal length.
    - ValueError: If elements in the 'lambda_restraints' list are not consistent.
    - ValueError: If unable to handle the case of asynchronous changes in
    'lambda_electrostatics' and 'lambda_sterics'.
    - ValueError: If the lambda_lst is not monotonic.

    Example usage:
        mbar_lambda_dict = {'lambda_restraints': [0.1, 0.2, 0.3],
                            'lambda_electrostatics': [0.0, 0.5, 1.0],
                            'lambda_sterics': [0.0, 0.5, 1.0]}
        lambda_lst = from_mbar_lambda_dict_to_lambda_lst(mbar_lambda_dict)
        print(lambda_lst)
    """

    # Check if lists are of equal length
    if len(set(map(len, mbar_lambda_dict.values()))) != 1:
        raise ValueError(
            "'lambda_restraints', 'lambda_electrostatics', 'lambda_sterics' are not equal."
            )

    lambda_restraints = mbar_lambda_dict['lambda_restraints']

    # Check if elements in 'lambda_restraints' are consistent
    if len(set(lambda_restraints)) != 1:
        raise ValueError("Elements in 'lambda_restraints' list are not consistent.")

    lambda_electrostatics = mbar_lambda_dict['lambda_electrostatics']
    lambda_sterics = mbar_lambda_dict['lambda_sterics']

    # Check conditions and return lambda_lst
    if lambda_electrostatics == lambda_sterics:
        lambda_lst = lambda_electrostatics
        # Check if the lambda_lst is monotonic
        if not is_monotonic(lambda_lst):
            raise ValueError("The lambda_lst is not monotonic.")
        return lambda_lst, 'electrostatics'
    elif len(set(lambda_sterics)) != 1:
        lambda_lst = lambda_electrostatics
        if not is_monotonic(lambda_lst):
            raise ValueError("The lambda_lst is not monotonic.")
        return lambda_lst, 'electrostatics'
    elif len(set(lambda_electrostatics)) != 1:
        lambda_lst = lambda_sterics
        if not is_monotonic(lambda_lst):
            raise ValueError("The lambda_lst is not monotonic.")
        return lambda_lst, 'sterics'
    else:
        raise ValueError(
            "Unable to handle asynchronous changes in 'lambda_electrostatics' and 'lambda_sterics'."
            )

def from_lambda_dict_to_lambda_float(lambda_dict, mbar_lambda_dict):
    '''Convert the lambda dict to lambda value(float).
    Parameters
    ----------
    - lambda_dict (dict): The lambda values of restraints, electrostatics and sterics. 
        The keys are 'lambda_restraints', 'lambda_electrostatics' and 'lambda_sterics'. 
        The values are three respective lambda list with one value, whose datatype is [float, ].
    - mbar_lambda_dictThe lambda values of states whose energy need to be calculated.
        The keys are 'lambda_restraints', 'lambda_electrostatics', and 'lambda_sterics'.
        The values are three respective lambda lists.
    '''
    if check_iflambda_in_mbar(lambda_dict, mbar_lambda_dict):
        change_lambda_type = from_mbar_lambda_dict_to_lambda_lst(mbar_lambda_dict)[1]
        key = 'lambda_' + change_lambda_type
    else:
        raise ValueError("The lambda_dict is not in mbar_lambda_dict.")
    return lambda_dict[key][0]
