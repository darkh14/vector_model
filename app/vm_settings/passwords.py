""" Module for saving and operating with passwords and other confident data

    functions:
        set_password
        get_password
    variables:
    CACHE - for caching data
"""

import keyring

__all__ = ['set_password', 'get_password']

CACHE = {}


def set_password(service_name: str, user_name: str, password: str) -> None:
    """ for setting passwords
    :param service_name: service name parameter (additional parameter)
    :param user_name: username parameter
    :param password: password to set
    """
    keyring.set_password(service_name, user_name, password)
    CACHE[(service_name, user_name)] = password


def get_password(service_name: str, user_name: str) -> str:
    """ for setting passwords
    :param service_name: service name parameter (additional parameter)
    :param user_name: username parameter
    :return: password
    """
    if (service_name, user_name) in CACHE.keys():
        result = CACHE[(service_name, user_name)]
    else:
        result = keyring.get_password(service_name, user_name)

    return result
