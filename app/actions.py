""" Module for working with general actions.
    It also has action decorator to check parameters in action functions
    and add status and error text to output

    functions:
        action - decorator for checking parameters and adding status and error text to output
    get_actions - returns general actions available in request

"""

from typing import Any
from vm_versions import get_version
from general import ping

__all__ = ['get_actions']


def get_actions() -> list[dict[str, Any]]:
    """ forms general actions` dict.
    :return: dict of function object (general actions)
    """

    result = list()

    result.append({'name': 'ping', 'path': 'ping', 'func': _ping, 'http_method': 'get',
                   'requires_authentication': False})
    result.append({'name': 'get_version', 'path': 'get_version', 'func': _get_version, 'http_method': 'get',
                   'requires_authentication': True})

    return result


def _ping() -> dict[str, Any]:
    """ Action for testing connection with service
    :return: result of checking connection
    """
    return ping()


def _get_version() -> str:
    """
    Returns version of vector_model
    :return: version of module
    """
    return get_version()
