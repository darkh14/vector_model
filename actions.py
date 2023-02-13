""" Module for working with general actions.
    It also has action decorator to check parameters in action functions
    and add status and error text to output

    functions:
        action - decorator for checking parameters and adding status and error text to output
    get_actions - returns general actions available in request

"""

from typing import Callable, Any
from vm_versions import get_version
from general import test, ping

__all__ = ['get_actions', 'get_action_names_without_db_using']


def get_actions() -> dict[str, Callable]:
    """ forms general actions` dict.
    :return: dict of function object (general actions)
    """
    return dict({'test': _test, 'get_version': _get_version, 'ping': _ping})


def get_action_names_without_db_using() -> list[str]:
    """
    Returns action names without db using not to initialize db connector
    :return: action names list
    """
    return ['ping']


def _get_version(parameters: dict[str, Any]) -> str:
    """
    Returns version of vector_model
    :param parameters: dict of request parameters
    :return: version of module
    """
    return get_version()


def _test(parameters: dict[str, Any]) -> dict[str, Any]:
    """ Action for testing and debugging
    :param parameters: dict of request parameters
    :return: result of testing
    """
    return test(parameters)


def _ping(parameters: dict[str, Any]) -> dict[str, Any]:
    """ Action for testing connection with service
    :param parameters: dict of request parameters
    :return: result of checking connection
    """
    return ping(parameters)
