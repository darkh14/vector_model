""" Module for defining actions of vm_settings package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable
from .controller import get_var, set_var
from . import api_types


def _get_var(name: str) -> str | int | float | bool:
    """
    For getting var from settings
    :param name: name of parameter
    :return: required var
    """
    return get_var(name)


def _set_var(var_to_set: api_types.VarToSet) -> str:
    """
    For setting var to settings
    :param var_to_set: name and value of var
    :return: result of set var
    """
    set_var(var_to_set.name, var_to_set.value)
    return 'Parameter "{}" is set'.format(var_to_set.name)


def get_actions() -> list[dict[str, Callable]]:
    """ forms actions dict available for vm_logging
    :return: dict of available actions (functions)
    """

    result = list()

    result.append({'name': 'settings_get_var', 'path': 'settings/get_var', 'func': _get_var, 'http_method': 'get',
                   'requires_db': False, 'result_type': str | int | float | bool})
    result.append({'name': 'settings_set_var', 'path': 'settings/set_var', 'func': _set_var, 'http_method': 'post',
                   'requires_db': False, 'result_type': str})

    return result
