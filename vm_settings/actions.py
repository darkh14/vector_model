""" Module for defining actions of vm_settings package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable, Any
from actions import action

from .controller import get_var, set_var


@action(['settings_name'])
def _get_var(parameters: dict[str, str]) -> Any:

    return get_var(parameters['settings_name'])


@action(['settings_name', 'settings_value'])
def _set_var(parameters: dict[str, Any]) -> str:

    set_var(parameters['settings_name'], parameters['settings_value'])
    return 'Parameter "{}" is set'.format(parameters['settings_name'])


def get_actions() -> dict[str, Callable]:
    """ forms actions dict available for vm_logging"""
    return {'settings_get_var': _get_var, 'settings_set_var': _set_var}
