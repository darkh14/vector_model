""" Module for defining actions of vm_settings package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable, Any, Optional
from functools import wraps
from vm_logging.exceptions import ParameterNotFoundException
from .controller import get_var, set_var


def action(check_fields: Optional[list[str]] = None) -> Callable:
    """ Decorator. Adds checking required field in request parameters.
        Fields for checking define in "check_fields" parameter
        :param check_fields: fields to be checked in decorator
        :return: decorated method
    """
    def action_with_check(func: Callable):
        @wraps(func)
        def wrapper(parameters, *args, **kwargs):
            nonlocal check_fields
            if not check_fields:
                c_check_fields = []
            else:
                c_check_fields = check_fields.copy()

            for field in c_check_fields:
                if field not in parameters:
                    raise ParameterNotFoundException(field)

            result = func(parameters, *args, *kwargs)

            return result

        return wrapper

    return action_with_check


@action(['settings_name'])
def _get_var(parameters: dict[str, str]) -> Any:
    """
    For getting var from settings
    :param parameters: dict of request parameters
    :return: required var
    """
    return get_var(parameters['settings_name'])


@action(['settings_name', 'settings_value'])
def _set_var(parameters: dict[str, Any]) -> str:
    """
    For setting var to settings
    :param parameters: dict of request parameters
    :return: result of set var
    """
    set_var(parameters['settings_name'], parameters['settings_value'])
    return 'Parameter "{}" is set'.format(parameters['settings_name'])


def get_actions() -> dict[str, Callable]:
    """ forms actions dict available for vm_logging
    :return: dict of available actions (functions)
    """
    return {'settings_get_var': _get_var, 'settings_set_var': _set_var}
