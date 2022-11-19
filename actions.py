""" Module for working with general actions.
    It also has action decorator to check parameters in action functions
    and add status and error text to output

    functions:
        action - decorator for checking parameters and adding status and error text to output
    get_actions - returns general actions available in request

"""

from functools import wraps
from typing import Optional, Callable, Any
from vm_logging.exceptions import ParameterNotFoundException
from vm_versions import get_version
from general import test

__all__ = ['action', 'get_actions']


def action(check_fields: Optional[list[str]] = None) -> Callable:
    """ Decorator. Adds checking required field in request parameters.
        Fields for checking define in "check_fields" parameter
        Also this decorator converts result to dict {'status': 'OK', 'error_text': '', 'result': result}
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

            return {'status': 'OK', 'error_text': '', 'result': result}

        return wrapper

    return action_with_check


def get_actions() -> dict[str, Callable]:
    """ forms general actions dict"""
    return dict({'test': _test, 'get_version': _get_version})


def _get_version(parameters: dict[str, Any]) -> dict[str, Any]:
    return {'result': get_version()}


def _test(parameters: dict[str, Any]) -> dict[str, Any]:
    """ Action for testing and debugging """
    return test(parameters)
