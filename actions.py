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

__all__ = ['get_actions']


def get_actions() -> dict[str, Callable]:
    """ forms general actions dict"""
    return dict({'test': _test, 'get_version': _get_version})


def _get_version(parameters: dict[str, Any]) -> dict[str, Any]:
    return {'result': get_version()}


def _test(parameters: dict[str, Any]) -> dict[str, Any]:
    """ Action for testing and debugging """
    return test(parameters)
