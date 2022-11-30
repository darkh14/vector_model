""" Module for defining actions of db processor package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable, Any
from .controller import check_connection


def get_actions() -> dict[str, Callable]:
    """ forms actions dict available for db_processing"""
    return dict({'db_check_connection': _check_connection})


def _check_connection(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For checking connection """
    return check_connection(parameters)
