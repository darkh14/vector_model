""" Module for defining actions of vm_models package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable


def get_actions() -> dict[str, Callable]:
    """ forms actions dict available for vm_models"""
    return dict()
