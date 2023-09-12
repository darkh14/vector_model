""" Module for defining actions of vm_logging package

    functions:
        get_actions() returning list of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable


def get_actions() -> list[dict[str, Callable]]:
    """ forms actions dict available for vm_logging
    :return: list of actions of vm_logging package
    """
    return list()
