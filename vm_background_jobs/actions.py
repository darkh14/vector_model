""" Module for defining actions of background jobs package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

from typing import Callable

__all__ = ['get_actions']


def get_actions() -> dict[str, Callable]:
    return {}
