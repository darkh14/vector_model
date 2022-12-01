""" Module for defining actions of vm_models package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable, Any


def get_actions() -> dict[str, Callable]:
    """ forms actions dict available for vm_models"""
    return dict()


def _fit(parameters: dict[str, Any]) -> dict[str, Any]:
    pass


def _predict(parameters: dict[str, Any]) -> dict[str, Any]:
    pass


def _initialize(parameters: dict[str, Any]) -> dict[str, Any]:
    pass


def _drop(parameters: dict[str, Any]) -> dict[str, Any]:
    pass


def _get_info(parameters: dict[str, Any]) -> dict[str, Any]:
    pass


def _drop_fitting(parameters: dict[str, Any]) -> dict[str, Any]:
    pass


def _update(parameters: dict[str, Any]) -> dict[str, Any]:
    pass
