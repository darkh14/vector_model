""" Module for defining actions of data processor package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable, Any
from . import controller


def get_actions() -> dict[str, Callable]:
    return {
        'data_initialize_loading': _data_initialize_loading,
        'data_load_package': _data_load_package,
        'data_drop_loading': _data_drop_loading,
        'data_set_loading_status': _data_set_loading_status,
        'data_set_package_status': _data_set_package_status,
        'data_get_loading_info': _data_get_loading_info,
        'data_delete': _data_delete
            }


def _data_initialize_loading(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new loading """
    return controller.initialize_loading(parameters)


def _data_load_package(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For loading data from package """
    return controller.load_package(parameters)


def _data_drop_loading(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting loading and package objects from db """
    return controller.drop_loading(parameters)


def _data_set_loading_status(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For setting loading status (for administrators) """
    return controller.set_loading_status(parameters)


def _data_get_loading_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting loading and its packages info """
    return controller.get_loading_info(parameters)


def _data_set_package_status(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For setting package status (for administrators) """
    return controller.set_package_status(parameters)


def _data_delete(parameters: dict[str, Any]) -> dict[str, Any]:
    """ Deletes data, loaded by loading or by filter """
    return controller.delete_data(parameters)
