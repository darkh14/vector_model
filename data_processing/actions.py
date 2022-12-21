""" Module for defining actions of data processor package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable, Any
from . import controller


def get_actions() -> dict[str, Callable]:
    """
    Returns actions of loading package to use in requests
    :return: dict of function objects - actions of data loading package
    """
    return {
        'data_initialize_loading': _data_initialize_loading,
        'data_load_package': _data_load_package,
        'data_drop_loading': _data_drop_loading,
        'data_set_loading_status': _data_set_loading_status,
        'data_set_package_status': _data_set_package_status,
        'data_get_loading_info': _data_get_loading_info,
        'data_delete': _data_delete,
        'data_get_count': _get_data_count
            }


def _data_initialize_loading(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new loading
    :param parameters: dict of request parameters
    :return: new loading info
    """
    return controller.initialize_loading(parameters)


def _data_load_package(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For loading data from package
    :param parameters: dict of request parameters
    :return: result of package loading
    """
    return controller.load_package(parameters)


def _data_drop_loading(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting loading and package objects from db
    :param parameters: dict of request parameters
    :return: result of data dropping
    """
    return controller.drop_loading(parameters)


def _data_set_loading_status(parameters: dict[str, Any]) -> str:
    """ For setting loading status (for administrators)
    :param parameters: dict of request parameters
    :return: result of setting loading status
    """
    return controller.set_loading_status(parameters)


def _data_get_loading_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting loading and its packages info
    :param parameters: dict of request parameters
    :return: loading information - statuses, dates etc
    """
    return controller.get_loading_info(parameters)


def _data_set_package_status(parameters: dict[str, Any]) -> str:
    """ For setting package status (for administrators)
    :param parameters: dict of request parameters
    :return: result of package status setting
    """
    return controller.set_package_status(parameters)


def _data_delete(parameters: dict[str, Any]) -> str:
    """ Deletes data, loaded by loading or by filter
    :param parameters: dict of request parameters
    :return: result of data deleting
    """
    return controller.delete_data(parameters)


def _get_data_count(parameters: dict[str, Any]) -> int:
    """ Returns general number of documents in data collection
    :param parameters: dict of request parameters
    :return: number of documents in data collection
    """
    return controller.get_data_count(parameters)
