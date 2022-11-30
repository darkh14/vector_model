""" Module for controlling data loading and data converting

    functions:
        initialize_loading  - for initializing new loading
        load_package - for load current package
        drop_loading - for deleting loading (deletes loading and its packages in db collection, not deletes data)
        set_loading_status - for setting loading status
        set_package_status - for setting package status
        get_loading_info - returns loading statuses and other info
        delete_data - for deleting data by loading or by filter
        _get_loading - for getting loading object
        ---


"""

from typing import Any
from vm_logging.exceptions import RequestProcessException
from db_processing import get_connector
from .loader import Loading, delete_all_data
from vm_background_jobs.decorators import execute_in_background

__all__ = ['initialize_loading', 'load_package', 'drop_loading', 'set_loading_status', 'get_loading_info']


def initialize_loading(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new loading """
    loading = _get_loading(parameters)
    result = loading.initialize()
    return result


@execute_in_background
def load_package(parameters: dict[str, Any]) -> str:
    """ For loading package data """
    loading = _get_loading(parameters)
    loading.load_package(parameters['loading'].get('package'))
    return 'Package is loaded'


def drop_loading(parameters: dict[str, Any]) -> str:
    """ For deleting loading object and its packages """
    loading = _get_loading(parameters)
    loading.drop(need_to_delete_data=parameters['loading'].get('delete_data'))
    return 'Loading is dropped'


def set_loading_status(parameters: dict[str, Any]) -> str:
    """ For setting status of loading (for administrators) """
    loading = _get_loading(parameters)
    loading.set_status(parameters['loading'].get('status'), set_for_packages=True)
    return 'Loading status is set'


def set_package_status(parameters: dict[str, Any]) -> str:
    """ For setting status of package (for administrators) """
    loading = _get_loading(parameters)
    loading.set_package_status(parameters['loading'].get('package'))
    return 'Status is set'


def get_loading_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting info of loading and packages """
    loading = _get_loading(parameters)
    info = loading.get_loading_info()
    return info


def delete_data(parameters: dict[str, Any]) -> str:
    """ For deleting data by loading or by filter.
    If loading is in parameters data deletes by loading, else by filter
    """

    if 'loading' not in parameters:
        data_filter = parameters.get('filter') or {}
        db_connector = get_connector(parameters.get('db'))
        delete_all_data(db_connector, data_filter)
    else:
        loading = _get_loading(parameters)
        loading.delete_data()

    return 'Data are deleted'


def _get_loading(parameters: dict[str | Any]) -> Loading:
    """ Gets loading object. Sets db connector to loading """
    if 'loading' not in parameters:
        raise RequestProcessException('Property "loading" is not found in request parameters')

    db_connector = get_connector(parameters.get('db'))
    return Loading(parameters['loading'], db_connector)
