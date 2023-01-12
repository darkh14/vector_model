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

from typing import Any, Callable

from vm_logging.exceptions import RequestProcessException, ParametersFormatError
from db_processing import get_connector
from .loader import Loading, delete_all_data
from vm_background_jobs.decorators import execute_in_background

__all__ = ['initialize_loading', 'load_package', 'drop_loading', 'set_loading_status', 'get_loading_info']


def initialize_loading(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new loading
    :param parameters: dict of request parameters
    :return: new loading info
    """

    result = None

    match parameters:
        case {'loading': {'id': str(loading_id),
                          'type': str(loading_type),
                          'packages': list(packages)}} if loading_id and loading_type and packages:
            loading = Loading(loading_id=loading_id, loading_type=loading_type, packages=packages)

            result = loading.initialize()
        case _:
            raise ParametersFormatError('Wrong loading parameters format! Check "loading" parameter')

    return result


@execute_in_background
def load_package(parameters: dict[str, Any]) -> str:
    """ For loading package data. Can be executed in background job
    :param parameters: dict of request parameters
    :return: result of package loading
    """

    match parameters:
        case {'loading': {'id': str(loading_id), 'package': dict(package)}} if loading_id and package:
            loading = Loading(loading_id)
            loading.load_package(package)
        case _:
            raise ParametersFormatError('Wrong loading parameters format! Check "loading" parameter')

    return 'Package loading is started' if parameters.get('background_job') else 'Package is loaded'


def drop_loading(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting loading object and its packages
    :param parameters: dict of request parameters
    :return: result of data dropping
    """

    result = None

    match parameters:
        case {'loading': {'id': str(loading_id)}} if loading_id:
            loading = Loading(loading_id)

            result = loading.drop(need_to_delete_data=parameters['loading'].get('delete_data'))
        case _:
            raise ParametersFormatError('Wrong loading parameters format! Check "loading" parameter')

    return result


def set_loading_status(parameters: dict[str, Any]) -> str:
    """ For setting status of loading (for administrators)
    :param parameters: dict of request parameters
    :return: result of setting loading status
    """

    match parameters:
        case {'loading': {'id': str(loading_id), 'status': str(status)}} if loading_id and status:
            loading = Loading(loading_id)
            loading.set_status(status, set_for_packages=True, from_outside=True)
        case _:
            raise ParametersFormatError('Wrong loading parameters format! Check "loading" parameter')

    return 'Loading status is set'


def set_package_status(parameters: dict[str, Any]) -> str:
    """ For setting status of package (for administrators)
    :param parameters: dict of request parameters
    :return: result of setting package status
    """

    match parameters:
        case {'loading': {'id': str(loading_id), 'package': dict(package)}} if loading_id and package:
            loading = Loading(loading_id)
            loading.set_package_status(package)
        case _:
            raise ParametersFormatError('Wrong loading parameters format! Check "loading" parameter')

    return 'Package status is set'


def get_loading_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting info of loading and packages
    :param parameters: dict of request parameters
    :return: loading information - statuses, dates etc
    """

    info = None

    match parameters:
        case {'loading': {'id': str(loading_id)}} if loading_id:
            loading = Loading(loading_id)
            info = loading.get_loading_info()
        case _:
            raise ParametersFormatError('Wrong loading parameters format! Check "loading" parameter')

    return info


def delete_data(parameters: dict[str, Any]) -> str:
    """ For deleting data by loading or by filter.
    If loading is in parameters data deletes by loading, else by filter
    :param parameters: dict of request parameters
    :return: result of data deleting
    """

    if 'loading' not in parameters:
        data_filter = parameters.get('filter') or {}
        delete_all_data(data_filter)
    else:

        match parameters:
            case {'loading': {'id': str(loading_id)}} if loading_id:
                loading = Loading(loading_id)
                loading.delete_data()
            case _:
                raise ParametersFormatError('Wrong loading parameters format! Check "loading" parameter')

    return 'Data are deleted'


def get_data_count(parameters: dict[str, Any]) -> int:
    """ Returns general number of documents in data collection
    :param parameters: dict of request parameters
    :return: number of documents in data collection
    """
    data_filter = parameters.get('filter') or {}
    db_connector = get_connector()
    return db_connector.get_count('raw_data', data_filter)
