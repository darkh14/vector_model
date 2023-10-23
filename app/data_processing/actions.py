""" Module for defining actions of data processor package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Any, Optional
from . import controller, api_types
import api_types as general_api_types


def get_actions() -> list[dict[str, Any]]:
    """
    Returns actions of loading package to use in requests
    :return: dict of function objects - actions of data loading package
    """

    result = list()

    result.append({'name': 'data_initialize_loading', 'path': '{db_name}/data/initialize_loading',
                   'func': _data_initialize_loading,
                   'http_method': 'post', 'requires_authentication': True})

    result.append({'name': 'data_get_loading_info', 'path': '{db_name}/data/get_loading_info',
                   'func': _data_get_loading_info,
                   'http_method': 'get', 'requires_authentication': True})

    result.append({'name': 'data_get_count', 'path': '{db_name}/data/get_count',
                   'func': _get_data_count,
                   'http_method': 'post', 'requires_authentication': True})

    result.append({'name': 'data_delete', 'path': '{db_name}/data/delete',
                   'func': _data_delete,
                   'http_method': 'post', 'requires_authentication': True})

    result.append({'name': 'data_set_package_status', 'path': '{db_name}/data/set_package_status',
                   'func': _data_set_package_status,
                   'http_method': 'post', 'requires_authentication': True})

    result.append({'name': 'data_set_loading_status', 'path': '{db_name}/data/set_loading_status',
                   'func': _data_set_loading_status,
                   'http_method': 'post', 'requires_authentication': True})

    result.append({'name': 'data_drop_loading', 'path': '{db_name}/data/drop_loading',
                   'func': _data_drop_loading,
                   'http_method': 'get', 'requires_authentication': True})

    result.append({'name': 'data_load_package', 'path': '{db_name}/data/load_package',
                   'func': _data_load_package,
                   'http_method': 'post', 'requires_authentication': True})

    return result


def _data_initialize_loading(loading_data: api_types.Loading) -> str:
    """ For initializing new loading
    :param loading_data: data of initializing loading
    :return: str of success of initializing
    """
    return controller.initialize_loading(loading_data)


def _data_load_package(package: api_types.PackageWithData,
                       background_job: Optional[bool] = False) -> general_api_types.BackgroundJobResponse:
    """ For loading data from package
    :param package: package with data
    :return: result of package loading
    """
    return controller.load_package(package, background_job=background_job)


def _data_drop_loading(id: str, delete_data: Optional[bool] = False) -> api_types.LoadingInfo:
    """ For deleting loading and package objects from db
    :param id: id of loading to drop
    :return: result of data dropping (loading info)
    """

    result = controller.drop_loading(id, delete_data)
    return result


def _data_set_loading_status(loading_status_body: api_types.LoadingStatusBody) -> str:
    """ For setting loading status (for administrators)
    :param loading_status_body: parameters with id and status to set
    :return: result of setting loading status
    """
    return controller.set_loading_status(loading_status_body.id, loading_status_body.status)


def _data_get_loading_info(id: str) -> api_types.LoadingInfo:
    """ For getting loading and its packages info
    :param id: id of loading,
    :return: loading information - statuses, dates etc
    """
    return controller.get_loading_info(id)


def _data_set_package_status(package_status_body: api_types.PackageStatusBody) -> str:
    """ For setting package status (for administrators)
    :param package_status_body: parameters of package and its status
    :return: result of package status setting
    """
    return controller.set_package_status(package_status_body.loading_id,
                                         package_status_body.id,
                                         package_status_body.status)


def _data_delete(loading_id: Optional[str] = None,
                 data_filter_body: Optional[api_types.DataFilterBody] = None) -> str:
    """ Deletes data, loaded by loading or by filter
    :param loading_id: id of deleting loading
    :param data_filter_body: filter of data query
    :return: result of data deleting
    """
    return controller.delete_data(loading_id, data_filter_body.data_filter if data_filter_body else None)


def _get_data_count(data_filter_body: api_types.DataFilterBody) -> int:
    """ Returns general number of documents in data collection
    :param data_filter_body: filter of data query
    :return: number of documents in data collection
    """
    return controller.get_data_count(data_filter_body.data_filter)
