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

from typing import Any, Callable, Optional

from db_processing import get_connector
from .loader import Loading, delete_all_data
from . import api_types
import api_types as general_api_types
from vm_background_jobs.decorators import execute_in_background

__all__ = ['initialize_loading', 'load_package', 'drop_loading', 'set_loading_status', 'get_loading_info']


def initialize_loading(loading_data: api_types.Loading) -> str:
    """ For initializing new loading
    :param loading_data: info of initializing loading
    :return: new loading info
    """

    loading_data_dict = loading_data.to_dict()

    loading = Loading(loading_id=loading_data_dict['id'], loading_type=loading_data_dict['type'],
                      packages=loading_data_dict['packages'])

    loading.initialize()

    return 'Loading id "{}" is initialized'.format(loading.get_id())


@execute_in_background
def load_package(package: api_types.PackageWithData, job_id: str = '') -> general_api_types.BackgroundJobResponse:
    """ For loading package data. Can be executed in background job
    :param package: package with data
    :param job_id: id of background job if executes in background mode
    :return: result of package loading
    """

    loading = Loading(package.loading_id)
    loading.load_package(package, job_id)

    result = {'description': 'Package is loaded', 'mode': general_api_types.ExecutionModes.DIRECTLY, 'pid': 0}

    return general_api_types.BackgroundJobResponse.model_validate(result)


def drop_loading(loading_id: str, need_to_delete_data: Optional[bool] = False) -> api_types.LoadingInfo:
    """ For deleting loading object and its packages
    :param loading_id: id of loading to drop
    :param need_to_delete_data: True if we need to delete data of loading
    :return: result of data dropping (loading info)
    """

    loading = Loading(loading_id)

    result = loading.drop(need_to_delete_data=need_to_delete_data)

    return api_types.LoadingInfo.model_validate(result)


def set_loading_status(loading_id: str, loading_status: api_types.LoadingStatuses) -> str:
    """ For setting status of loading (for administrators)
    :param loading_id: id of required loading
    :param loading_status: status to set
    :return: result of setting loading status
    """

    loading = Loading(loading_id)
    loading.set_status(loading_status, set_for_packages=True, from_outside=True)

    return 'Loading status is set'


def set_package_status(loading_id: str, package_id: str, package_status: api_types.LoadingStatuses) -> str:
    """ For setting status of package (for administrators)
    :param loading_id: id of required loading
    :param package_id: id of required package
    :param package_status: status to set
    :return: result of setting package status
    """

    loading = Loading(loading_id)
    loading.set_package_status({'id': package_id, 'status': package_status})

    return 'Package status is set'


def get_loading_info(loading_id: str) -> api_types.LoadingInfo:
    """ For getting info of loading and packages
    :param loading_id: id of loading,
    :return: loading information - statuses, dates etc
    """

    loading = Loading(loading_id)
    info = loading.get_loading_info()

    result = api_types.LoadingInfo.model_validate(info)

    return result


def delete_data(loading_id: str, data_filter: dict[str, Any]) -> str:
    """ For deleting data by loading or by filter.
    If loading_id is in parameters data deletes by loading, else by filter
    :param loading_id: id of deleting loading (all data in loading)
    :param data_filter: dict of data filter (to delete data by filter)
    :return: result of data deleting
    """

    if not loading_id:
        delete_all_data(data_filter)
    else:
        loading = Loading(loading_id)
        loading.delete_data()

    return 'Data are deleted'


def get_data_count(data_filter: Optional[dict[str, Any]] = None) -> int:
    """ Returns general number of documents in data collection
    :param data_filter: dict of data filter
    :return: number of documents in data collection
    """
    db_connector = get_connector()
    return db_connector.get_count('raw_data', data_filter)


def get_action_before_background_job(func_name: str, args: tuple[Any], kwargs: dict[str, Any]) -> Optional[Callable]:

    loading = Loading(args[0].loading_id)
    result = loading.get_action_before_background_job(func_name, args, kwargs)

    return result
