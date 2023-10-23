""" Module for defining actions of background jobs package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

from typing import Callable, Optional
from . import controller
from . import api_types
__all__ = ['get_actions']


def get_actions() -> list[dict[str, Callable]]:
    """
    Returns actions of background jobs package
    :return: list of actions dict (functions and its parameters)
    """
    result = list()

    result.append({'name': 'jobs_get_info', 'path': '{db_name}/jobs/get_info', 'func': _get_jobs_info, 'http_method': 'post',
                   'requires_authentication': True})

    result.append({'name': 'jobs_delete', 'path': '{db_name}/jobs/delete', 'func': _delete_background_job, 'http_method': 'get',
                   'requires_authentication': True})

    return result


def _get_jobs_info(job_filter: Optional[api_types.FilterBody] = None) -> list[api_types.JobInfo]:
    """ Returning background jobs info (id, status, start_date, end_date, pid) according to filter,
    :param job_filter: filter to choose jobs
    :return: list if job information
    """

    job_filter_dict = job_filter.model_dump() if job_filter else None

    result = controller.get_jobs_info(job_filter_dict)['jobs']
    return list(map(api_types.JobInfo.model_validate, result))


def _delete_background_job(id: str) -> str:
    """ Removing job line from db
    :param id: id of job to delete
    :return: result of job deleting
    """
    result = controller.delete_background_job(id)
    return result
