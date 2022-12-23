""" Module for working with background jobs. Provides getting jobs info and deleting job
        Functions:
            get_jobs_info - for getting jobs info. return info of jobs according to filter
            delete_background_job - deleting one job from id

"""

from typing import Any
from vm_logging.exceptions import ParameterNotFoundException
from .background_jobs import BackgroundJob

__all__ = ['get_jobs_info', 'delete_background_job', 'set_background_job_interrupted']

def get_jobs_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting jobs info. return info of jobs according to filter
    :param parameters: dict of request parameters
    :return: dict if job information
    """
    job_filter = parameters.get('filter')

    return {'jobs': BackgroundJob.get_jobs_info(job_filter)}


def delete_background_job(parameters: dict[str, Any]) -> str:
    """ Deleting one job from id
    :param parameters: dict of request parameters
    :return: result of job deleting
    """

    job_from_parameters = parameters.get('job')
    if not job_from_parameters:
        raise ParameterNotFoundException('Parameter "job" is not found in request parameters')

    job_id = job_from_parameters.get('id')
    if not job_id:
        raise ParameterNotFoundException('Parameter "id" is not found in job parameter')

    background_job = BackgroundJob(job_id)
    background_job.delete()

    return 'Background job is deleted'

def set_background_job_interrupted(job_id: str) -> str:
    """ set status "interrupted" if background job
    :param job_id: id of current job
    :return: result of interrupting job
    """

    background_job = BackgroundJob(job_id)
    background_job.set_interrupted()

    return 'Background job is deleted'
