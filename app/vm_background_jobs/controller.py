""" Module for working with background jobs. Provides getting jobs info and deleting job
        Functions:
            get_jobs_info - for getting jobs info. return info of jobs according to filter
            delete_background_job - deleting one job from id

"""

from typing import Any
from .background_jobs import BackgroundJob

__all__ = ['get_jobs_info', 'delete_background_job', 'set_background_job_interrupted']


def get_jobs_info(job_filter: dict[str, Any]) -> dict[str, Any]:
    """ For getting jobs info. return info of jobs according to filter
    :param job_filter: dict of filter to choose job
    :return: dict if job information
    """

    return {'jobs': BackgroundJob.get_jobs_info(job_filter)}


def delete_background_job(job_id: str) -> str:
    """ Deleting one job from id
    :param job_id: id of job to delete
    :return: result of job deleting
    """

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
