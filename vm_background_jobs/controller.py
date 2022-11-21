""" Module for working with background jobs """

from typing import Any
from vm_logging.exceptions import ParameterNotFoundException
from .background_jobs import BackgroundJob


def get_jobs_info(parameters: dict[str, Any]) -> dict[str, Any]:
    job_filter = parameters.get('filter')
    db_path = parameters.get('db')

    if not db_path:
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    return {'jobs': BackgroundJob.get_jobs_info(job_filter, db_path)}


def delete_background_job(parameters: dict[str, Any]) -> str:

    db_path = parameters.get('db')

    if not db_path:
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    job_from_parameters = parameters.get('job')
    if not job_from_parameters:
        raise ParameterNotFoundException('Parameter "job" is not found in request parameters')

    job_id = job_from_parameters.get('id')
    if not job_id:
        raise ParameterNotFoundException('Parameter "id" is not found in job parameter')

    background_job = BackgroundJob(job_id, db_path)
    background_job.delete()

    return 'Background job is deleted'
