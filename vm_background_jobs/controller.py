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
