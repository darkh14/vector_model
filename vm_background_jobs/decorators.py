""" Module contains decorator forcing decorated function to execute in separate subprocess
    Functions:
        execute_in_background - to execute long-term function in background

"""

from typing import Callable, Any

from .background_jobs import BackgroundJob
from vm_logging.exceptions import ParameterNotFoundException

__all__ = ['execute_in_background']


def execute_in_background(func: Callable) -> Callable:

    def wrapper(wrapper_parameters: dict[str, Any], **kwargs) -> dict[str, Any]:

        if wrapper_parameters.get('background_job'):
            if not wrapper_parameters.get('db'):
                raise ParameterNotFoundException('"db" parameter is not found in parameters')

            background_job = BackgroundJob(db_path=wrapper_parameters['db'], subprocess_mode=False)
            result = background_job.execute_function(func, wrapper_parameters)
        else:
            result = func(wrapper_parameters, **kwargs)

        return result

    return wrapper
