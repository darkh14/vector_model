""" Module contains decorator forcing decorated function to execute in separate subprocess
    Functions:
        execute_in_background - to execute long-term function in background

"""

from typing import Callable, Any
from functools import wraps

from .background_jobs import BackgroundJob

__all__ = ['execute_in_background']


def execute_in_background(func: Callable) -> Callable:
    """ Decorator for executing functions in subprocess. Uses BackgroundJob class
    :param func: decorating method
    :return: decorated method
    """

    @wraps(func)
    def wrapper(wrapper_parameters: dict[str, Any], **kwargs) -> dict[str, Any]:

        if wrapper_parameters.get('background_job'):

            background_job = BackgroundJob(subprocess_mode=False)
            result = background_job.execute_function(func, wrapper_parameters)
        else:
            result = func(wrapper_parameters, **kwargs)

        return result

    return wrapper
