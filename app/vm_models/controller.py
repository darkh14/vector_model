""" Module contains list of models (cache). Provides getting model,
    fitting, predicting and other functions to work with models
        Functions:
            _get_model - for getting model
            fit - for fitting model
            predict - for predicting data with model
            initialize - for initializing new model

"""

from typing import Any, Callable, Optional
import os

from vm_models.models import get_model_class
from vm_models.models import get_additional_actions as model_get_actions
from vm_background_jobs.decorators import execute_in_background
from . import api_types
import api_types as general_api_types
from starlette.background import BackgroundTask
from fastapi.responses import FileResponse
from fastapi import UploadFile

__all__ = ['fit', 'predict', 'initialize', 'drop', 'get_info', 'drop_fitting', 'get_additional_actions']


@execute_in_background
def fit(model_id: str, fitting_parameters: Optional[dict[str, Any]], job_id: str = '') -> general_api_types.BackgroundJobResponse:
    """ For fitting model
    :param model_id: id of model to fit
    :param fitting_parameters:  parameters of fitting
    :param job_id: id of job if fitting is background
    :return: result of fitting
    """

    model = get_model_class()(model_id)
    model.fit(fitting_parameters, job_id)

    result = {'description': 'Model is fit', 'mode': general_api_types.ExecutionModes.DIRECTLY, 'pid': 0}
    return general_api_types.BackgroundJobResponse.model_validate(result)


def predict(model_id: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    For predicting data using model
    :param model_id: id of model to predict
    :param inputs: input data
    :return: predicted data with description
    """

    model = get_model_class()(model_id)
    result = model.predict(inputs)

    return result


def initialize(model_data: api_types.Model) -> str:
    """ For initializing new model
    :param model_data: data of new model,
    :return: result of initializing
    """

    model = get_model_class()(model_data.id)
    result = model.initialize(model_data.model_dump())

    return result


def drop(model_id: str) -> str:
    """ For deleting model from db
    :param model_id: id of model to drop
    :return: result of dropping
    """

    model = get_model_class()(model_id)
    result = model.drop()

    return result


def get_info(model_id: str) -> api_types.ModelInfo:
    """ For getting model info
    :param model_id: id of model to get info
    :return: model info
    """

    model = get_model_class()(model_id)
    result = model.get_info()

    return api_types.ModelInfo.model_validate(result)


def drop_fitting(model_id: str) -> str:
    """ For deleting fit data from model
    :param model_id: id of model to drop fitting
    :return: result of dropping
    """

    model = get_model_class()(model_id)
    result = model.drop_fitting()

    return result


def get_additional_actions() -> list[dict[str, Callable]]:
    """
    Forms dict of additional action from model package
    :return: dict of actions (functions)
    """
    return model_get_actions()


def get_action_before_background_job(func_name: str, args: tuple[Any], kwargs: dict[str, Any]) -> Optional[Callable]:
    """Returns function which will be executed before model fit
    @param func_name: name of fit function
    @param args: positional arguments of fit function
    @param kwargs: keyword arguments of fit function.
    @return: function to execute before fit
    """
    model = get_model_class()(args[0])
    result = model.get_action_before_background_job(func_name, args, kwargs)

    return result


def get_action_error_background_job(func_name: str, args: tuple[Any], kwargs: dict[str, Any]) -> Optional[Callable]:
    """Returns function which will be executed while error model fit
    @param func_name: name of fit function
    @param args: positional arguments of fit function
    @param kwargs: keyword arguments of fit function.
    @return: function to execute before fit
    """
    model = get_model_class()(args[0])
    result = model.get_action_error_background_job(func_name, args, kwargs)

    return result


def save_model_file(model_id: str) -> str:

    model = get_model_class()(model_id)
    result = model.get_model_engine_file()

    return FileResponse(path=result, filename='model_{}.zip'.format(model.id), media_type='multipart/form-data',
                        background=BackgroundTask(os.remove, result))


def load_model_from_file(model_id: str, model_data: UploadFile) -> str:

    model = get_model_class()(model_id)
    model.load_model_from_binary(model_data.file, model_data.filename)

    return 'model loaded'

