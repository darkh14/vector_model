""" Module for defining actions of vm_models package

    functions:
        get_actions() returning dict of actions {request_type, action_object}
        _fit - for fitting model
        _predict - for predicting data with model
        _initialize - for initializing new model
        _drop - for deleting model from db
        _get_info - for getting model info
        _drop_fitting - for deleting fit data of model
        _update - for updating model parameters
"""

__all__ = ['get_actions']

from typing import Any, Optional, Dict

from . import controller
from . import api_types
import api_types as general_api_types
from data_processing import api_types as data_api_types
from fastapi.responses import FileResponse
from fastapi import UploadFile


def get_actions() -> list[dict[str, Any]]:
    """ forms actions dict available for vm_models
    :return: dict of actions (functions)
    """

    result = list()

    result.append({'name': 'model_initialize', 'path': '{db_name}/model/initialize',
                   'func': _initialize,
                   'http_method': 'post', 'requires_authentication': True})

    result.append({'name': 'model_get_info', 'path': '{db_name}/model/get_info',
                   'func': _get_info,
                   'http_method': 'get', 'requires_db': True, 'requires_authentication': True})

    result.append({'name': 'model_drop', 'path': '{db_name}/model/drop',
                   'func': _drop,
                   'http_method': 'get', 'requires_authentication': True})

    result.append({'name': 'model_fit', 'path': '{db_name}/model/fit',
                   'func': _fit,
                   'http_method': 'post', 'requires_authentication': True})

    result.append({'name': 'model_predict', 'path': '{db_name}/model/predict',
                   'func': _predict,
                   'http_method': 'post', 'requires_authentication': True})

    result.append({'name': 'model_drop_fitting', 'path': '{db_name}/model/drop_fitting',
                   'func': _drop_fitting,
                   'http_method': 'get', 'requires_authentication': True})

    result.append({'name': 'model_save', 'path': '{db_name}/model/save',
                   'func': _save_model,
                   'http_method': 'get', 'requires_authentication': True})

    result.append({'name': 'model_load', 'path': '{db_name}/model/load',
                   'func': _load_model,
                   'http_method': 'post', 'requires_authentication': True})

    result.extend(controller.get_additional_actions())

    return result


def _fit(id: str, fitting_parameters: Optional[api_types.FittingParameters] = None,
         background_job: bool = False) -> general_api_types.BackgroundJobResponse:
    """ For fitting model
    :param id: id of model to fit
    :param fitting_parameters: parameters of fitting
    :return: result of fitting
    """
    fitting_parameters_parsed = fitting_parameters.model_dump() if fitting_parameters else None
    return controller.fit(id, fitting_parameters_parsed, background_job=background_job)


def _predict(id: str, inputs: data_api_types.Inputs) -> data_api_types.PredictedOutputs:
    """ For predicting data with model
    :param inputs: list of input data
    :return: predicted data with description
    """

    result = controller.predict(id, inputs.model_dump()['inputs'])
    return data_api_types.PredictedOutputs.model_validate(result)


def _initialize(model: api_types.Model) -> str:
    """ For initializing new model
    :param model: data of new model,
    :return: result of initializing
    """
    return controller.initialize(model)


def _drop(id: str) -> str:
    """ For deleting model from db
    :param id: id of model to drop
    :return: result of dropping
    """
    return controller.drop(id)


def _get_info(id: str) -> api_types.ModelInfo:
    """ For getting model info
    :param id: id of model to get info
    :return: model info
    """
    return controller.get_info(id)


def _drop_fitting(id: str) -> str:
    """ For deleting fit data from model
    :param id: id of model to drop fitting
    :return: result of dropping
    """
    return controller.drop_fitting(id)


def create_temp_file():
    fd, path = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'w') as f:
        f.write('TEST\n')
    try:
        yield path
    finally:
        os.unlink(path)


def _save_model(id: str) -> FileResponse:
    """ Saves ML model as file
    :param id: id of model
    :return: file, contains ML model
    """

    return controller.save_model_file(id)


def _load_model(id: str, model: UploadFile) -> str:
    """ Loads ML model from file
    :param id: id of model
    :param body: file of model
    :return: result of loading
    """
    return controller.load_model_from_file(id, model)
