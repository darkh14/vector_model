""" Module contains list of models (cache). Provides getting model,
    fitting, predicting and other functions to work with models
        Functions:
            _get_model - for getting model
            fit - for fitting model
            predict - for predicting data with model
            initialize - for initializing new model

"""

from typing import Any, Callable
from functools import wraps

from .model import Model, get_model
from vm_logging.exceptions import ModelException, ParameterNotFoundException

MODELS: list[Model] = list()


__all__ = ['fit', 'predict', 'initialize', 'drop', 'update', 'get_info', 'drop_fitting']


def _check_input_parameters(func: Callable):
    @wraps(func)
    def wrapper(parameters: dict[str, Any]):

        if not parameters.get('model'):
            raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

        if not parameters.get('db'):
            raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

        result = func(parameters)
        return result

    return wrapper

@_check_input_parameters
def fit(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For fitting model """
    pass


def predict(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For predicting data with model """
    pass

@_check_input_parameters
def initialize(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new model """
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    model = _get_model(parameters['model'], parameters['db'])

    result = model.initialize(parameters['model'])

    return result


@_check_input_parameters
def drop(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting model from db """
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    model = _get_model(parameters['model'], parameters['db'])

    result = model.drop()

    return result


@_check_input_parameters
def get_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting model info """

    model = _get_model(parameters['model'], parameters['db'])

    result = model.get_info()

    return result


def drop_fitting(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting fit data from model """
    pass


def update(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For updating model parameters """
    pass


def _get_model(input_model: dict[str, Any], db_path: str) -> Model:

    if not input_model.get('id'):
        raise ModelException('Parameter "id" is not found in model parameters')

    models = [c_model for c_model in MODELS if c_model.id == input_model['id']]

    if models:
        model = models[0]
    else:
        model = get_model(input_model['id'], db_path)
        MODELS.append(model)

    return model
