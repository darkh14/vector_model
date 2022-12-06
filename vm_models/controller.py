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

from vm_models.models import base_model, get_model_class
from vm_models.models import get_additional_actions as model_get_actions
from vm_logging.exceptions import ModelException, ParameterNotFoundException

MODELS: list[base_model.Model] = list()


__all__ = ['fit', 'predict', 'initialize', 'drop', 'get_info', 'drop_fitting', 'get_additional_actions']


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
    model = _get_model(parameters['model'], parameters['db'])

    result = model.fit()

    return result


def predict(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For predicting data with model """
    pass


@_check_input_parameters
def initialize(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new model """
    print('initialize')
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    model = _get_model(parameters['model'], parameters['db'])

    result = model.initialize(parameters['model'])
    print('MODELS - append - 1')
    MODELS.append(model)

    return result


@_check_input_parameters
def drop(parameters: dict[str, Any]) -> str:
    """ For deleting model from db """
    print('drop')
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    model = _get_model(parameters['model'], parameters['db'])

    index = -1
    for c_index, c_model in enumerate(MODELS):
        if c_model.id == model.id:
            index = c_index
            break

    if index != -1:
        print('MODELS - pop')
        MODELS.pop(index)

    result = model.drop()

    del model

    return result


@_check_input_parameters
def get_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting model info """
    print('get_info')
    model = _get_model(parameters['model'], parameters['db'])

    print('-- start initialized "{}" - in db: {}, in MODELS {}'.format(model.initialized, model._db_connector.get_count('models'), len(MODELS)))
    result = model.get_info()
    print('-- end initialized "{}" - in db: {}, in MODELS {}'.format(model.initialized, model._db_connector.get_count('models'), len(MODELS)))

    return result


def drop_fitting(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting fit data from model """
    pass


def get_additional_actions() -> dict[str|Callable]:
    return model_get_actions()

def _get_model(input_model: dict[str, Any], db_path: str) -> base_model.Model:

    if not input_model.get('id'):
        raise ModelException('Parameter "id" is not found in model parameters')

    models = [c_model for c_model in MODELS if c_model.id == input_model['id']]

    if models:
        model = models[0]
    else:
        model = get_model_class()(input_model['id'], db_path)
        if model.initialized:
            print('MODELS - append - 2')
            MODELS.append(model)

    return model