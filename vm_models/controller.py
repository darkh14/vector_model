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
from vm_background_jobs.decorators import execute_in_background
from .model_filters import get_fitting_filter_class


__all__ = ['fit', 'predict', 'initialize', 'drop', 'get_info', 'drop_fitting', 'get_additional_actions']


def _transform_model_parameters_for_fitting(func: Callable):
    """
    Decorator to transform model parameters
    :param func: function to decorate
    :return: decorated function
    """
    @wraps(func)
    def wrapper(parameters: dict[str, Any]):

        match parameters:
            case {'model': {'fitting_parameters': dict(fitting_parameters)}} if fitting_parameters:

                c_fitting_parameters = fitting_parameters.copy()

                if 'filter' in c_fitting_parameters:
                    input_filter = c_fitting_parameters['filter']
                    filter_obj = get_fitting_filter_class()(input_filter)
                    c_fitting_parameters['filter'] = filter_obj.get_value_as_model_parameter()

                if 'job_id' in parameters:
                    c_fitting_parameters['job_id'] = parameters['job_id']

                parameters['model']['fitting_parameters'] = c_fitting_parameters

            case _:
                raise ParameterNotFoundException('Parameter "fitting_parameters" not found in model parameters')

        result = func(parameters)
        return result

    return wrapper


@_transform_model_parameters_for_fitting
@execute_in_background
def fit(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For fitting model
    :param parameters: request parameters
    :return: result of fitting
    """

    result = None

    match parameters:
        case {'model': {'id': str(model_id), 'fitting_parameters': dict(fitting_parameters)}} if model_id:
            model = get_model_class()(model_id)
            result = model.fit(fitting_parameters)
        case _:
            raise ParameterNotFoundException('Wrong request parameters format. Check "model" parameter')

    return result


def predict(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    For predicting data using model
    :param parameters: request parameters
    :return: predicted data with description
    """

    result = None

    match parameters:
        case {'model': {'id': str(model_id)}, 'inputs': list(inputs)} if model_id and inputs:
            model = get_model_class()(model_id)
            result = model.predict(inputs)
        case _:
            raise ParameterNotFoundException('Wrong request parameters format. Check "model" and inputs parameters')

    return result


def initialize(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new model
    :param parameters: request parameters
    :return: result of initializing
    """

    result = None

    match parameters:
        case {'model': {'id': str(model_id)}} if model_id:
            model = get_model_class()(model_id)
            result = model.initialize(parameters['model'])
        case _:
            raise ParameterNotFoundException('Wrong request parameters format. Check "model" parameter')

    return result


def drop(parameters: dict[str, Any]) -> str:
    """ For deleting model from db
    :param parameters: request parameters
    :return: result of dropping
    """

    result = None

    match parameters:
        case {'model': {'id': str(model_id)}} if model_id:
            model = get_model_class()(model_id)
            result = model.drop()
        case _:
            raise ParameterNotFoundException('Wrong request parameters format. Check "model" parameter')

    return result


def get_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting model info
    :param parameters: request parameters
    :return: model info
    """

    result = None

    match parameters:
        case {'model': {'id': str(model_id)}} if model_id:
            model = get_model_class()(model_id)
            result = model.get_info()
        case _:
            raise ParameterNotFoundException('Wrong request parameters format. Check "model" parameter')

    return result


def drop_fitting(parameters: dict[str, Any]) -> str:
    """ For deleting fit data from model
    :param parameters: request parameters
    :return: result of dropping
    """

    result = None

    match parameters:
        case {'model': {'id': str(model_id)}} if model_id:
            model = get_model_class()(model_id)
            result = model.drop_fitting()
        case _:
            raise ParameterNotFoundException('Wrong request parameters format. Check "model" parameter')

    return result


def get_additional_actions() -> dict[str | Callable]:
    """
    Forms dict of additional action from model package
    :return: dict of actions (functions)
    """
    return model_get_actions()
