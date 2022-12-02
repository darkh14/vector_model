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

from typing import Callable, Any

from . import controller


def get_actions() -> dict[str, Callable]:
    """ forms actions dict available for vm_models"""
    return {
        'model_fit': _fit,
        'model_predict': _predict,
        'model_initialize': _initialize,
        'model_drop': _drop,
        'model_get_info': _get_info,
        'model_drop_fitting': _drop_fitting,
        'model_update': _update
            }


def _fit(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For fitting model """
    pass


def _predict(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For predicting data with model """
    pass


def _initialize(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For initializing new model """
    return controller.initialize(parameters)


def _drop(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting model from db """
    return controller.drop(parameters)


def _get_info(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For getting model info """
    return controller.get_info(parameters)


def _drop_fitting(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For deleting fit data from model """
    pass


def _update(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For updating model parameters """
    pass
