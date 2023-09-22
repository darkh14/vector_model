""" Package for model parameters and model fitting parameters.
    Modules:
        base_parameters - module for base classes ModelParameters, FittingParameters
        vbm_parameters - module for vbm classes
        ... modules for other services
"""

from typing import Type, Callable

from . import base_parameters, vbm_parameters
from vm_settings import get_var
from vm_logging.exceptions import ModelException

__all__ = ['base_parameters', 'vbm_parameters', 'get_model_parameters_class']

SERVICE_NAME: str = ''


def get_model_parameters_class(fitting: bool = False) -> \
        Type[base_parameters.ModelParameters] | Type[base_parameters.FittingParameters]:
    """ Function for getting model parameters or fitting parameters class.
        Choosing from subclasses of ModelParameters class where
        service name = SERVICE_NAME var
        :param fitting: returns fitting parameters if True else model parameters
        :return: required parameters class
    """
    global SERVICE_NAME

    if not SERVICE_NAME:
        SERVICE_NAME = get_var('SERVICE_NAME')

    subclasses = (base_parameters.FittingParameters.__subclasses__() if fitting
        else base_parameters.ModelParameters.__subclasses__())

    parameters_classes = [cls for cls in subclasses if cls.service_name == SERVICE_NAME]

    if not parameters_classes:
        raise ModelException('Can not find {} parameters class for service "{}"'.format('fitting' if fitting
                                                                                        else 'model', SERVICE_NAME))

    return parameters_classes[0]


def get_additional_actions() -> dict[str, Callable]:
    """
    Gets additional actions of model parameters package
    :return actions dict
    """
    return {}
