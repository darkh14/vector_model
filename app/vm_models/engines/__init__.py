""" Package for engine classes.
    Modules:
        base_engine - module for base class Engine
        vbm_engine - module for vbm class
        ... modules for other services
"""

from typing import Type, Optional

from . import base_engine, vbm_engine
from vm_settings import get_var
from ..model_types import ModelTypes

__all__ = ['base_engine', 'vbm_engine', 'get_engine_class']

SERVICE_NAME: str = ''


def get_engine_class(model_type: ModelTypes) -> Type[base_engine.BaseEngine]:
    """ Function for getting engine class. Choosing from subclasses of ModelParameters class where
        service name = SERVICE_NAME var
        :param model_type: type of model to get
        :return: required engine class
    """
    global SERVICE_NAME

    if not SERVICE_NAME:
        SERVICE_NAME = get_var('SERVICE_NAME')

    engine_class = _get_class_from_subclasses(base_engine.BaseEngine, SERVICE_NAME, model_type)

    if not engine_class:
        engine_class = base_engine.BaseEngine

    return engine_class


def _get_class_from_subclasses(parent_class: Type[base_engine.BaseEngine], service_name: str = '',
                               model_type: ModelTypes = ModelTypes.NeuralNetwork)\
        -> Optional[Type[base_engine.BaseEngine]]:
    """
    Returns required engine class from subclasses of parent class (recursively)
    :param parent_class: the class from whose subclasses the required class is selected
    :param service_name: filter service_name = csl.service_name
    :param model_type: filter model_type = csl.model_type
    :return: required engine class
    """
    result_class = None

    for subclass in parent_class.__subclasses__():
        if ((not service_name or subclass.service_name == service_name)
                and (not model_type or subclass.model_type == model_type)):

            result_class = subclass

            break

        result_subclass = _get_class_from_subclasses(subclass, service_name, model_type)

        if result_subclass:
            result_class = result_subclass
            break

    return result_class
