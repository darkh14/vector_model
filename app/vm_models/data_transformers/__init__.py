""" Package for data transformer classes.
    Modules:
        base_transformer - module for base classes DataTransformer etc.
        vbm_transformer - module for vbm classes
        ... modules for other services
"""

from typing import Type, Optional

from . import base_transformer
from vm_settings import get_var
from ..model_types import DataTransformersTypes, ModelTypes
from .base_transformer import BaseTransformer
from vm_logging.exceptions import ModelException
from . import vbm_transformer

SERVICE_NAME: str = ''

__all__ = ['base_transformer', 'vbm_transformer', 'get_transformer_class']


def get_transformer_class(transformer_type: DataTransformersTypes, model_type: ModelTypes) -> Type[BaseTransformer]:
    """ Function for getting data transformer class. Choosing from subclasses of DataTransformer class where
        service name = SERVICE_NAME var
        :param transformer_type: type of required transformer (reader, categorical encoder etc.)
        :param model_type: type of model - optional (some models require special transformers)
        :return: required transformer class
    """
    global SERVICE_NAME

    if not SERVICE_NAME:
        SERVICE_NAME = get_var('SERVICE_NAME')

    transformer_class = _get_class_from_subclasses(BaseTransformer, SERVICE_NAME, model_type, transformer_type)

    if not transformer_class:
        transformer_class = _get_class_from_subclasses(BaseTransformer, SERVICE_NAME, None, transformer_type)

        if not transformer_class:
            transformer_class = _get_class_from_subclasses(BaseTransformer, '', None,
                                                           transformer_type)

            if not transformer_class:
                raise ModelException('Can not find right transformer class' +
                                     'with type "{}" '.format(transformer_type.value))

    return transformer_class


def _get_class_from_subclasses(parent_class: Type[BaseTransformer], service_name: str = '',
                        model_type: Optional[ModelTypes] = None,
                        transformer_type: Optional[DataTransformersTypes] = None) -> Optional[Type[BaseTransformer]]:
    """
    Recursive function to seek required class from subclasses
    :param parent_class: class to get subclasses for seek required class
    :param service_name: service name str for filter cls.service_name == service_name
    :param model_type: model type str for filter cls.model_type == model_type, if not found model_type == ""
    :param transformer_type: transformer type str for filter cls.transformer_type == transformer_type
    :return: required transformer class
    """
    result_class = None

    for subclass in parent_class.__subclasses__():
        if ((not service_name or subclass.service_name == service_name)
                and (not model_type or subclass.model_type == model_type)
                and (not transformer_type or subclass.transformer_type == transformer_type)):
            result_class = subclass

            break

        result_subclass = _get_class_from_subclasses(subclass, service_name, model_type, transformer_type)

        if result_subclass:
            result_class = result_subclass
            break

    return result_class
