""" Package for data transformer classes.
    Modules:
        base_transformer - module for base classes DataTransformer etc.
        vbm_transformer - module for vbm classes
        ... modules for other services
"""

from typing import Type

from . import base_transformer
from vm_settings import get_var
from ..model_types import DataTransformersTypes
from .base_transformer import BaseTransformer
from vm_logging.exceptions import ModelException
from . import vbm_transformer

SERVICE_NAME: str = ''

__all__ = ['base_transformer', 'vbm_transformer', 'get_transformer_class']

def get_transformer_class(transformer_type: DataTransformersTypes) -> Type[BaseTransformer]:
    """ Function for getting data transformer class. Choosing from subclasses of DataTransformer class where
        service name = SERVICE_NAME var
    """
    global SERVICE_NAME

    if not SERVICE_NAME:
        SERVICE_NAME = get_var('SERVICE_NAME')

    transformer_classes = [cls for cls in BaseTransformer.__subclasses__()
                           if cls.service_name == '' and cls.transformer_type == transformer_type]

    if not transformer_classes:
        raise ModelException('Can not find right transformer class with type "{}" '.format(transformer_type.value))

    transformer_class = transformer_classes[0]

    transformer_classes = [cls for cls in transformer_class.__subclasses__()
                      if cls.service_name == SERVICE_NAME and cls.transformer_type == transformer_type]

    if transformer_classes:
        transformer_class = transformer_classes[0]

    return transformer_class