"""
    Package of loading_engines base abstract module in package -
    real classes must be inherited by base classes in base module
        Modules:
            base_engine - for defining abstract class for loading engine
            vbm_engine - for loading data in "vector budget model"
"""

from typing import Type

from . import base_engine, vbm_engine
from vm_settings import get_var
from vm_logging.exceptions import LoadingProcessException
from .base_engine import BaseEngine

__all__ = ['base_engine', 'vbm_engine', 'get_engine_class', 'BaseEngine']

SERVICE_NAME: str = ''


def get_engine_class() -> Type[base_engine.BaseEngine]:
    """ Function for getting loading engine class. Choosing from subclasses of BaseEngine class where service name =
        SERVICE_NAME var
    """
    global SERVICE_NAME

    if not SERVICE_NAME:
        SERVICE_NAME = get_var('SERVICE_NAME')

    engine_classes = [cls for cls in base_engine.BaseEngine.__subclasses__() if cls.service_name == SERVICE_NAME]

    if not engine_classes:
        raise LoadingProcessException('Can not find loading engine class for service "{}"'.format(SERVICE_NAME))

    return engine_classes[0]
