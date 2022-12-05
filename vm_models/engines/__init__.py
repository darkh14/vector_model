""" Package for engine classes.
    Modules:
        base_engine - module for base class Engine
        vbm_engine - module for vbm class
        ... modules for other services
"""

from typing import Type

from . import base_engine, vbm_engine
from vm_settings import get_var

__all__ = ['base_engine', 'vbm_engine', 'get_engine_class']

SERVICE_NAME: str = ''

def get_engine_class() -> Type[base_engine.Engine]:
    """ Function for getting engine class. Choosing from subclasses of ModelParameters class where
        service name = SERVICE_NAME var
    """
    global SERVICE_NAME

    if not SERVICE_NAME:
        SERVICE_NAME = get_var('SERVICE_NAME')

    engine_classes = [cls for cls in base_engine.Engine.__subclasses__()
                      if cls.service_name == SERVICE_NAME]

    if not engine_classes:
        engine_classes.append(base_engine.Engine)

    return engine_classes[0]
