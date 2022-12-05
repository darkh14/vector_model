""" Package for model classes.
    Modules:
        base_model - module for base class Model
        vbm_parameters - module for vbm class
        ... modules for other services
"""

from typing import Type, Callable
import types

from . import base_model, vbm_model
from vm_settings import get_var

__all__ = ['base_model', 'vbm_model', 'get_model_class', 'get_additional_actions']

SERVICE_NAME: str = ''

def get_model_class() -> Type[base_model.Model]:
    """ Function for getting model class. Choosing from subclasses of ModelParameters class where
        service name = SERVICE_NAME var
    """
    global SERVICE_NAME

    if not SERVICE_NAME:
        SERVICE_NAME = get_var('SERVICE_NAME')

    model_classes = [cls for cls in base_model.Model.__subclasses__()
                      if cls.service_name == SERVICE_NAME]

    if not model_classes:
        model_classes.append(base_model.Model)

    return model_classes[0]


def get_additional_actions() -> dict[str, Callable]:

    additional_actions = {}

    package_modules = []
    for mod_name, mod in globals().items():
        if (type(mod) == types.ModuleType
                and mod.__package__ == __name__
                and 'get_additional_actions' in mod.__dir__()):

            package_modules.append(mod)

    for mod in package_modules:
        additional_actions.update(mod.get_additional_actions())

    return additional_actions