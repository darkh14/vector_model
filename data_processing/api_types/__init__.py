"""
    Package of data api types. For defining api types. Defines all inner data api types
        Modules:
            base_api_types - for defining abstract api types
            vbm_api_types - for defining api types in "vector budget model"
"""
import enum
import pkgutil
import importlib
import inspect
from pydantic import BaseModel
from typing import Any

from . import base_api_types, vbm_api_types
from vm_settings import SERVICE_NAME

CLASSES_DICT = {}
BASE_CLASSES_DICT = {}


def pred(module_member: Any) -> Any:
    """
    Returns True if module_member is class,
    and it is subclass of pydantic BaseModel,
    or it is subclass of Enum
    """
    return inspect.isclass(module_member) and (BaseModel in module_member.__mro__ or enum.Enum in module_member.__mro__)


for _, modname, is_package in pkgutil.iter_modules(__path__):
    if not is_package:
        imported_module = importlib.import_module('.' + modname, __name__)
        if imported_module.SERVICE_NAME == SERVICE_NAME:
            for class_name, module_class in inspect.getmembers(imported_module, pred):

                CLASSES_DICT[class_name] = module_class


for class_name, module_class in inspect.getmembers(base_api_types, pred):
    BASE_CLASSES_DICT[class_name] = module_class


def __getattr__(name):
    """
    Overloading of getattr to return attribute of base api type
    or sub api type
    """
    if name in CLASSES_DICT:
        return CLASSES_DICT[name]
    elif name in BASE_CLASSES_DICT:
        return BASE_CLASSES_DICT[name]
    else:
        raise AttributeError()
