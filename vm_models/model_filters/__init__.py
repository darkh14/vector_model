""" Package for model filter classes.
    Modules:
        base_filter - module for base class Model
        ... modules for other services
"""

from typing import Type

from . import base_filter, vbm_filter
from vm_settings import get_var

__all__ = ['base_filter', 'get_fitting_filter_class']

SERVICE_NAME: str = ''

def get_fitting_filter_class() -> Type[base_filter.FittingFilter]:
    """ Function for getting model filter class. Choosing from subclasses of BaseFilter class where
        service name = SERVICE_NAME var
    """
    global SERVICE_NAME

    if not SERVICE_NAME:
        SERVICE_NAME = get_var('SERVICE_NAME')

    filter_classes = [cls for cls in base_filter.FittingFilter.__subclasses__()
                      if cls.service_name == SERVICE_NAME]

    if not filter_classes:
        filter_classes.append(base_filter.FittingFilter)

    return filter_classes[0]
