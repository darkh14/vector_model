"""
    Package of data preprocessors. For data preprocessing. Performs preprocessing
    data before loading. And preprocessing input predicting data
        Modules:
            base_data_preprocessor - for defining abstract class for data preprocessor
            vbm_data_preprocessor - for preprocessing data in "vector budget model"
"""

from typing import Type

from . import base_data_preprocessor, vbm_data_preprocessor
from vm_settings import SERVICE_NAME
from vm_logging.exceptions import LoadingProcessException
from .base_data_preprocessor import BaseDataPreprocessor

__all__ = ['base_data_preprocessor', 'vbm_data_preprocessor', 'get_data_preprocessing_class', 'BaseDataPreprocessor']


def get_data_preprocessing_class() -> Type[BaseDataPreprocessor]:
    """ Function for getting loading engine class. Choosing from subclasses of BaseEngine class where service name =
        SERVICE_NAME var
        :return: required class for preprocessing data
    """

    data_preprocessing_classes = [cls for cls in base_data_preprocessor.BaseDataPreprocessor.__subclasses__()
                                  if cls.service_name == SERVICE_NAME]

    if not data_preprocessing_classes:
        raise LoadingProcessException('Can not find '
                                      'data preprocessing class for service "{}"'.format(SERVICE_NAME))

    return data_preprocessing_classes[0]
