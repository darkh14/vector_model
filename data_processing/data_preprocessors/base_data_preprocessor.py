""" Module for abstract data preprocessing class.
    Classes:
        BaseDataPreprocessor - abstract class for preprocessing data
"""

from typing import ClassVar, Any
import pandas as pd


class BaseDataPreprocessor:
    """ Abstract class for preprocessing data. Providing primary data processing.
        Applies before fitting data loading and before input data predicting
    """
    service_name: ClassVar[str] = ''

    def __init__(self, **kwargs) -> None: ...

    def preprocess_data_for_loading(self, data: list[dict[str, Any]], loading_id:
                                    str, package_id: str) -> pd.DataFrame:
        """ For preprocess data before loading
        :param data: - list of loading data
        :param loading_id: current id of loading
        :param package_id: current id of package
        :return: data after preprocessing
        """
        return pd.DataFrame(data)

    def preprocess_data_for_predicting(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """ For preprocess data before predicting
        :param data: - list oof input data
        :return: data after preprocessing
        """
        return pd.DataFrame(data)
