""" Module for abstract loading engine class.
    Classes:
        BaseEngine - abstract class for loading data
"""


from typing import ClassVar
from abc import ABC, abstractmethod
from db_processing.connectors import base_connector
from db_processing.controller import get_connector
from ..loading_types import LoadingTypes
from .. import api_types


class BaseEngine(ABC):
    """ Abstract class for loading data. Provides loading and deleting data """
    service_name: ClassVar[str] = ''

    def __init__(self) -> None:
        """
        Initialisation - defines db connector
        """
        self._db_connector: base_connector.Connector = get_connector()

    @abstractmethod
    def load_data(self,
                  data: api_types.PackageWithData,
                  loading_id:  str,
                  package_id: str,
                  loading_type: LoadingTypes,
                  is_first_package: bool = False) -> bool:
        """
        Abstract method for data loading
        :param data: data array to load in db
        :param loading_id: id of data loading
        :param package_id: id of current data package
        :param loading_type: full or increment
        :param is_first_package: True if it is first package of loading
        :return True if loading is successful else False
        """
        ...

    @abstractmethod
    def delete_data(self, loading_id: str, package_id: str) -> bool:
        """
        Abstract method For deleting data
        :param loading_id: id of data loading
        :param package_id: id of data package
        :return True if deleting is successful else False
        """

    @abstractmethod
    def check_data(self, data: api_types.PackageWithData, checking_parameter_name: str = 'data', **kwargs) -> None:
        """
        Checks raw data: checks fields content in rows of data.
        :param data: data list to check
        :param checking_parameter_name name of checking parameter, which will be displayed in error message
        :param kwargs additional parameters
        :return None
        """
