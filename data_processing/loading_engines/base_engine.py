""" Module for abstract loading engine class.
    Classes:
        BaseEngine - abstract class for loading data
"""


from typing import ClassVar, Any
from abc import ABC, abstractmethod
from db_processing.connectors import base_connector
from ..loading_types import LoadingTypes


class BaseEngine(ABC):
    """ Abstract class for loading data. Provides loading and deleting data """
    service_name: ClassVar[str] = ''

    def __init__(self, db_connector: [base_connector.Connector]) -> None:
        """
        Initialisation - defines db connector
        :param db_connector: db connector object  for working with db
        """
        self._db_connector: base_connector.Connector = db_connector

    @abstractmethod
    def load_data(self, data: list[dict[str, Any]], loading_id:  str, package_id: str,
                  loading_type: LoadingTypes) -> bool:
        """
        Abstract method for data loading
        :param data: data array to load in db
        :param loading_id: id of data loading
        :param package_id: id of current data package
        :param loading_type: full or increment
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
        ...



