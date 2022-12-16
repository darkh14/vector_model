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

    def __init__(self, db_connector: [base_connector.Connector]):
        self._db_connector: base_connector.Connector = db_connector

    @abstractmethod
    def load_data(self, data: list[dict[str, Any]], loading_id:  str, package_id: str,
                  loading_type: LoadingTypes) -> bool: ...
    """ For loading data """

    @abstractmethod
    def delete_data(self, loading_id: str, package_id: str) -> bool: ...
    """ For deleting data """



