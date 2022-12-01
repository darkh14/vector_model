"""Module for working with db
    classes:
        Connector - for connection with db, getting and setting values in collections (abstract class)
        DBFilter  - for forming filter for inserting, getting, updating to use in db (abstract class)

"""

from abc import ABC, abstractmethod
from typing import Any, Optional, ClassVar, Mapping, Callable, Type
from functools import wraps
import inspect
import vm_settings
from vm_logging.exceptions import DBConnectorException
from id_generator import IdGenerator

__all__ = ['Connector', 'DBFilter', 'db_filter_type']
db_filter_type = Mapping[str, Any]


class DBFilter(ABC):
    """ Class for forming filter for getting, setting and deleting data"""
    def __init__(self, condition: Optional[db_filter_type] = None):
        self._condition = condition

    def set_condition(self, condition: Optional[dict | list] = None) -> None:
        """ for setting condition in current class """
        self._condition = condition

    @abstractmethod
    def get_filter(self) -> Any: ...
    """ for forming filter (dict, str or other) for desired db """


class Connector(ABC):
    """ Abstract class for forming connection to db and operating with db.
    provides working with different types of db
    """
    type: ClassVar = ''
    """type of db. empty for base class"""

    def __init__(self, db_path: str = '', db_name: str = ''):

        if not db_path and not db_name:
            raise DBConnectorException('DB path or DB name must be set!')

        if db_name:
            self._db_path = ''
            self._db_name = db_name
        else:
            self._db_path: str = db_path
            self._db_name: str = self._get_db_name_by_path()

        self._connection_string: str = self._form_connection_string()
        self._connect()

    @abstractmethod
    def _form_connection_string(self) -> str: ...

    @abstractmethod
    def _connect(self) -> bool: ...

    @abstractmethod
    def get_line(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> Optional[dict]: ...
    """ For getting one line in collection, supports filter or no filter.
        If filter is defined:
            returns lines found by filter
        Else:
            returns all collection  
    """

    @abstractmethod
    def get_lines(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> list[dict]: ...
    """ For getting lines in collection or all collection, supports filter or no filter"""

    @abstractmethod
    def set_line(self, collection_name: str, value: dict[str, Any], db_filter: Optional[db_filter_type]) -> bool: ...
    """ For setting one line in collection, supports filter or no filter.
        When filter is defined:
            When old line is found by filter:
                Old line found by filter will be replaced with new line
            else:
                New line will be added to collection.                
        else:
            New line will be added to collection.
    """

    def set_lines(self, collection_name: str, value: list[dict[str, Any]],
                  db_filter: Optional[db_filter_type] = None) -> bool: ...
    """ For setting lines in collection or full collection, supports filter or no filter
        When filter is defined:
            When old lines is found by filter:
                1. All old lines found by filter  will be deleted; 2. New lines will be added
            else:
                New lines will be added                
        else:
            New lines will be added
    """

    @abstractmethod
    def delete_lines(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> bool: ...
    """ For deleting lines in collection or full collection, supports filter or no filter
        When filter is defined:
            All lines  found by filter will be deleted
        else:
            Collection will be dropped    
    """

    @abstractmethod
    def get_count(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> int:...
    """ Returns number of documents in collection. Supports filter """

    def get_db_path(self) -> str:
        """ for getting db path from inner _db_path value"""
        return self._db_path

    @property
    def db_name(self):
        return self._db_name

    @property
    def db_path(self):
        return self._db_path

    def _get_db_name_by_path(self) -> str:
        """ Gets db name by db path. USES DB_NAMES setting"""

        db_names = vm_settings.get_var('DB_NAMES')
        if self._db_path in db_names:
            db_name = db_names[self._db_path]
        else:
            db_name = self._generate_db_name_by_path()
            db_names[self._db_path] = db_name

            vm_settings.set_var('DB_NAMES', db_names)

        return db_name

    def _generate_db_name_by_path(self) -> str:
        """ Generates db name from db_path. Not random """
        return 'vm_' + IdGenerator.get_id_by_name(self._db_path)

    @abstractmethod
    def _get_filter_class(self) -> Type[DBFilter]:
        """ Returns db filter object to convert filter for db"""
        return DBFilter

    def _get_filter_object(self, db_filter: db_filter_type) -> DBFilter:
        """ Returns db filter object to convert filter for db"""
        return self._get_filter_class()(db_filter)

    @staticmethod
    def filter_processing_method(method: Callable):
        """ Decorator for converting and applying filter object """
        @wraps(method)
        def wrapper(self, *args, **kwargs):

            temp_args = list(args)

            filter_dict = kwargs.get('db_filter')

            if filter_dict:
                filter_object = self._get_filter_object(filter_dict)
                kwargs['db_filter'] = filter_object.get_filter()
            else:
                arg_names = inspect.getfullargspec(method).args
                for ind in range(len(args)):
                    if arg_names[ind + 1] == 'db_filter':
                        filter_dict = temp_args[ind]
                        filter_object = self._get_filter_object(filter_dict)
                        temp_args[ind] = filter_object.get_filter()

            result = method(self, *temp_args, **kwargs)

            return result

        return wrapper
