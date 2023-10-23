"""Module for working with db
    classes:
        Connector - for connection with db, getting and setting values in collections (abstract class)
        DBFilter  - for forming filter for inserting, getting, updating to use in db (abstract class)

"""

from abc import ABC, abstractmethod
from typing import Any, Optional, ClassVar, Mapping, Callable, Type
from functools import wraps
import inspect
from vm_logging.exceptions import DBConnectorException
from id_generator import IdGenerator

__all__ = ['Connector', 'DBFilter', 'db_filter_type', 'SETTINGS_DB_NAME']
db_filter_type = Mapping[str, Any]

SETTINGS_DB_NAME = 'vm_settings'


class DBFilter(ABC):
    """ Class for forming filter for getting, setting and deleting data
    Methods:
        set_condition - to set condition of filter
        get_filter - to get filter value for required DB
    """
    def __init__(self, condition: Optional[db_filter_type] = None) -> None:
        """
        Defines _condition value
        :param condition: value of filter
        """
        self._condition = condition

    def set_condition(self, condition: Optional[dict | list] = None) -> None:
        """ for setting condition in current class
        :param condition: condition of filter
        """
        self._condition = condition

    @abstractmethod
    def get_filter(self) -> Any:
        """
        For forming filter (dict, str or other) for desired db
        :return filter value for DB
        """


class Connector:
    """ Abstract class for forming connection to db and operating with db.
    provides working with different types of db
    """
    type: ClassVar[str] = ''
    """type of db. empty for base class"""

    def __init__(self, db_name: str = '', db_path: str = '') -> None:
        """
        Checks parameters, defines local variables and connects to db
        :param db_path: path to db for initializing required connector
        :param db_name: name of db - for information
        """
        if not db_path and not db_name:
            raise DBConnectorException('DB path or DB name must be set!')

        self.settings_connector = self._get_settings_connector(db_name)

        if db_name:
            self._db_name = db_name
            self._db_path = self._get_db_path_by_name()
        else:
            self._db_path: str = db_path
            self._db_name: str = self._get_db_name_by_path()

        self._connection_string: str = self._form_connection_string()
        self._connect()


    def _get_settings_connector(self, db_name: str = '') -> Optional['Connector']:
        return Connector(db_name=SETTINGS_DB_NAME) if self._db_name != SETTINGS_DB_NAME else None


    def _form_connection_string(self) -> str:
        """ Abstract method
        Returns connection string to connect to db
        """
        ...

    def _connect(self) -> bool:
        """ Abstract method
        Connects to db
        """
        ...

    def get_line(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> Optional[dict]:
        """
        For getting one line in collection, supports filter or no filter.
            If filter is defined:
                returns lines found by filter
            Else:
                returns all collection
        :param collection_name: name of required collection
        :param db_filter: optional, filter to find required line
        :return dict of db line
        """
        ...


    def get_lines(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> list[dict]:
        """
        For getting lines in collection or all collection, supports filter or no filter
        :param collection_name: name of required collection
        :param db_filter: optional, filter to find required line
        :return list of db lines
        """
        ...

    def set_line(self, collection_name: str, value: dict[str, Any], db_filter: Optional[db_filter_type] = None) -> bool:
        """
        For setting one line in collection, supports filter or no filter.
            When filter is defined:
                When old line is found by filter:
                    Old line found by filter will be replaced with new line
                else:
                    New line will be added to collection.
            else:
                New line will be added to collection
        :param collection_name: name of required collection
        :param value: value to set as line
        :param db_filter: filter to find required line
        """
        ...

    def set_lines(self, collection_name: str, value: list[dict[str, Any]],
                  db_filter: Optional[db_filter_type] = None) -> bool:
        """
        For setting lines in collection or full collection, supports filter or no filter
        When filter is defined:
            When old lines is found by filter:
                1. All old lines found by filter  will be deleted; 2. New lines will be added
            else:
                New lines will be added
        else:
            New lines will be added
        :param collection_name: name of required collection
        :param value: value list to set as lines
        :param db_filter: filter to find required line
        :return result of lines setting, True if successful
        """
        ...


    def delete_lines(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> bool:
        """
        For deleting lines in collection or full collection, supports filter or no filter
        When filter is defined:
            All lines  found by filter will be deleted
        else:
            Collection will be dropped
        :param collection_name: name of required collection
        :param db_filter: optional, filter to find required lines to delete
        """
        ...

    def get_count(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> int:
        """
        Returns number of documents in collection. Supports filter
        :param collection_name: name of required collection
        :param db_filter: optional, filter to find required lines to count
        """
        ...

    def get_db_path(self) -> str:
        """ for getting db path from inner _db_path value
        :return: string of db path
        """
        return self._db_path

    @property
    def db_name(self):
        """
        Returns value of _db_name. Cannot be set outside __init__
        :return: db_name string
        """
        return self._db_name

    @property
    def db_path(self):
        """
        Returns value of _db_path. Cannot be set outside __init__
        :return: db_path string
        """
        return self._db_path

    def _get_db_name_by_path(self) -> str:
        """ Gets db name by db path. from special settings db.
        :return: db name string
        """
        if not self._db_path:
            return SETTINGS_DB_NAME

        base_settings = self.settings_connector.get_line('bases', {'path': self._db_path})

        if not base_settings:
            raise DBConnectorException('DB with path "{}" does not exist'.format(self._db_path))

        return base_settings['name']

    def _get_db_path_by_name(self) -> str:
        """ Gets db name by db path. from special settings db.
        :return: db name string
        """
        if self._db_name == SETTINGS_DB_NAME:
            return ''

        base_settings = self.settings_connector.get_line('bases', {'name': self._db_name})

        if not base_settings:
            raise DBConnectorException('DB "{}" does not exist'.format(self._db_name))

        return base_settings['path']

    @staticmethod
    def _generate_db_name() -> str:
        """ Generates db name. Random
        :return: generated db name
        """
        return 'db_{}'.format(IdGenerator.get_random_id()[-7:])

    def _get_filter_class(self) -> Type[DBFilter]:
        """ Returns db filter object to convert filter for db
        :return: db filter class
        """
        return DBFilter

    def _get_filter_object(self, db_filter: db_filter_type) -> DBFilter:
        """ Returns db filter object to convert filter for db
        :param db_filter: input dict value of filter
        :return: db filter object
        """
        return self._get_filter_class()(db_filter)

    @staticmethod
    def filter_processing_method(method: Callable) -> Callable:
        """ Decorator for converting and applying filter object
        :param method: method to decorate
        :return: decorated method
        """
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


    def get_collection_names(self) -> list[str]:
        """Method to copy current db,
        :return list of collection names of current db
        """
        ...

    def create_db(self, db_path: str) -> dict[str, Any]:
        if self.db_name != SETTINGS_DB_NAME:
            raise DBConnectorException('New db can be created only with settings db connector')

        base_settings = self.get_line('bases', {'path': db_path})

        if base_settings:
            raise DBConnectorException('DB with path "{}" already exists'.format(db_path))

        c_db_name = self._generate_db_name()
        c_base_settings = {'name': c_db_name, 'path': db_path}

        self.set_line('bases', c_base_settings, {'name': c_db_name})

        return c_base_settings

    def drop_db(self) -> str:
        if self._db_name == SETTINGS_DB_NAME:
            raise DBConnectorException('Settings DB cannot be dropped')

        self.settings_connector.delete_lines('bases', {'name': self._db_name})

        return 'DB "{}" is dropped'.format(self._db_name)
