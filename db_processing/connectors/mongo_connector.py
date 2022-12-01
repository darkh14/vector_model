"""Module for realisation of base_connector classes. Realises work with MONGO DB.
    classes:
        Connector - for connection with db, getting and setting values in collections
        DBFilter  - for forming filter for inserting, getting, updating to use in db

"""


import pymongo
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError, OperationFailure
from typing import Optional, ClassVar, Callable, Type, Any
from .base_connector import Connector, DBFilter, db_filter_type
from functools import wraps
import vm_settings
from vm_logging.exceptions import DBConnectorException

__all__ = ['MongoConnector', 'MongoFilter']


class MongoFilter(DBFilter):
    """ Class for converting filter for using with MONGO DB"""

    def get_filter(self) -> db_filter_type:
        """ No need to convert """
        return self._condition


class MongoConnector(Connector):
    """ Class realises working with DB using MONGO DB.
        See docs of base class
    """
    type: ClassVar = 'mongo_db'

    def __init__(self, db_path: str = ''):

        super().__init__(db_path)
        self._connection: pymongo.MongoClient = self._get_connection()
        self._db = self._get_db()  # pymongo.database.Database

    def _form_connection_string(self) -> str:
        """Forms connection string from setting vars
        for example 'mongodb://username:password@localhost:27017/?authSource=admin'

        """
        user = vm_settings.get_var('DB_USER')
        password = vm_settings.get_secret_var('DB_PASSWORD')
        host = vm_settings.get_var('DB_HOST')
        port = vm_settings.get_var('DB_PORT')
        auth_source = vm_settings.get_var('DB_AUTH_SOURCE')
        if user:
            result = 'mongodb://{user}:{password}@{host}:{port}/'.format(user=user, password=password,
                                                                         host=host, port=port)
            if auth_source:
                result += '?authSource={auth_source}'.format(auth_source=auth_source)
        else:
            result = 'mongodb://{host}:{port}/'.format(host=host, port=port)

        return result

    @staticmethod
    def safe_db_action(method: Callable):
        """ Provides actions with DB with try-except. Raises DBConnectorException """
        @wraps(method)
        def wrapper(self, *args, **kwargs):

            try:
                result = method(self, *args, **kwargs)
            except ServerSelectionTimeoutError as server_exception:
                raise DBConnectorException('MONGO DB Server connection error! ' + str(server_exception))
            except ConfigurationError as conf_exception:
                raise DBConnectorException('MONGO DB configuration error !' + str(conf_exception))
            except OperationFailure as conf_exception:
                raise DBConnectorException('MONGO DB operation error ! ' + str(conf_exception))

            return result

        return wrapper

    @safe_db_action
    @Connector.filter_processing_method
    def get_line(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> Optional[dict]:
        """ See base method docs """
        collection = self._get_collection(collection_name)

        c_filter = db_filter if db_filter else None
        result = collection.find_one(c_filter, projection={'_id': False})

        return result or None

    @safe_db_action
    @Connector.filter_processing_method
    def get_lines(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> list[dict]:
        """ See base method docs """
        collection = self._get_collection(collection_name)

        c_filter = db_filter if db_filter else None
        result = list(collection.find(c_filter, projection={'_id': False}))

        return result

    @safe_db_action
    @Connector.filter_processing_method
    def set_line(self, collection_name: str, value: dict[str, Any], db_filter: Optional[db_filter_type] = None) -> bool:
        """ See base method docs """
        collection = self._get_collection(collection_name)

        c_filter = db_filter if db_filter else None
        if c_filter:
            result = collection.replace_one(c_filter, value, upsert=True)
        else:
            result = collection.insert_one(value)

        return bool(getattr(result, 'acknowledged'))

    @safe_db_action
    @Connector.filter_processing_method
    def set_lines(self, collection_name: str, value: list[dict[str, Any]],
                  db_filter: Optional[db_filter_type] = None) -> bool:
        """ See base method docs """
        collection = self._get_collection(collection_name)

        c_filter = db_filter if db_filter else None
        result = True
        if c_filter:
            result = collection.delete_many(c_filter)
            result = bool(getattr(result, 'acknowledged'))

        if result:
            result = collection.insert_many(value)

        return bool(getattr(result, 'acknowledged'))

    @safe_db_action
    @Connector.filter_processing_method
    def delete_lines(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> bool:
        """ See base method docs """
        c_filter = db_filter if db_filter else None
        if c_filter:
            collection = self._get_collection(collection_name)
            result = collection.delete_many(c_filter)
            result = bool(getattr(result, 'acknowledged'))
        else:
            result = self._db.drop_collection(collection_name)
            result = result is not None

        return result

    @safe_db_action
    @Connector.filter_processing_method
    def get_count(self, collection_name: str, db_filter: Optional[db_filter_type] = None) -> int:
        """ See base method docs """
        c_filter = db_filter if db_filter else {}

        collection = self._get_collection(collection_name)

        return collection.count_documents(c_filter)

    def _connect(self) -> bool:
        """Doing nothing. In mongo DB connection actually occurs when acting with a collection.
           Before doing this, you need to define the connection object and the db object (in __init__)
        """
        return True

    def _get_filter_class(self) -> Type[DBFilter]:
        """ For getting filter object (to convert filter for DB)"""
        return MongoFilter

    def _get_connection(self) -> pymongo.MongoClient:
        """ Gets db connection object from pymongo """
        try:
            result = pymongo.MongoClient(self._connection_string)
        except ConfigurationError as conf_ex:
            raise ConfigurationError('Configuration error! ' + str(conf_ex))

        return result

    def _get_db(self):
        """ Gets db object from connection object"""
        return self._connection[self._db_name]

    def _get_collection(self, collection_name):
        """ Gets collection object from db object """
        return self._db.get_collection(collection_name)
