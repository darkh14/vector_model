"""
    Module for working with db connectors. Provides getting db connector and caching connectors

    functions:
        get_connector - for getting connector, see this function docs
"""

import vm_settings
from typing import Type, Optional, Any
from vm_logging.exceptions import DBConnectorException
from db_processing.connectors.base_connector import Connector, SETTINGS_DB_NAME

CONNECTORS: list[Connector] = []
CURRENT_CONNECTOR: Optional[Connector] = None
DB_TYPE = ''

__all__ = ['get_connector_by_path',
           'get_connector_by_name',
           'get_connector',
           'check_connection',
           'initialize_connector_by_path',
           'initialize_connector_by_name',
           'drop_connector',
           'copy_db',
           'create_db',
           'drop_db',
           'get_db_list',
           'set_collection',
           'get_collection',
           'SETTINGS_DB_NAME']


def get_connector() -> Connector:
    return CURRENT_CONNECTOR


def get_connector_by_path(db_path: str = '', without_caching: bool = False) -> Connector:
    """ Gets correct connector. Tries to get connector from cache (find by db_path).
        If it could not find correct connector, it creates connector by DB_TYPE and add to CONNECTORS cache
        :param db_path: path to required db
        :param without_caching not to cache connector if True
        :return: db connector object
    """
    global DB_TYPE

    if without_caching:
        if not DB_TYPE:
            DB_TYPE = vm_settings.get_var('DB_TYPE')

        return _get_connector_class()(db_path=db_path)

    if not CURRENT_CONNECTOR:
        if db_path:
            initialize_connector_by_path(db_path=db_path)
        else:
            raise DBConnectorException('Can not initialize db connector with empty db path')

    return CURRENT_CONNECTOR


def get_connector_by_name(db_name: str = '', without_caching: bool = False) -> Connector:

    global DB_TYPE

    if without_caching:
        if not DB_TYPE:
            DB_TYPE = vm_settings.get_var('DB_TYPE')

        return _get_connector_class()(db_name=db_name)

    if db_name == SETTINGS_DB_NAME:

        global CONNECTORS

        c_connectors = [con for con in CONNECTORS if con.db_name == db_name]
        if c_connectors:
            return c_connectors[0]
        else:
            if not DB_TYPE:
                DB_TYPE = vm_settings.get_var('DB_TYPE')

            c_connector = _get_connector_class()(db_name=db_name)
            CONNECTORS.append(c_connector)
            return c_connector

    if not CURRENT_CONNECTOR:
        if db_name:
            initialize_connector_by_name(db_name=db_name)
        else:
            raise DBConnectorException('Can not initialize db connector with empty db path')

    return CURRENT_CONNECTOR


def initialize_connector_by_name(db_name: str) -> None:
    """
    Creates db connector by db_name and adds it to CONNECTORS
    @param db_name: name (id) of DB to initialize connector
    @return: None
    """
    _initialize_connector_by_path_name(db_name=db_name)


def initialize_connector_by_path(db_path: str) -> None:
    """
    Creates db connector by db_path and adds it to CONNECTORS
    @param db_path: name of DB to initialize connector
    @return: None
    """
    _initialize_connector_by_path_name(db_path=db_path)


def _initialize_connector_by_path_name(db_path: str = '', db_name: str = '') -> None:
    """
    Creates db connector by db_name or db path and adds it to CONNECTORS
    @param db_name: name (id) of DB to initialize connector
    @param db_path: connection string of DB to initialize connector
    @return: None
    """
    if not db_path and not db_name:
        raise ValueError('db_path or db_name parameter must be defined')

    global DB_TYPE, CONNECTORS, CURRENT_CONNECTOR

    if not DB_TYPE:
        DB_TYPE = vm_settings.get_var('DB_TYPE')

    if db_path:
        c_connectors = [con for con in CONNECTORS if con.db_path == db_path]
    else:
        c_connectors = [con for con in CONNECTORS if con.db_name == db_name]

    if c_connectors:
        CURRENT_CONNECTOR = c_connectors[0]
    else:
        if db_path:
            CURRENT_CONNECTOR = _get_connector_class()(db_path=db_path)
        else:
            CURRENT_CONNECTOR = _get_connector_class()(db_name=db_name)
        CONNECTORS.append(CURRENT_CONNECTOR)


def drop_connector() -> None:
    """ To drop current connector in the end of request """
    global CURRENT_CONNECTOR
    CURRENT_CONNECTOR = None


def _get_connector_class() -> Type[Connector]:
    """ Chooses right connector from subclasses of base Connector class by DB_TYPE
    :return: db connector class
    """
    global DB_TYPE

    cls_list = [cls for cls in Connector.__subclasses__() if cls.type == DB_TYPE]

    if not cls_list:
        raise DBConnectorException('Can not find correct class for creating db connector. DB_TYPE -{}'.format(DB_TYPE))

    return cls_list[0]


def check_connection() -> str:
    """
    For checking connection. Raises exception if checking is failed
    :return: str result of checking
    """

    connector = get_connector()
    line = {'test_field_1': 13, 'test_field_2': 666}
    connector.set_line('test', line.copy())

    line2 = connector.get_line('test', {'test_field_1': 13})

    if line != line2:
        raise DBConnectorException('Read/write error. Read and written lines is not equal')

    connector.delete_lines('test')

    return 'DB connection is OK'


def create_db(db_path: str) -> dict[str, Any]:
    connector = get_connector_by_name(SETTINGS_DB_NAME)
    result = connector.create_db(db_path)

    return result


def copy_db(db_copy_to: str) -> dict[str, Any]:
    """
    For copying db,
    :param db_copy_to: db connection string copy to
    :return: str result of copying
    """

    connector_source = get_connector()
    new_db_dict = connector_source.settings_connector.create_db(db_copy_to)
    connector_receiver = get_connector_by_path(db_path=db_copy_to, without_caching=True)

    collection_names = connector_receiver.get_collection_names()

    if collection_names:
        raise DBConnectorException('DB receiver is also exist')

    collection_names = connector_source.get_collection_names()

    for collection_name in collection_names:
        current_collection = connector_source.get_lines(collection_name)
        connector_receiver.set_lines(collection_name, current_collection)

    return new_db_dict


def get_db_list() -> list[dict[str, Any]]:
    connector = get_connector_by_name(SETTINGS_DB_NAME)
    db_list = connector.get_lines('bases')
    return db_list


def drop_db() -> str:
    """
    For checking connection. Raises exception if checking is failed
    :return: str result of checking
    """

    connector = get_connector()
    result = connector.drop_db()

    return result


def get_collection(collection_name: str, data_filter: dict[str, Any]) -> list[dict[str, Any]]:
    """
    To set any collection in db manually
    :return: str result of setting
    """
    connector = get_connector()

    result = connector.get_lines(collection_name, db_filter=data_filter)

    return result


def set_collection(collection_name: str, data: list[dict[str, Any]], replace: bool = False) -> str:
    """
    To set any collection in db manually
    :return: str result of setting
    """
    connector = get_connector()

    if replace:
        connector.delete_lines(collection_name)

    connector.set_lines(collection_name, data)

    return 'collection "{}" is set'.format(collection_name)
