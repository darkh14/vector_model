"""
    Module for working with db connectors. Provides getting db connector and caching connectors

    functions:
        get_connector - for getting connector, see this function docs
"""

from . import connectors
import vm_settings
from typing import Type, Any, Optional
from vm_logging.exceptions import DBConnectorException

CONNECTORS: list[connectors.base_connector.Connector] = []
CURRENT_CONNECTOR: Optional[connectors.base_connector.Connector] = None
DB_TYPE = ''

__all__ = ['get_connector', 'check_connection', 'initialize_connector', 'drop_connector']


def get_connector(db_path: str = '') -> connectors.base_connector.Connector:
    """ Gets correct connector. Tries to get connector from cache (find by db_path).
        If it could not find correct connector, it creates connector by DB_TYPE and add to CONNECTORS cache
        :param db_path: path to required db
        :return: db connector object
    """

    if not CURRENT_CONNECTOR:
        if db_path:
            initialize_connector(db_path)
        else:
            raise DBConnectorException('Can not initialize db connector with empty db path')

    return CURRENT_CONNECTOR


def initialize_connector(db_path: str) -> None:

    global DB_TYPE, CONNECTORS, CURRENT_CONNECTOR

    if not DB_TYPE:
        DB_TYPE = vm_settings.get_var('DB_TYPE')

    c_connectors = [con for con in CONNECTORS if con.get_db_path() == db_path]

    if c_connectors:
        CURRENT_CONNECTOR = c_connectors[0]
    else:
        CURRENT_CONNECTOR = _get_connector_class()(db_path)
        CONNECTORS.append(CURRENT_CONNECTOR)


def drop_connector() -> None:
    """ To drop current connector in the end of request """
    global CURRENT_CONNECTOR
    CURRENT_CONNECTOR = None


def _get_connector_class() -> Type[connectors.base_connector.Connector]:
    """ Chooses right connector from subclasses of base Connector class by DB_TYPE
    :return: db connector class
    """
    global DB_TYPE

    cls_list = [cls for cls in connectors.base_connector.Connector.__subclasses__() if cls.type == DB_TYPE]

    if not cls_list:
        raise DBConnectorException('Can not find correct class for creating db connector. DB_TYPE -{}'.format(DB_TYPE))

    return cls_list[0]


def check_connection(parameters: dict[str, Any]) -> str:
    """
    For checking connection. Raises exception if checking is failed
    :param parameters: dict of request parameters
    :return: str result of checking
    """

    connector = get_connector(parameters['db'])
    line = {'test_field_1': 13, 'test_field_2': 666}
    connector.set_line('test', line.copy())

    line2 = connector.get_line('test', {'test_field_1': 13})

    if line != line2:
        raise DBConnectorException('Read/write error. Read and written lines is not equal')

    connector.delete_lines('test')

    return 'DB connection is OK'
