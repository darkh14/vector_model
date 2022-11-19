"""
    Module for working with db connectors. Provides getting db connector and caching connectors

    functions:
        get_connector - for getting connector, see this function docs
"""

from . import connectors
import vm_settings
from typing import Type
from vm_logging.exceptions import DBConnectorException

CONNECTORS: list[connectors.base_connector.Connector] = []
DB_TYPE = ''

__all__ = ['get_connector']


def get_connector(db_path: str):
    """ Gets correct connector. Tries to get connector from cache (find by db_path).
        If it could not find correct connector, it creates connector by DB_TYPE and add to CONNECTORS cache
    """

    global DB_TYPE, CONNECTORS

    if not DB_TYPE:
        DB_TYPE = vm_settings.get_var('DB_TYPE')

    c_connectors = [con for con in CONNECTORS if con.get_db_path() == db_path]

    if c_connectors:
        c_connector = c_connectors[0]
    else:
        c_connector = _get_connector_class()(db_path)
        CONNECTORS.append(c_connector)

    return c_connector


def _get_connector_class() -> Type[connectors.base_connector.Connector]:
    """ Chooses right connector from subclasses of base Connector class by DB_TYPE"""
    global DB_TYPE

    cls_list = [cls for cls in connectors.base_connector.Connector.__subclasses__() if cls.type == DB_TYPE]

    if not cls_list:
        raise DBConnectorException('Can not find correct class for creating db connector. DB_TYPE -{}'.format(DB_TYPE))

    return cls_list[0]
