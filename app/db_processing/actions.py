""" Module for defining actions of db processor package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Any
from .controller import check_connection, copy_db, drop_db
from . import api_types


def get_actions() -> list[dict[str, Any]]:
    """ forms actions dict available for db_processing
    :return: list of action descriptions
    """

    result = list()

    result.append({'name': 'db_check_connection', 'path': 'db/check_connection', 'func': _check_connection,
                   'http_method': 'get', 'requires_db': True, 'requires_authentication': True})
    result.append({'name': 'db_copy', 'path': 'db/copy', 'func': _copy_db,
                   'http_method': 'post', 'requires_db': True, 'requires_authentication': True})
    result.append({'name': 'db_drop', 'path': 'db/drop', 'func': _drop_db,
                   'http_method': 'get', 'requires_db': True, 'requires_authentication': True})
    return result


def _check_connection() -> str:
    """ For checking connection
    :return: result of checking connection
    """
    return check_connection()


def _copy_db(db_class: api_types.DBCopyTo) -> str:
    """ For checking connection
    :param db_class: data model contains db connection string copy to
    :return: result of checking connection
    """

    return copy_db(db_class.db_copy_to)


def _drop_db() -> str:
    """
    For checking connection. Raises exception if checking is failed
    :return: str result of checking
    """
    return drop_db()
