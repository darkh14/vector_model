""" Module for defining actions of db processor package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Any
from .controller import check_connection, copy_db, drop_db, create_db, get_db_list
from . import api_types


def get_actions() -> list[dict[str, Any]]:
    """ forms actions dict available for db_processing
    :return: list of action descriptions
    """

    result = list()

    result.append({'name': 'db_check_connection', 'path': '{db_name}/db/check_connection', 'func': _check_connection,
                   'http_method': 'get', 'requires_authentication': True})
    result.append({'name': 'db_create', 'path': 'db/create', 'func': _create_db,
                   'http_method': 'post', 'requires_authentication': True})
    result.append({'name': 'db_copy', 'path': '{db_name}/db/copy', 'func': _copy_db,
                   'http_method': 'post', 'requires_authentication': True})
    result.append({'name': 'db_get_all', 'path': 'db/get_all', 'func': _get_db_list,
                   'http_method': 'get', 'requires_authentication': True})
    result.append({'name': 'db_drop', 'path': '{db_name}/db/drop', 'func': _drop_db,
                   'http_method': 'get', 'requires_authentication': True})
    return result


def _check_connection() -> str:
    """ For checking connection
    :return: result of checking connection
    """
    return check_connection()


def _create_db(db_data: api_types.InputDB) -> api_types.OutputDB:
    return api_types.OutputDB.model_validate(create_db(db_data.path))


def _copy_db(db_class: api_types.InputDB) -> api_types.OutputDB:
    """ For checking connection
    :param db_class: data model contains db connection string copy to
    :return: result of checking connection
    """

    return api_types.OutputDB.model_validate(copy_db(db_class.path))


def _get_db_list() -> list[api_types.OutputDB]:
    result = [api_types.OutputDB.model_validate(el) for el in get_db_list()]
    return result


def _drop_db() -> str:
    """
    For checking connection. Raises exception if checking is failed
    :return: str result of checking
    """
    return drop_db()
