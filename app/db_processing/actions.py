""" Module for defining actions of db processor package

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

import pandas as pd
import datetime

from typing import Any, Optional
from .controller import check_connection, copy_db, drop_db, create_db, get_db_list, set_collection, get_collection
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
    result.append({'name': 'db_get_collection', 'path': '{db_name}/db/get_collection', 'func': _get_collection,
                   'http_method': 'post', 'requires_authentication': True})
    result.append({'name': 'db_set_collection', 'path': '{db_name}/db/set_collection', 'func': _set_collection,
                   'http_method': 'post', 'requires_authentication': True})

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


def _dict_to_date(sr):

    return datetime.datetime.strptime(sr['$date'], '%Y-%m-%dT%H:%M:%S.%fZ')


def _set_collection(collection_name: str, data: api_types.Collection, replace: Optional[bool] = False,
                    write_dates: Optional[bool] = False):
    data_dict = data.model_dump()

    if write_dates and data_dict['data']:
        pd_data = pd.DataFrame(data_dict['data'])

        data_str = dict(pd_data.iloc[0])

        date_fields = []

        for key, value in data_str.items():
            if isinstance(value, dict) and '$date' in value.keys():
                date_fields.append(key)

        for date_field in date_fields:
            pd_data[date_field] = pd_data[date_field].apply(_dict_to_date)

        data_dict = {'data': pd_data.to_dict(orient='records')}

    result = set_collection(collection_name, data_dict['data'], replace=replace, write_dates=write_dates)

    return result


def _get_collection(collection_name: str, data_filter_body: Optional[api_types.DataFilterBody] = None) -> (
        api_types.Collection):

    data_filter = None
    if data_filter_body is not None:
        data_filter = data_filter_body.model_dump()['data_filter']

    result = get_collection(collection_name, data_filter)

    return api_types.Collection.model_validate({'data': result})
