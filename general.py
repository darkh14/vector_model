"""Module for general functions and classes
    functions:
    get_id - for getting random id string
"""

import uuid
from typing import Any
from db_processing.controller import get_connector


def test(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For testing and debugging """
    db_connector = get_connector(parameters['db'])

    # data = db_connector.get_line('test')
    # print(data)
    #
    # data = db_connector.get_line('test', {'author': 'user'})
    # print(data)
    #
    # data = db_connector.get_line('test', {'author': 'user11'})
    # print(data)

    # data = db_connector.get_lines('test', {'author': 'user111'})
    # print(data)

    # data = db_connector.set_line('test', {'author': 'user111', 'title': 'var 5', 'data': 'some data 4'}, {'author': 'user111'})
    # print(data)

    data = db_connector.delete_lines('test')
    #
    # lines = []
    # lines.append({'author': 'admin', 'title': 'var 1', 'data': 'big data'})
    # lines.append({'author': 'admin', 'title': 'var 2', 'data': 'small data'})
    # lines.append({'author': 'user', 'title': 'var 3', 'data': 'user data'})
    #
    # data = db_connector.set_lines('test', lines)

    # lines = []
    # lines.append({'author': 'admin', 'title': 'var 4', 'data': 'ddddd'})
    # lines.append({'author': 'user', 'title': 'var 5', 'data': 'sssss'})
    #
    # data = db_connector.set_lines('test', lines, {'author': 'user'})

    # data = db_connector.set_line('test', {'author': 'admin', 'title': 'var 666', 'data': '666'}, {'author': 'admin'})

    print('Test!')
    return {'result': data}
