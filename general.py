"""Module for general functions and classes
    functions:
    test - for testing and debugging
    ping - for testing connection
"""
from datetime import datetime
from typing import Any


def test(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For testing and debugging
    :param parameters: dict of request parameters
    :return: result of testing
    """

    print('Test!')
    return {'result': 'Test is passed'}


def ping(parameters: dict[str, Any]) -> dict[str, Any]:
    """ For testing connection
    :param parameters: dict of request parameters
    :return: result of checking connection
    """
    return {'description': 'ping OK', 'current_date': datetime.utcnow().strftime('%d.%m.%Y %H:%M:%S')}
