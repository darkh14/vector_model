"""Module for general functions and classes
    functions:
    test - for testing and debugging
    ping - for testing connection
"""
from datetime import datetime
from typing import Any


def ping() -> dict[str, Any]:
    """ For testing connection
    :return: result of checking connection
    """

    return {'description': 'ping OK', 'current_date': datetime.utcnow()}
