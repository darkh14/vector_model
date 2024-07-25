"""Module for saving and getting current version of service"""

from typing import Final

VERSION: Final = '3.8.0.2'
""" Current version of service """

__all__ = ['get_version']


def get_version() -> str:
    """ Gets current version of service
    :return: version str
    """
    return VERSION
