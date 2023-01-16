""" Module for base fitting filter class
    Classes:
        FittingFilter - base class to transform input filter parameters for reading data
            for fitting to filter, which we can use in db
"""

from typing import Any, ClassVar

__all__ = ['FittingFilter']


class FittingFilter:
    """ Base class to transform input filter parameters for reading data
        for fitting to filter, which we can use in db
    """
    service_name: ClassVar[str] = ''

    def __init__(self, filter_value: bytes | dict[str, Any]) -> None:
        """
        Defines local filter value
        :param filter_value: input filter value
        """
        self._value: bytes | dict[str, Any] = filter_value

    def get_value_as_bytes(self) -> bytes:
        """
        Convert value to bytes
        :return: value as bytes
        """
        return self._value

    def get_value_as_db_filter(self) -> dict[str, Any]:
        """
        Convert value to use it as a db filter
        :return: value as a db filter
        """
        return self._value

    def get_value_as_json_serializable(self) -> dict[str, Any]:
        """
        Convert value to use it as a json serializable value to send it as model info
        :return: value as a json serializable
        """
        return self._value
    