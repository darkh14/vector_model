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

    def __init__(self, filter_value: bytes | dict[str, Any], for_model: bool = False) -> None:
        """
        Defines local filter value and local parameter "for_model"
        :param filter_value: input filter value
        :param for_model: parameter says that filter will be used in model, not in fitting
        """
        self._value: bytes | dict[str, Any] = filter_value
        self._for_model: bool = for_model

    def get_value_as_model_parameter(self) -> bytes:
        """
        Convert value to write it to db as a model parameter
        :return: value as a model parameter
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
    