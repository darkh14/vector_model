"""
    VBM (Vector budget model)
    Module for fitting filter class.
    Classes:
        VbmFittingFilter -  class to transform input filter parameters for reading data
        for fitting to filter, which we can use in db (mongo)
"""

from typing import Any, Optional, ClassVar
from datetime import datetime
import pickle

from .base_filter import FittingFilter

__all__ = ['VbmFittingFilter']

FILTER_TYPE = str | int | float | dict[str, Any] | list[dict[str, Any]]


class VbmFittingFilter(FittingFilter):
    """ Class to transform input filter parameters for reading data
        for fitting to filter, which we can use in db (mongo)
    """
    service_name: ClassVar[str] = 'vbm'

    def __init__(self, filter_value: bytes | dict[str, Any]) -> None:
        """
        Defines local filter value
        :param filter_value: input filter value
        """
        super().__init__(filter_value)

        if isinstance(self._value, bytes):
            self._value = pickle.loads(self._value)
        else:
            self._value = self._transform_filter_to_inner_value(self._value)

    def get_value_as_bytes(self) -> bytes:
        """
        Convert value to bytes
        :return: value as bytes
        """
        return pickle.dumps(self._value, protocol=pickle.HIGHEST_PROTOCOL)

    def get_value_as_json_serializable(self) -> Optional[dict | list | int | float | str | datetime]:
        """
        Convert value to use it as a json serializable value to send it as model info
        :return: value as a json serializable
        """

        return self._value

    # noinspection PyMethodMayBeStatic
    def _transform_filter_to_inner_value(self, value: FILTER_TYPE) -> dict[str, Any]:
        """
        Return value to store it in object
        :return: inner value
        """

        return value
