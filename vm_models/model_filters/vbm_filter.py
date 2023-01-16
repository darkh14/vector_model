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
        result = self._value.copy()

        return self._transform_dates_to_str(result)

    def _transform_filter_to_inner_value(self, value: FILTER_TYPE) -> dict[str, Any]:
        """
        Return value to store it in object
        :return: inner value
        """
        result_filter = self._transform_filter_recursively(value)

        return result_filter

    def _transform_filter_recursively(self, filter_value: FILTER_TYPE, transform_date: bool = False) -> FILTER_TYPE:
        """
        Converts dates and special keys to mongo format. Recursively
        :param filter_value: value to convert
        :param transform_date: sign that value contains date string
        :return: converted value
        """
        if isinstance(filter_value, dict):
            new_value = {}
            for key, value in filter_value.items():

                if key in ['_in', '_gt', '_lt', '_gte', '_lte', '_and', '_or', '_not']:
                    sub_key = '$' + key[1:]
                    new_value[sub_key] = self._transform_filter_recursively(value, transform_date=transform_date)
                elif key in ['period_date', 'loading_date']:
                    new_value[key] = self._transform_filter_recursively(value, transform_date=True)
                else:
                    new_value[key] = self._transform_filter_recursively(value, transform_date=transform_date)

        elif isinstance(filter_value, list):
            new_value = []
            for el in filter_value:
                sub_value = self._transform_filter_recursively(el, transform_date=transform_date)
                new_value.append(sub_value)
        else:
            if transform_date:
                new_value = datetime.strptime(filter_value, '%d.%m.%Y')
            else:
                new_value = filter_value

        return new_value

    def _transform_dates_to_str(self, value: dict | list | int | float | str | datetime) -> \
            Optional[dict | list | int | float | str | datetime]:
        """
        Transforms date fields in value to str. Uses recursion
        :param value: value to transform
        :return: transformed value
        """
        if isinstance(value, list):
            result = []
            for el in value:
                result.append(self._transform_dates_to_str(el))
        elif isinstance(value, dict):
            result = {}
            for name, val in value.items():
                c_name = name
                if name.startswith('$'):
                    c_name = '_' + name[1:]
                result[c_name] = self._transform_dates_to_str(val)
        elif isinstance(value, datetime):
            result = value.strftime('%d.%m.%Y')
        else:
            result = value

        return result
