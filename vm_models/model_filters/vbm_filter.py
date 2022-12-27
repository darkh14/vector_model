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


class VbmFittingFilter(FittingFilter):
    """ Class to transform input filter parameters for reading data
        for fitting to filter, which we can use in db (mongo)
    """
    service_name: ClassVar[str] = 'vbm'

    def __init__(self, filter_value: bytes | dict[str, Any], for_model: bool = False) -> None:
        """
        Defines local filter value
        :param filter_value: input filter value
        :param for_model: parameter says that filter will be used in model, not in fitting
        """
        super().__init__(filter_value, for_model)

        if isinstance(self._value, bytes):
            self._value = pickle.loads(self._value)
        else:
            if self._for_model:
                self._value = self._get_value_db_for_model()
            else:
                self._value = self._transform_period_value(self._value)

    def get_value_as_model_parameter(self) -> bytes:
        """
        Converts value to write it to db as a model parameter. Uses pickle
        :return: value as a model parameter
        """
        return pickle.dumps(self._value, protocol=pickle.HIGHEST_PROTOCOL)

    def get_value_as_json_serializable(self) -> Optional[dict | list | int | float | str | datetime]:
        """
        Convert value to use it as a json serializable value to send it as model info
        :return: value as a json serializable
        """
        result = self._value.copy()

        return self._transform_dates_to_str(result)

    def _get_value_db_for_model(self) -> dict[str, Any]:
        """
        Return value to use as db filter for model
        :return: value as db filter
        """
        data_filter = {}

        for name, value in self._value.items():
            if name in ['organisation', 'scenario']:
                new_value = [el['id'] for el in value]
                data_filter[name + '_id'] = new_value
            else:
                data_filter[name] = value

        filter_list = []

        for name, value in data_filter.items():
            if name == 'date_from':
                filter_el = {'period_date': {'$gte': datetime.strptime(value, '%d.%m.%Y')}}
            elif name == 'date_to':
                filter_el = {'period_date': {'$lte': datetime.strptime(value, '%d.%m.%Y')}}
            elif isinstance(value, list):
                filter_el = {name: {'$in': value}}
            else:
                filter_el = {name: value}

            filter_list.append(filter_el)

        if not filter_list:
            result_filter = {}
        elif len(filter_list) == 1:
            result_filter = filter_list[0]
        else:
            result_filter = {'$and': filter_list}

        return result_filter

    def _transform_period_value(self, value, transform_to_date: bool = False) -> Any:
        """
        Transforms str fields in value as dates. Uses recursion
        :param value: value to transform
        :param transform_to_date: transform current value to date
        :return: transformed value
        """
        if isinstance(value, list):
            result = []
            for el in value:
                result.append(self._transform_period_value(el))
        elif isinstance(value, dict):
            result = {}
            for name, el in value.items():
                if name == 'period_date':
                    result[name] = self._transform_period_value(el, True)
                else:
                    result[name] = self._transform_period_value(el, transform_to_date)
        elif transform_to_date:
            result = datetime.strptime(value, '%d.%m.%Y')
        else:
            result = value

        return result

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
