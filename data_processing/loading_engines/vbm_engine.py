""" VBM (Vector budget model)
    Module for loading data.

    Classes:
        VbmEngine - main class for loading data (base class is .base_engine.BaseEngine)

"""
from typing import ClassVar, Any

import pandas as pd

from vm_logging.exceptions import ParametersFormatError
from .base_engine import BaseEngine
from ..data_preprocessors import get_data_preprocessing_class
from ..loading_types import LoadingTypes


class VbmEngine(BaseEngine):
    """ Main class for loading data (base class is .base_engine.BaseEngine)
        Methods:
            load_data - for loading data
            delete_data - for deleting data
    """
    service_name: ClassVar[str] = 'vbm'

    def load_data(self, data: list[dict[str, Any]], loading_id:  str,
                  package_id: str, loading_type: LoadingTypes) -> bool:
        """ Loads data.
        :param data: data array
        :param loading_id: id of loading object, which uses to load data
        :param package_id: id of package object, which uses to load data
        :param loading_type: type of loading - full or increment
        :return rue if loading is successful else False
        """

        self.check_data(data)

        pd_data = self.preprocess_data(data, loading_id, package_id)

        data_to_write = list(pd_data.to_dict('records'))

        if loading_type == LoadingTypes.INCREMENT:
            pd_data_grouped = pd_data[['organisation', 'scenario', 'period', 'indicator_short_id', 'analytics_key_id']]
            for row in pd_data_grouped.iterrows():
                self._db_connector.delete_lines('raw_data', dict(row[1]))

        self._db_connector.set_lines('raw_data', data_to_write)

        return True

    def delete_data(self, loading_id: str, package_id: str) -> bool:
        """ Deletes loaded data
        :param loading_id: id of loading object, which loading data we want to delete
        :param package_id: id of package object, which loading data we want to delete
        :return True if deleting is successful else False
        """
        self._db_connector.delete_lines('raw_data', {'loading_id': loading_id, 'package_id': package_id})
        return True

    def preprocess_data(self, data: list[dict[str, Any]], loading_id:  str = '', package_id: str = '') -> pd.DataFrame:
        """ Adds additional fields to data array and converts data list to pandas DataFrame
        :param data: rqw data array
        :param loading_id: id of loading object, which uses to load data, will be added as field to data
        :param package_id: id of package object, which uses to load data, will be added as field to data
        :return preprocessed data
        """

        data_preprocessor = get_data_preprocessing_class()()

        return data_preprocessor.preprocess_data_for_loading(data, loading_id, package_id)

    def check_data(self, data: list[dict[str, Any]], checking_parameter_name: str = 'data',
                   for_fa: bool = False, **kwargs) -> None:
        """
        Checks raw data: checks fields content in rows of data.
        :param data: data list to check
        :param checking_parameter_name name of checking parameter, which will be displayed in error message
        :param for_fa: True if checking for factor analysis else False
        :param kwargs additional parameters (for inheriting)
        :return None
        """
        wrong_row_numbers = []

        for num, row in enumerate(data):
            match row:
                case {'organisation': {'id': str(), 'name': str()},
                      'scenario': {'id': str(), 'name': str()},
                      'period': str(),
                      'indicator': {'type': str(), 'name': str(), 'id': str()},
                      'analytics': list(r_analytics),
                      'sum': int() | float(),
                      'qty': int() | float(),} if not for_fa:

                    if not self._check_data_analytics(r_analytics):
                        wrong_row_numbers.append(num + 1)

                case {'organisation': {'id': str(), 'name': str()},
                      'is_main_period': bool(),
                      'period': str(),
                      'indicator': {'type': str(), 'name': str(), 'id': str()},
                      'analytics': list(r_analytics),
                      'sum_base': int() | float(),
                      'sum_calculated': int() | float(),
                      'qty_base': int() | float(),
                      'qty_calculated': int() | float()} if for_fa:

                    if not self._check_data_analytics(r_analytics):
                        wrong_row_numbers.append(num + 1)

                case _:
                    wrong_row_numbers.append(num + 1)

            if wrong_row_numbers:
                if len(wrong_row_numbers) > 10:
                    wrong_row_numbers = wrong_row_numbers[:10]
                wrong_row_numbers = [str(el) for el in wrong_row_numbers]
                raise ParametersFormatError('Wrong "{}" parameter format'.format(checking_parameter_name) +
                                            ' in row(s) {}'.format(', '.join(wrong_row_numbers)))

    # noinspection PyMethodMayBeStatic
    def _check_data_analytics(self, analytics: list[dict[str, Any]]) -> bool:
        """
        Method to check format of analytic lis in row of input data
        :param analytics: analytic list to check
        :return: result of checking - True is OK (no errors)
        """
        result = True

        for r_analytics_row in analytics:
            match r_analytics_row:
                case {'kind': str(), 'type': str(), 'name': str(), 'id': str()}:
                    pass
                case _:
                    result = False
                    break

        return result
