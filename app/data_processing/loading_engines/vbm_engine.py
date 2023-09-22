""" VBM (Vector budget model)
    Module for loading data.

    Classes:
        VbmEngine - main class for loading data (base class is .base_engine.BaseEngine)

"""
from typing import ClassVar, Any

import pandas as pd

from .base_engine import BaseEngine
from ..data_preprocessors import get_data_preprocessing_class
from ..loading_types import LoadingTypes
from .. import api_types


class VbmEngine(BaseEngine):
    """ Main class for loading data (base class is .base_engine.BaseEngine)
        Methods:
            load_data - for loading data
            delete_data - for deleting data
    """
    service_name: ClassVar[str] = 'vbm'

    def load_data(self,
                  data: api_types.PackageWithData,
                  loading_id:  str,
                  package_id: str,
                  loading_type: LoadingTypes,
                  is_first_package: bool = False) -> bool:
        """ Loads data.
        :param data: all data package
        :param loading_id: id of loading object, which uses to load data
        :param package_id: id of package object, which uses to load data
        :param loading_type: type of loading - full or increment
        :param is_first_package: True if it is first package of loading
        :return rue if loading is successful else False
        """

        self.check_data(data)

        pd_data = self.preprocess_data(data, loading_id, package_id)

        additional_data_dict = self.preprocess_additional_data(data)

        if loading_type == LoadingTypes.INCREMENT:
            row_ids = list(pd_data['row_id'].unique())
            self._db_connector.delete_lines('raw_data', {'row_id': {'$in': row_ids}})
        elif is_first_package:
            self._db_connector.delete_lines('raw_data')
            self._delete_additional_data()

        data_to_write = pd_data.loc[pd_data['remove'] == False]

        data_to_write.drop('remove', axis=1, inplace=True)
        data_to_write = list(data_to_write.to_dict('records'))

        if data_to_write:
            self._db_connector.set_lines('raw_data', data_to_write)

        self._load_additional_data(additional_data_dict)

        return True

    def _load_additional_data(self, additional_data: dict[str, pd.DataFrame]) -> None:

        for name, data_element in additional_data.items():

            field_to_group = 'key' if name == 'analytic_keys' else 'id'

            group_values = list(data_element[field_to_group].unique())
            # pd_data_grouped = data_element[[field_to_group]]
            # for row in pd_data_grouped.iterrows():
            #     self._db_connector.delete_lines(name, dict(row[1]))
            self._db_connector.delete_lines(name, {field_to_group: {'$in': group_values}})

            self._db_connector.set_lines(name, list(data_element.to_dict('records')))

    def _delete_additional_data(self):

        fields = [el for el in api_types.PackageWithData.get_data_fields() if el != 'data']

        for field in fields:
            self._db_connector.delete_lines(field)

    def delete_data(self, loading_id: str, package_id: str) -> bool:
        """ Deletes loaded data
        :param loading_id: id of loading object, which loading data we want to delete
        :param package_id: id of package object, which loading data we want to delete
        :return True if deleting is successful else False
        """
        self._db_connector.delete_lines('raw_data',
                                        {'loading_id': loading_id, 'package_id': package_id})

        return True

    # noinspection PyMethodMayBeStatic
    def preprocess_data(self, data: api_types.PackageWithData,
                        loading_id:  str = '',
                        package_id: str = '') -> pd.DataFrame:
        """ Adds additional fields to data array and converts data list to pandas DataFrame
        :param data: package with data
        :param loading_id: id of loading object, which uses to load data, will be added as field to data
        :param package_id: id of package object, which uses to load data, will be added as field to data
        :return preprocessed data
        """

        data_preprocessor = get_data_preprocessing_class()()

        return data_preprocessor.preprocess_data_for_loading(data, loading_id, package_id)

    # noinspection PyMethodMayBeStatic
    def preprocess_additional_data(self, data: api_types.PackageWithData) -> dict[str, pd.DataFrame]:

        data_preprocessor = get_data_preprocessing_class()()

        return data_preprocessor.preprocess_additional_data(data)

    def check_data(self, data: api_types.PackageWithData, checking_parameter_name: str = 'data',
                   for_fa: bool = False, **kwargs) -> None:
        """
        Checks raw data: checks fields content in rows of data.
        :param data: data list to check
        :param checking_parameter_name name of checking parameter, which will be displayed in error message
        :param for_fa: True if checking for factor analysis else False
        :param kwargs additional parameters (for inheriting)
        :return None
        """
        # checking data format is in fastapi and pydatic
        pass

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
