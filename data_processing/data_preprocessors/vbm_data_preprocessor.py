""" VBM (Vector budget model)
    Module for preprocessing data.

    Classes:
        VbmDataPreprocessor - main class for preprocessing data (base class is
            .base_data_preprocessor.BaseDataPreprocessor)

"""

from typing import Any, ClassVar
from datetime import datetime
import pandas as pd

from .base_data_preprocessor import BaseDataPreprocessor
from id_generator import IdGenerator


class VbmDataPreprocessor(BaseDataPreprocessor):
    service_name: ClassVar[str] = 'vbm'

    def preprocess_data_for_predicting(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """ For preprocessing data before predicting

        :param data: list of loading data
        :return: data after preprocessing
        """
        return self._preprocess_data(data)

    def preprocess_data_for_loading(self, data: list[dict[str, Any]],
                                    loading_id: str, package_id: str) -> pd.DataFrame:
        """ For preprocessing data before loading
        :param data: list of loading data
        :param loading_id: id of loading object, which uses to load data, will be added as field to data;
        :param package_id: id of package object, which uses to load data, will be added as field to data.
        :return: data after preprocessing
        """
        return self._preprocess_data(data, loading_id, package_id)

    def _preprocess_data(self, data: list[dict[str, Any]], loading_id:  str = '', package_id: str = '') -> pd.DataFrame:
        """ Adds additional fields to data array and converts data list to pandas DataFrame.
                :param data: list - data array to preprocess
                :param loading_id: str - id of loading object, which uses to load data, will be added as field to data
                :param package_id: str - id of package object, which uses to load data, will be added as field to data
                :return: pd.Dataframe data after preprocessing
        """
        pd_data = pd.DataFrame(data)

        pd_data = self._add_short_ids_to_data(pd_data)
        pd_data['loading_id'] = loading_id
        pd_data['package_id'] = package_id

        if loading_id and package_id:
            pd_data['loading_date'] = datetime.utcnow()

        pd_data['period_date'] = pd_data['period'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y'))

        pd_data['organisation_id'] = pd_data['organisation'].apply(lambda x: x['id'])
        pd_data['organisation_name'] = pd_data['organisation'].apply(lambda x: x['name'])

        pd_data['scenario_id'] = pd_data['scenario'].apply(lambda x: x['id'])
        pd_data['scenario_name'] = pd_data['scenario'].apply(lambda x: x['name'])

        pd_data['indicator_name'] = pd_data['indicator'].apply(lambda x: x['name'])

        pd_data['indicator'] = pd_data[['indicator', 'indicator_short_id']].apply(
            self._add_short_id_to_indicator, axis=1)

        pd_data['analytics'] = pd_data['analytics'].apply(self._add_short_id_to_analytics)

        return pd_data

    def _add_short_ids_to_data(self, pd_data: pd.DataFrame) -> pd.DataFrame:
        """ Adds short ids to indicators, analytics and analytic keys
                :param pd_data: pd.DataFrame - data array
                :return: pd.DataFrame - data array with short ids added
        """
        pd_data['indicator_short_id'] = pd_data['indicator'].apply(IdGenerator.get_short_id_from_dict_id_type)
        pd_data['analytics'] = pd_data['analytics'].apply(self._add_short_id_to_analytics)

        pd_data['analytics_key_id'] = pd_data['analytics'].apply(IdGenerator.get_short_id_from_list_of_dict_short_id)

        return pd_data

    @staticmethod
    def _add_short_id_to_indicator(ind_value: pd.Series) -> dict[str, Any]:
        """ Adds short id to indicator. Used in pandas data series apply() method
                :param ind_value: pd.Series - value of indicator
                :return: value with short id added
        """
        result = ind_value[0]
        result['short_id'] = ind_value[1]
        return result

    @staticmethod
    def _add_short_id_to_analytics(analytics_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """ Adds short id to analytics. Used in pandas data series apply() method
                :param analytics_list: list - array of analytics.
                :return: analytics with short ids added
        """
        for an_el in analytics_list:
            an_el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(an_el)
        return analytics_list
