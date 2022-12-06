""" VBM (Vector budget model)
    Module for loading data.

    Classes:
        VbmEngine - main class for loading data (base class is .base_engine.BaseEngine)

"""
from typing import ClassVar, Any
from .base_engine import BaseEngine
from id_generator import IdGenerator
from ..loading_types import LoadingTypes

import pandas as pd
from datetime import datetime


class VbmEngine(BaseEngine):
    """ Main class for loading data (base class is .base_engine.BaseEngine)
        Methods:
            load_data - for loading data
            delete_data - for deleting data
    """
    service_name: ClassVar = 'vbm'

    def load_data(self, data: list[dict[str, Any]], loading_id:  str,
                  package_id: str, loading_type: LoadingTypes):
        """ Loads data.
            Parameters:
                data: list - data array;
                loading_id: str - id of loading object, which uses to load data
                package_id: str - id of package object, which uses to load data
                loading_type: str - type of loading

        """

        pd_data = self._preprocess_data(data, loading_id, package_id)

        data_to_write = list(pd_data.to_dict('records'))

        if loading_type == LoadingTypes.INCREMENT:
            pd_data_grouped = pd_data[['organisation', 'scenario', 'period', 'indicator_short_id', 'analytics_key_id']]
            for row in pd_data_grouped.iterrows():
                self._db_connector.delete_lines('raw_data', dict(row[1]))

        self._db_connector.set_lines('raw_data', data_to_write)

    def delete_data(self, loading_id: str, package_id: str) -> bool:
        """ Deletes data.
            Parameters:
                loading_id: str - id of loading object, which loading data we want to delete
                package_id: str - id of package object, which loading data we want to delete

        """
        self._db_connector.delete_lines('raw_data', {'loading_id': loading_id, 'package_id': package_id})
        return True

    def _preprocess_data(self, data: list[dict[str, Any]], loading_id:  str, package_id: str):
        """ Adds additional fields to data array and converts data list to pandas DataFrame.
            Parameters:
                data: list - data array to load;
                loading_id: str - id of loading object, which uses to load data, will be added as field to data;
                package_id: str - id of package object, which uses to load data, will be added as field to data.
        """
        pd_data = pd.DataFrame(data)

        pd_data = self._add_short_ids_to_data(pd_data)
        pd_data['loading_id'] = loading_id
        pd_data['package_id'] = package_id

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
            Parameters:
                pd_data: pandas.DataFrame - data array
        """
        pd_data['indicator_short_id'] = pd_data['indicator'].apply(IdGenerator.get_short_id_from_dict_id_type)
        pd_data['analytics'] = pd_data['analytics'].apply(self._add_short_id_to_analytics)

        pd_data['analytics_key_id'] = pd_data['analytics'].apply(IdGenerator.get_short_id_from_list_of_dict_short_id)

        return pd_data

    @staticmethod
    def _add_short_id_to_analytics(analytics_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """ Adds short id to analytics.
            Parameters:
                analytics_list: list - array of analytics
        """
        for an_el in analytics_list:
            an_el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(an_el)
        return analytics_list

    @staticmethod
    def _add_short_id_to_indicator(ind_value: pd.Series) -> dict[str, Any]:
        """ Adds short id to indicator.
            Parameters:
                ind_value: pandas.Series - value of indicator
        """
        result = ind_value[0]
        result['short_id'] = ind_value[1]
        return result
