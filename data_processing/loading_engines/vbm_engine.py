""" VBM (Vector budget model)
    Module for loading data.

    Classes:
        VbmEngine - main class for loading data (base class is .base_engine.BaseEngine)

"""
from typing import ClassVar, Any
from .base_engine import BaseEngine
from ..data_preprocessors import get_data_preprocessing_class
from ..loading_types import LoadingTypes


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

        pd_data = self.preprocess_data(data, loading_id, package_id)

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

    def preprocess_data(self, data: list[dict[str, Any]], loading_id:  str = '', package_id: str = ''):
        """ Adds additional fields to data array and converts data list to pandas DataFrame.
            Parameters:
                data: list - data array to load;
                loading_id: str - id of loading object, which uses to load data, will be added as field to data;
                package_id: str - id of package object, which uses to load data, will be added as field to data.
        """

        deta_preprocessor = get_data_preprocessing_class()()

        return deta_preprocessor.preprocess_data_for_loading(data, loading_id, package_id)



