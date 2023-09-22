""" VBM (Vector budget model)
    Module for preprocessing data.

    Classes:
        VbmDataPreprocessor - main class for preprocessing data (base class is
            .base_data_preprocessor.BaseDataPreprocessor)

"""

from typing import Any, ClassVar
from datetime import datetime
import pandas as pd
from id_generator import IdGenerator

from .base_data_preprocessor import BaseDataPreprocessor
from .. import api_types


class VbmDataPreprocessor(BaseDataPreprocessor):
    service_name: ClassVar[str] = 'vbm'

    def preprocess_data_for_predicting(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """ For preprocessing data before predicting

        :param data: list of loading data
        :return: data after preprocessing
        """

        pd_data = pd.DataFrame(data)

        return pd_data

    def preprocess_data_for_loading(self, data: api_types.PackageWithData,
                                    loading_id: str, package_id: str) -> pd.DataFrame:
        """ For preprocessing data before loading

        :param data: list of loading data
        :param loading_id: id of loading object, which uses to load data, will be added as field to data;
        :param package_id: id of package object, which uses to load data, will be added as field to data.
        :return: data after preprocessing
        """

        pd_data = pd.DataFrame([el.model_dump() for el in data.data])

        pd_data['loading_id'] = loading_id
        pd_data['package_id'] = package_id

        if loading_id and package_id:
            pd_data['loading_date'] = datetime.utcnow()

        pd_data['row_id'] = pd_data.apply(self.get_raw_data_row_hash, axis=1)

        return pd_data

    def preprocess_additional_data(self, data: api_types.PackageWithData) -> dict[str, pd.DataFrame]:
        """ For preprocess additional data before loading
        :param data: - list of all loading data
        :return: additional data after preprocessing
        """

        additional_data_fields = [el for el in data.get_data_fields() if el != 'data']

        result = dict()

        for field in additional_data_fields:
            data_element = getattr(data, field)
            if data_element:

                result_element = pd.DataFrame([el.model_dump() for el in data_element])
                if field == 'scenarios':
                    result_element['periodicity'] = result_element['periodicity'].apply(lambda x: x.value)
                result[field] = result_element
        return result

    @staticmethod
    def get_raw_data_row_hash(data_row: pd.Series) -> str:
        """
        For hash from data row (indicator, period and analytics) to form id of data row
        this id helps to choose and delete specific rows while loading data
        :param data_row: row of data to hash it
        :return result of data hashing
        """
        hash_str = ' '.join([data_row['period'].strftime('%d.%m.%Y %H:%M:%S'),
                            data_row['organisation'],
                            data_row['scenario'],
                            data_row['indicator'],
                            data_row['analytic_key']])
        return IdGenerator.get_id_by_name(hash_str)
