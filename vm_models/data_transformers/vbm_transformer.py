
from typing import Any, Optional
import pandas as pd

from .base_transformer import RowColumnTransformer
from ..model_parameters import vbm_parameters

class VbmRowColumnTransformer(RowColumnTransformer):
    service_name = 'vbm'
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        raw_data = x
        raw_data.rename({'organisation': 'organisation_struct', 'scenario': 'scenario_struct', 'period': 'period_str',
                         'indicator': 'indicator_struct'}, axis=1, inplace=True)

        raw_data.rename({'organisation_id': 'organisation', 'scenario_id': 'scenario', 'period_date': 'period',
                         'indicator_short_id': 'indicator'}, axis=1, inplace=True)

        data_grouped = raw_data[['organisation', 'scenario',
                                 'period']].groupby(by=['organisation', 'scenario', 'period'], as_index=False).sum()

        data_result = data_grouped.copy()

        for ind_data in self._model_parameters.x_indicators:

            column_name = 'ind_' + ind_data['short_id']

            ind_data = raw_data[['organisation', 'scenario',
                    'period', 'value']].loc[raw_data['indicator'] == ind_data['short_id']].groupby(by=['organisation',
                                                             'scenario', 'period'], as_index=False).sum()

            data_result = data_result.merge(ind_data, on=['organisation', 'scenario', 'period'], how='left')

            data_result.rename({'value': column_name}, axis=1, inplace=True)

        return data_result
