""" VBM (Vector Budget model) Contains model class for VBM
"""

from typing import Any, Optional
import numpy as np
import pandas as pd

from .base_model import Model
from ..model_types import DataTransformersTypes

class VbmModel(Model):
    service_name = 'vbm'
    def _form_output_columns_description(self) -> dict[str, Any]:

        result_description = {}

        for col_name in self.fitting_parameters.y_columns:

            col_list = col_name.split('_')

            indicators = [ind for ind in self.parameters.y_indicators if ind['short_id'] == col_list[1]]

            indicator = indicators[0] if indicators else None

            analytics = None
            if 'an' in col_list:
                analytic_keys = [key for key in self.fitting_parameters.y_analytic_keys
                                 if key['short_id'] == col_list[3]]

                if analytic_keys:
                    analytics = analytic_keys[0]['analytics']

            result_description[col_name] = {'indicator': indicator, 'analytics': analytics}

        return result_description

    def _y_to_data(self, y: np.ndarray, x_data: pd.DataFrame) ->  pd.DataFrame:
        result = pd.DataFrame(y, columns=self.fitting_parameters.y_columns)

        result[['organisation', 'scenario', 'period']] = x_data[['organisation_struct', 'scenario_struct', 'period']]

        return result


    def _get_model_estimators(self, for_predicting: bool = False,
                              fitting_parameters: Optional[dict[str, Any]] = None) -> list[tuple[str, Any]]:

        estimators = super()._get_model_estimators(for_predicting, fitting_parameters)

        estimators = [es for es in estimators if es[0] != DataTransformersTypes.SCALER.value]
        return estimators