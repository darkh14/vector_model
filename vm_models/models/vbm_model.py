""" VBM (Vector Budget model) Contains model class for VBM
"""

from typing import Any

from .base_model import Model

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

