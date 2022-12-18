""" VBM (Vector Budget model) Contains model class for VBM
"""

from typing import Any, Optional, Callable
import numpy as np
import pandas as pd

from .base_model import Model
from ..model_types import DataTransformersTypes
from vm_logging.exceptions import ModelException, ParameterNotFoundException
from ..model_filters import get_fitting_filter_class

__all__ = ['VbmModel', 'get_additional_actions']

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

    def calculate_feature_importances(self, fi_parameters: dict[str, Any]) -> dict[str, Any]:

        self._check_before_fi_calculating(fi_parameters)

        self.fitting_parameters.set_start_fi_calculation()
        self._write_to_db()

        try:
            result = self._fi_calculate_model(fi_parameters['epochs'], fi_parameters)
        except Exception as ex:
            self.fitting_parameters.set_error_fi_calculation(str(ex))
            self._write_to_db()
            raise ex

        if not self.fitting_parameters.fi_calculation_is_error:

            # self.fitting_parameters.metrics = self._engine.metrics

            self.fitting_parameters.set_end_fi_calculation()
            self._write_to_db()

        return result

    def _check_before_fi_calculating(self, fi_parameters:  dict[str, Any]) -> None:

        if not self.fitting_parameters.is_fit:
            raise ModelException('Model is not fit. Fit model before fi calculation')

        if 'epochs' not in fi_parameters:
            raise ModelException('Parameter "epochs" not found in fi parameters')

        if self.fitting_parameters.fi_calculation_is_started:
            raise ModelException('Another fi calculation is started yet. Wait for end of fi calculation')

    def _fi_calculate_model(self, epochs, fi_parameters):

        # pipeline = self._get_model_pipeline(for_predicting=False, fitting_parameters=fitting_parameters)
        # data = pipeline.fit_transform(None)
        #
        # x, y = self._data_to_x_y(data)
        # input_number = len(self.fitting_parameters.x_columns)
        # output_number = len(self.fitting_parameters.y_columns)
        # self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number, self._db_path,
        #                                                       self.fitting_parameters.is_first_fitting())
        # result = self._engine.fit(x, y, epochs, fitting_parameters)

        # if not retrofit:
        #     date_from = None
        # else:
        #     date_from = datetime.datetime.strptime(date_from, '%d.%m.%Y')
        #
        # indicator_filter = [ind_data['short_id'] for ind_data in self.x_indicators + self.y_indicators]
        #
        # db_filter = {key: value for key, value in self.filter.items() if key not in ['date_from', 'date_to']}
        #
        # data = self._data_processor.read_raw_data(indicator_filter, date_from=date_from, ad_filter=db_filter)
        # additional_data = {'x_indicators': self.x_indicators,
        #                    'y_indicators': self.y_indicators,
        #                    'periods': self.periods,
        #                    'organisations': self.organisations,
        #                    'scenarios': self.scenarios,
        #                    'x_columns': self.x_columns,
        #                    'y_columns': self.y_columns,
        #                    'filter': self.filter}
        # x, y, x_columns, y_columns = self._data_processor.get_x_y_for_fitting(data, additional_data)
        #
        # self._temp_input = x
        # # self._inner_model = self._get_inner_model(len(self.x_columns), len(self.y_columns), retrofit=retrofit)
        #
        # epochs = epochs or 1000
        # validation_split = validation_split or 0.2
        #
        # fi_model = KerasRegressor(build_fn=self._get_model_for_feature_importances,
        #                           epochs=epochs,
        #                           verbose=2,
        #                           validation_split=validation_split)
        # fi_model.fit(x, y)
        #
        # fi = self._calculate_fi_from_model(fi_model, x, y, x_columns)

        result = {'description': 'FI calculating OK'}

        return result


def get_additional_actions() -> dict[str, Callable]:
    return {'model_calculate_feature_importances': _calculate_feature_importances,
            'model_get_feature_importances': _get_feature_importances
            }


def _calculate_feature_importances(parameters: dict[str, Any]) -> dict[str, Any]:
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    if 'fi_parameters' not in parameters['model']:
        raise ParameterNotFoundException('Parameter "fi_parameters" not found in model parameters')

    if 'fi_parameters' in parameters['model']:
        if 'filter' in parameters['model']['fi_parameters']:
            input_filter = parameters['model']['fi_parameters']['filter']
            filter_obj = get_fitting_filter_class()(input_filter)
            parameters['model']['fi_parameters']['filter'] = filter_obj.get_value_as_model_parameter()

        if 'job_id' in parameters:
            parameters['model']['fi_parameters']['job_id'] = parameters['job_id']

    model = VbmModel(parameters['model'], parameters['db'])

    result = model.calculate_feature_importances(parameters['model']['fi_parameters'])

    return result


def _get_feature_importances(parameters: dict[str, Any]) -> dict[str, Any]:
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    return {'FI': []}