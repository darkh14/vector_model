""" VBM (Vector Budget model) Contains model class for VBM
"""

from typing import Any, Optional, Callable
import numpy as np
import pandas as pd
import math

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import clone_model
from eli5.sklearn import PermutationImportance

from .base_model import Model
from ..model_types import DataTransformersTypes
from vm_logging.exceptions import ModelException, ParameterNotFoundException
from ..model_filters import get_fitting_filter_class
from ..engines import get_engine_class
from vm_background_jobs.decorators import execute_in_background
from vm_background_jobs.controller import set_background_job_interrupted



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

        self.fitting_parameters.set_start_fi_calculation(fi_parameters)
        self._write_to_db()

        try:
            result = self._fi_calculate_model(fi_parameters['epochs'], fi_parameters)
        except Exception as ex:
            self.fitting_parameters.set_error_fi_calculation(str(ex))
            self._write_to_db()
            raise ex

        if not self.fitting_parameters.fi_calculation_is_error:

            self.fitting_parameters.set_end_fi_calculation()
            self._write_to_db()

        return result

    def get_feature_importances(self) -> dict[str, Any]:

        if not self.fitting_parameters.fi_is_calculated:
            raise ModelException('FI is not calculated. Calculate feature importances first')

        return self.fitting_parameters.feature_importances

    def drop_fi_calculation(self) -> str:

        self._interrupt_fi_calculation_job()
        self.fitting_parameters.set_drop_fi_calculation()
        self._write_to_db()

        return 'Model "{}" id "{}" fi calculation is dropped'.format(self.parameters.name, self.id)

    def get_sensitivity_analysis(self, inputs_0: list[dict[str, Any]],
                                 inputs_plus: list[dict[str, Any]],
                                 inputs_minus: list[dict[str, Any]]) -> dict[str, Any]:

        pipeline = self._get_model_pipeline(for_predicting=True)
        data_0 = pipeline.transform(inputs_0)

        data_plus = pipeline.transform(inputs_plus)
        data_minus = pipeline.transform(inputs_minus)

        x_0 = self._data_to_x(data_0)

        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)
        self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number, self._db_path)
        y_0 = self._engine.predict(x_0)

        data_0[self.fitting_parameters.y_columns] = y_0

        data_0['y_all'] = data_0[self.fitting_parameters.y_columns].apply(sum, axis=1)

        ind_ids = list(set([ind['short_id'] for ind in self.parameters.x_indicators]))

        sa = {}

        for ind_id in ind_ids:
            ind_columns = [col for col in self.fitting_parameters.x_columns if col.split('_')[1] == ind_id]

            data_ind_plus = data_0.copy()
            data_ind_plus[ind_columns] = data_plus[ind_columns]
            x_ind_plus = self._data_to_x(data_ind_plus)

            data_ind_minus = data_0.copy()
            data_ind_minus[ind_columns] = data_minus[ind_columns]
            x_ind_minus = self._data_to_x(data_ind_minus)


            y_ind_plus = self._engine.predict(x_ind_plus)
            y_ind_minus = self._engine.predict(x_ind_minus)

            data_ind_plus[self.fitting_parameters.y_columns] = y_ind_plus
            data_ind_minus[self.fitting_parameters.y_columns] = y_ind_minus

            data_ind_plus['y_all'] = data_ind_plus[self.fitting_parameters.y_columns].apply(sum, axis=1)
            data_ind_minus['y_all'] = data_ind_minus[self.fitting_parameters.y_columns].apply(sum, axis=1)

            data_ind_plus['y_0'] = data_0['y_all']
            data_ind_minus['y_0'] = data_0['y_all']

            data_ind_plus['delta'] = data_ind_plus['y_all'] - data_ind_plus['y_0']
            data_ind_minus['delta'] = data_ind_minus['y_0'] - data_ind_minus['y_all']

            data_ind_plus['delta_percent'] = data_ind_plus[['delta', 'y_0']].apply(lambda ss: ss['delta']/ss['y_0']
                                                                                   if ss['y_0'] else 0, axis=1)
            data_ind_minus['delta_percent'] = data_ind_minus[['delta', 'y_0']].apply(lambda ss: ss['delta']/ss['y_0']
                                                                                   if ss['y_0'] else 0, axis=1)
            data_ind_plus = data_ind_plus.drop(['organisation', 'scenario'], axis=1)
            data_ind_minus = data_ind_minus.drop(['organisation', 'scenario'], axis=1)

            data_ind_plus = data_ind_plus.rename({'organisation_struct': 'organisation', 'scenario_struct':
                'scenario', 'y_all': 'y'}, axis=1)

            data_ind_minus = data_ind_minus.rename({'organisation_struct': 'organisation', 'scenario_struct':
                'scenario', 'y_all': 'y'}, axis=1)

            sa_ind_data = dict()
            sa_ind_data['indicator'] = [ind for ind in self.parameters.x_indicators if ind['short_id'] == ind_id][0]

            sa_ind_data_plus = data_ind_plus[['organisation', 'scenario', 'period', 'y', 'y_0',
                                              'delta', 'delta_percent']].to_dict('records')
            sa_ind_data_minus = data_ind_minus[['organisation', 'scenario', 'period', 'y', 'y_0',
                                              'delta', 'delta_percent']].to_dict('records')

            sa_ind_data['plus'] = sa_ind_data_plus
            sa_ind_data['minus'] = sa_ind_data_minus

            sa['ind_' + ind_id] = sa_ind_data

        return sa

    def _check_before_fi_calculating(self, fi_parameters:  dict[str, Any]) -> None:

        if not self._initialized:
            raise ModelException('Model is not initialized. Check model id before')

        if not self.fitting_parameters.is_fit:
            raise ModelException('Model is not fit. Fit model before fi calculation')

        if 'epochs' not in fi_parameters:
            raise ModelException('Parameter "epochs" not found in fi parameters')

        if self.fitting_parameters.fi_calculation_is_started:
            raise ModelException('Another fi calculation is started yet. Wait for end of fi calculation')

    def _fi_calculate_model(self, epochs, fi_parameters):

        pipeline = self._get_model_pipeline(for_predicting=False, fitting_parameters=fi_parameters)
        data = pipeline.fit_transform(None)

        x, y = self._data_to_x_y(data)
        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)
        self._engine = get_engine_class(self.parameters.type)('', input_number, output_number, self._db_path, True)

        validation_split = fi_parameters.get('validation_split') or 0.2

        fi_engine = KerasRegressor(build_fn=self._get_engine_for_fi,
                                  epochs=fi_parameters['epochs'],
                                  verbose=2,
                                  validation_split=validation_split)

        fi_engine.fit(x, y)

        self._calculate_fi_from_model(fi_engine, x, y)

        result = {'description': 'FI calculating OK'}

        return result

    def _get_engine_for_fi(self):
        inner_engine = clone_model(self._engine.inner_engine)
        self._engine.compile_engine(inner_engine)

        return inner_engine

    def _calculate_fi_from_model(self, fi_model: KerasRegressor, x: np.ndarray, y: np.ndarray) -> None:
        perm = PermutationImportance(fi_model, random_state=42).fit(x, y)

        fi = pd.DataFrame(perm.feature_importances_, columns=['error_delta'])
        fi['feature'] = self.fitting_parameters.x_columns
        fi = fi.sort_values(by='error_delta', ascending=False)

        fi['indicator'] = fi['feature'].apply(self._get_indicator_from_column_name)

        fi['analytics'] = fi['feature'].apply(self._get_analytics_from_column_name)

        fi['influence_factor'] = fi['error_delta'].apply(lambda error_delta: math.log(error_delta + 1)
                                    if error_delta > 0 else 0)

        if_sum = fi['influence_factor'].sum()
        fi['influence_factor'] = fi['influence_factor'] / if_sum

        fi_ind = fi.copy()

        fi_ind['indicator_short_id'] = fi_ind['indicator'].apply(lambda ind: ind['short_id'])

        fi_ind = fi_ind[['indicator_short_id',
                         'error_delta',
                         'influence_factor']].groupby(['indicator_short_id'], as_index=False).sum()

        fi_ind['indicator'] = fi_ind['indicator_short_id'].apply(self._get_indicator_from_short_id)

        fi_ind = fi_ind.sort_values(by='error_delta', ascending=False)

        fi = fi.to_dict('records')
        fi_ind = fi_ind.to_dict('records')

        self.fitting_parameters.feature_importances = {'extended': fi, 'grouped': fi_ind}

    def _get_indicator_from_column_name(self, column_name: str) -> dict[str, Any]:

        short_id = column_name.split('_')[1]

        return self._get_indicator_from_short_id(short_id)

    def _get_indicator_from_short_id(self, short_id: str) -> dict[str, Any]:

        indicators = [ind for ind in (self.parameters.x_indicators + self.parameters.y_indicators)
                      if ind['short_id'] == short_id]

        return indicators[0]

    def _get_analytics_from_column_name(self, column_name: str) -> list[dict[str, Any]]:

        column_list = column_name.split('_')

        if 'an' in column_list:
            result = [el['analytics'] for el in (self.fitting_parameters.x_analytic_keys + self.fitting_parameters.y_analytic_keys)
                      if el['short_id'] == column_list[3]][0]
        else:
            result = []

        return result

    def _interrupt_fi_calculation_job(self):
        if self.fitting_parameters.fi_calculation_is_started:
            set_background_job_interrupted(self.fitting_parameters.fi_calculation_job_id, self._db_path)


def get_additional_actions() -> dict[str, Callable]:
    return {'model_calculate_feature_importances': _calculate_feature_importances,
            'model_get_feature_importances': _get_feature_importances,
            'model_drop_fi_calculation': _drop_fi_calculation,
            'model_get_sensitivity_analysis': _get_sensitivity_analysis
            }


@execute_in_background
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

    model = VbmModel(parameters['model']['id'], parameters['db'])

    result = model.calculate_feature_importances(parameters['model']['fi_parameters'])

    return result


def _get_feature_importances(parameters: dict[str, Any]) -> dict[str, Any]:
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    model = VbmModel(parameters['model']['id'], parameters['db'])

    result = model.get_feature_importances()

    return result


def _drop_fi_calculation(parameters: dict[str, Any]) -> str:
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    model = VbmModel(parameters['model']['id'], parameters['db'])

    result = model.drop_fi_calculation()

    return result


def _get_sensitivity_analysis(parameters: dict[str, Any]) -> dict[str, Any]:
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if not parameters.get('db'):
        raise ParameterNotFoundException('Parameter "db" is not found in request parameters')

    check_fields = ['inputs_0', 'inputs_plus', 'inputs_minus']

    for field in check_fields:
        if field not in parameters:
            raise ParameterNotFoundException('Parameter "{}" is not found in request parameters'.format(field))

    model = VbmModel(parameters['model']['id'], parameters['db'])

    result = model.get_sensitivity_analysis(parameters['inputs_0'],
                                            parameters['inputs_plus'],
                                            parameters['inputs_minus'])

    return result