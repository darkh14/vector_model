""" VBM (Vector Budget model)
    Contains model class for VBM
    Classes:
        VbmModel - main model class. Compared with base class it provides feature importances calculation,
            sensitivity analysis calculation and factor analysis calculation
"""

from typing import Any, Optional, Callable, ClassVar
import numpy as np
import pandas as pd
import math

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential, clone_model
from eli5.sklearn import PermutationImportance
import plotly.graph_objects as go

from .base_model import Model
from ..model_types import DataTransformersTypes
from vm_logging.exceptions import ModelException, ParameterNotFoundException
from ..model_filters import get_fitting_filter_class
from ..engines import get_engine_class
from vm_background_jobs.decorators import execute_in_background
from vm_background_jobs.controller import set_background_job_interrupted
from id_generator import IdGenerator


__all__ = ['VbmModel', 'get_additional_actions']

class VbmModel(Model):
    """ Main model class. Compared with base class it provides feature importances calculation,
            sensitivity analysis calculation and factor analysis calculation

        Methods:
            calculate_feature_importances - for calculating feature importances and saving them to db
            get_feature_importances - for getting calculated feature importances
            drop_fi_calculation - to delete calculated feature importances from db
            get_sensitivity_analysis - for calculating and getting sensitivity analysis data
            get_factor_analysis - for calculating and getting factor analysis data
    """

    service_name: ClassVar[str] = 'vbm'

    def _form_output_columns_description(self) -> dict[str, Any]:
        """
        Forms and gets output columns description
        :return: formed description
        """
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
        """
        Converts output np array y to output pd. data
        :param y: y output predicted np array
        :param x_data: input x data
        :return: predicted output pd data
        """
        result = pd.DataFrame(y, columns=self.fitting_parameters.y_columns)

        result[['organisation', 'scenario', 'period']] = x_data[['organisation_struct', 'scenario_struct', 'period']]

        return result


    def _get_model_estimators(self, for_predicting: bool = False,
                              fitting_parameters: Optional[dict[str, Any]] = None) -> list[tuple[str, Any]]:
        """
        Gets list of estimators for data transforming before fitting or predicting.
        Excludes scaler compared to base class method
        :param for_predicting: True is need to form estimators for predicting
        :param fitting_parameters: parameters for fitting
        :return: list of estimators
        """
        estimators = super()._get_model_estimators(for_predicting, fitting_parameters)

        estimators = [es for es in estimators if es[0] != DataTransformersTypes.SCALER.value]
        return estimators

    def calculate_feature_importances(self, fi_parameters: dict[str, Any]) -> dict[str, Any]:
        """
        For calculating feature importances and saving them to db
        :param fi_parameters: parameters for fi calculation
        :return: result of fi calculation. calculated data is saved in model
        """
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
        """
        For getting calculated feature importances
        :return: calculated fi data
        """
        if not self.fitting_parameters.fi_is_calculated:
            raise ModelException('FI is not calculated. Calculate feature importances first')

        return self.fitting_parameters.feature_importances

    def drop_fi_calculation(self) -> str:
        """
        Deletes calculated feature importances from db
        :return: result of deleting
        """
        self._interrupt_fi_calculation_job()
        self.fitting_parameters.set_drop_fi_calculation()
        self._write_to_db()

        return 'Model "{}" id "{}" fi calculation is dropped'.format(self.parameters.name, self.id)

    def get_sensitivity_analysis(self, inputs_0: list[dict[str, Any]],
                                 inputs_plus: list[dict[str, Any]],
                                 inputs_minus: list[dict[str, Any]]) -> dict[str, Any]:
        """
        For calculating and getting sensitivity analysis data
        :param inputs_0: main input data
        :param inputs_plus: plus deviation input data
        :param inputs_minus: minus deviation input data
        :return: calculated sensitivity analysis data
        """
        pipeline = self._get_model_pipeline(for_predicting=True)
        data_0 = pipeline.transform(inputs_0)

        data_plus = pipeline.transform(inputs_plus)
        data_minus = pipeline.transform(inputs_minus)

        x_0 = self._data_to_x(data_0)

        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)
        self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number)
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
    def get_factor_analysis(self, inputs: list[dict[str, Any]],
                                 outputs: dict[str, Any],
                                 input_indicators: list[dict[str, Any]],
                                 output_indicator: dict[str, Any],
                                 get_graph: bool = False) -> dict[str, Any]:
        """
        For calculating and getting factor analysis data
        :param inputs: input data
        :param outputs: output data - scenarios
        :param input_indicators: input indicators list
        :param output_indicator: output indicator for fa calculation
        :param get_graph: true if we want to return also graph data
        :return: calculated fa data
        """
        self._check_before_fa_calculation()

        # method of chain substitutions
        result_data = []
        used_indicator_ids = []

        output_indicator_short_id_s = [el['short_id'] for el in self.parameters.y_indicators
                                       if el['id'] == output_indicator['id'] and el['type'] == output_indicator['type']]

        if not output_indicator_short_id_s:
            raise ModelException('Indicator "{}", id "{}" type "{}" not in model'.format(output_indicator['name'],
                                                                                         output_indicator['id'],
                                                                                         output_indicator['type']))

        output_indicator_short_id = output_indicator_short_id_s[0]

        output_columns = [col for col in self.fitting_parameters.y_columns
                          if col.split('_')[1] == output_indicator_short_id]

        input_data = pd.DataFrame(inputs)
        input_data['indicator_short_id'] = input_data['indicator'].apply(IdGenerator.get_short_id_from_dict_id_type)

        main_periods = input_data[['period',
                                   'is_main_period']].loc[input_data['is_main_period'] == True]['period'].unique()

        main_periods = list(main_periods)

        output_base = self._get_output_value_for_fa(input_data, outputs, used_indicator_ids,
                                                    main_periods, output_columns)

        for indicator_element in input_indicators:

            ind_short_id_s = [el['short_id'] for el in self.parameters.x_indicators
                                 if el['id'] == indicator_element['id'] and el['type'] == indicator_element['type']]

            if not ind_short_id_s:
                raise ModelException('Indicator "{}", id "{}" type "{}" not in model'.format(indicator_element['name'],
                                                                                             indicator_element['id'],
                                                                                             indicator_element['type']))
            ind_short_id = ind_short_id_s[0]

            output_value = self._get_output_value_for_fa(input_data, outputs, used_indicator_ids, main_periods,
                                                         output_columns, ind_short_id)

            result_element = {'indicator': indicator_element, 'value': output_value - output_base}

            result_data.append(result_element)
            used_indicator_ids.append(ind_short_id)

            output_base = output_value

        graph_string = ''

        if get_graph:
            graph_data = self._get_data_for_fa_graph(result_data, outputs)
            graph_string = self._get_fa_graph_bin(graph_data, output_indicator['name'])

        return {'fa': result_data, 'graph_data': graph_string}

    def _check_before_fi_calculating(self, fi_parameters:  dict[str, Any]) -> None:
        """
        For checking parameters before fi calculation
        :param fi_parameters: parameters to check
        """
        if not self._initialized:
            raise ModelException('Model is not initialized. Check model id before')

        if not self.fitting_parameters.is_fit:
            raise ModelException('Model is not fit. Fit model before fi calculation')

        if 'epochs' not in fi_parameters:
            raise ModelException('Parameter "epochs" not found in fi parameters')

        if self.fitting_parameters.fi_calculation_is_started:
            raise ModelException('Another fi calculation is started yet. Wait for end of fi calculation')

    def _fi_calculate_model(self, epochs, fi_parameters):
        """
        For fi calculation after prepare and check parameters. Method - permutation importances
        :param epochs: number of epochs to fit
        :param fi_parameters: parameters to calculate fi
        :return: info of calculating
        """
        pipeline = self._get_model_pipeline(for_predicting=False, fitting_parameters=fi_parameters)
        data = pipeline.fit_transform(None)

        x, y = self._data_to_x_y(data)
        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)
        self._engine = get_engine_class(self.parameters.type)('', input_number, output_number, True)

        validation_split = fi_parameters.get('validation_split') or 0.2

        fi_engine = KerasRegressor(build_fn=self._get_engine_for_fi,
                                  epochs=epochs,
                                  verbose=2,
                                  validation_split=validation_split)

        fi_engine.fit(x, y)

        self._calculate_fi_from_model(fi_engine, x, y)

        result = {'description': 'FI calculating OK'}

        return result

    def _get_engine_for_fi(self) -> Sequential:
        """
        Returns inner engine for calculating fi
        :return: keras sequential engine
        """
        inner_engine = clone_model(self._engine.inner_engine)
        self._engine.compile_engine(inner_engine)

        return inner_engine

    def _calculate_fi_from_model(self, fi_model: KerasRegressor, x: np.ndarray, y: np.ndarray) -> None:
        """
        Calculates fi using permutation importances
        :param fi_model: keras regressor model for permutation importances
        :param x: inputs to calculate fi
        :param y: outputs to calculate fi
        """
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
        """
        Gets dict of indicator from column name
        :param column_name: name of columns
        :return: indicator dict
        """
        short_id = column_name.split('_')[1]

        return self._get_indicator_from_short_id(short_id)

    def _get_indicator_from_short_id(self, short_id: str) -> dict[str, Any]:
        """
        Gets dict of indicator from its short id
        :param short_id: short id if indicator
        :return: indicator dict
        """
        indicators = [ind for ind in (self.parameters.x_indicators + self.parameters.y_indicators)
                      if ind['short_id'] == short_id]

        return indicators[0]

    def _get_analytics_from_column_name(self, column_name: str) -> list[dict[str, Any]]:
        """
        Gets analytic list from column name
        :param column_name: name of column
        :return: analytic list
        """
        column_list = column_name.split('_')

        if 'an' in column_list:
            result = [el['analytics'] for el in (self.fitting_parameters.x_analytic_keys + self.fitting_parameters.y_analytic_keys)
                      if el['short_id'] == column_list[3]][0]
        else:
            result = []

        return result

    def _interrupt_fi_calculation_job(self) -> None:
        """
        Interrupt process of fi calculation if it launched in subprocess
        """
        if self.fitting_parameters.fi_calculation_is_started:
            set_background_job_interrupted(self.fitting_parameters.fi_calculation_job_id)

    def _check_before_fa_calculation(self) -> None:
        """
        Checks parameters before fa calculation. Raises ModelException if checking is failed
        """
        if not self._initialized:
            raise ModelException('Error of calculating factor analysis data. Model is not initialized')

        if not self.fitting_parameters.is_fit:
            raise ModelException('Error of calculating factor analysis data. Model is not fit. '
                                     'Train the model before calculating')

    def _get_output_value_for_fa(self, input_data: pd.DataFrame,
                                 outputs: dict[str, Any],
                                 used_indicator_ids: list[str],
                                 main_periods: list[str],
                                 output_columns: list[str],
                                 current_ind_short_id: str = ''):
        """
        Forms output data according to one indicator while fa calculating
        :param input_data: main input data
        :param outputs: output data - scenarios
        :param used_indicator_ids: ids of indicators, which is previously used
        :param main_periods: periods of fa calculating
        :param output_columns: list of columns of output indicator
        :param current_ind_short_id: id of current input indicator
        :return: one indicator fa data
        """
        c_input_data = input_data.copy()

        c_input_data['scenario'] = c_input_data['organisation'].apply(lambda sc: outputs['calculated'])
        c_input_data['periodicity'] = c_input_data['scenario'].apply(lambda sc: sc['periodicity'])

        c_input_data['current_indicator_short_id'] = current_ind_short_id
        c_input_data['used_indicator_ids'] = None
        c_input_data['used_indicator_ids'] = c_input_data['used_indicator_ids'].apply(lambda ind: used_indicator_ids)

        c_input_data['value'] = c_input_data[['value_base',
                                              'value_calculated',
                                              'used_indicator_ids',
                                              'indicator_short_id',
                                              'current_indicator_short_id']].apply(self._get_value_for_fa, axis=1)

        c_input_data = c_input_data.drop(['value_base', 'value_calculated', 'used_indicator_ids',
                                          'current_indicator_short_id'], axis=1)

        pipeline = self._get_model_pipeline(for_predicting=True)
        data = pipeline.transform(c_input_data)

        x = self._data_to_x(data)

        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)
        if not self._engine:
            self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number)
        y = self._engine.predict(x)

        data[self.fitting_parameters.y_columns] = y

        output_data = data.loc[data['period'].isin(main_periods)].copy()

        output_data = output_data[output_columns]
        output_data['value'] = output_data.apply(sum, axis=1)

        output_value = output_data['value'].sum()

        return output_value


    @staticmethod
    def _get_value_for_fa(input_parameters):
        """
        Calculates fa value of one row. Used in pd.DataFrame.apply()
        :param input_parameters: values for calculating fa value
        :return: fa value
        """
        (value_based, value_calculated, used_indicator_ids,
         indicator_short_id, current_indicator_short_id) = input_parameters

        if current_indicator_short_id == indicator_short_id:
            value = value_calculated
        elif indicator_short_id in used_indicator_ids:
            value = value_calculated
        else:
            value = value_based

        return value

    def _get_data_for_fa_graph(self, result_data: list[dict[str, Any]], outputs: dict[str, Any]) -> pd.DataFrame:
        """
        Forms dataframe using to create fa graph html
        :param result_data: fa data to form fa graph data
        :param outputs: based and calculated values
        :return: prepared dataframe
        """
        result_data = pd.DataFrame(result_data)
        result_data['title'] = result_data['indicator'].apply(lambda x: x['name'])
        result_data['order'] = list(range(2, result_data.shape[0]+2))

        result_data.drop(['indicator'], axis=1, inplace=True)

        base_line = {'title': 'Базовый', 'value': outputs['based']['value'], 'order': 1}
        calculated_line = {'title': 'Расчетный', 'value': outputs['calculated']['value'], 'order': result_data.shape[0]+2}

        result_data = pd.concat([result_data, pd.DataFrame([base_line, calculated_line])])

        result_data = result_data.sort_values('order')

        return result_data


    def _get_fa_graph_bin(self, values: pd.DataFrame, out_indicator_name: str) -> str:
        """
        Forms fa graph html str
        :param values: values to form html
        :param out_indicator_name: name of output indicator, used in titles
        :return: formed html str
        """
        x_list = list(values['title'])
        y_list = list(values['value'])

        text_list = []
        for index, item in enumerate(y_list):
            if item > 0 and index != 0 and index != len(y_list) - 1:
                text_list.append('+{0:.2f}'.format(y_list[index]))
            else:
                text_list.append('{0:.2f}'.format(y_list[index]))

        for index, item in enumerate(text_list):
            if item[0] == '+' and index != 0 and index != len(text_list) - 1:
                text_list[index] = '<span style="color:#2ca02c">' + text_list[index] + '</span>'
            elif item[0] == '-' and index != 0 and index != len(text_list) - 1:
                text_list[index] = '<span style="color:#d62728">' + text_list[index] + '</span>'
            if index == 0 or index == len(text_list) - 1:
                text_list[index] = '<b>' + text_list[index] + '</b>'

        dict_list = []
        for i in range(0, 1200, 200):
            dict_list.append(dict(
                type="line",
                line=dict(
                    color="#666666",
                    dash="dot"
                ),
                x0=-0.5,
                y0=i,
                x1=6,
                y1=i,
                line_width=1,
                layer="below"))

        fig = go.Figure(go.Waterfall(
            name="Factor analysis", orientation="v",
            measure=["absolute", *(values.shape[0]-2) * ["relative"], "total"],
            x=x_list,
            y=y_list,
            text=text_list,
            textposition="outside",
            connector={"line": {"color": 'rgba(0,0,0,0)'}},
            increasing={"marker": {"color": "#2ca02c"}},
            decreasing={"marker": {"color": "#d62728"}},
            totals={'marker': {"color": "#9467bd"}},
            textfont={"family": "Open Sans, light",
                      "color": "black"
                      }
        ))

        fig.update_layout(
            title=
            {'text': '<b>Факторный анализ</b><br><span style="color:#666666">{}</span>'.format(out_indicator_name)},
            showlegend=False,
            height=650,
            font={
                'family': 'Open Sans, light',
                'color': 'black',
                'size': 14
            },
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="руб.",
            shapes=dict_list
        )

        fig.update_xaxes(tickangle=-45, tickfont=dict(family='Open Sans, light', color='black', size=14))
        fig.update_yaxes(tickangle=0, tickfont=dict(family='Open Sans, light', color='black', size=14))

        graph_str = fig.to_html()

        return graph_str


def get_additional_actions() -> dict[str, Callable]:
    """
    Formed additional actions of vbm model module
    :return: actions dict (functions)
    """
    return {'model_calculate_feature_importances': _calculate_feature_importances,
            'model_get_feature_importances': _get_feature_importances,
            'model_drop_fi_calculation': _drop_fi_calculation,
            'model_get_sensitivity_analysis': _get_sensitivity_analysis,
            'model_get_factor_analysis_data': _get_factor_analysis_data
            }


@execute_in_background
def _calculate_feature_importances(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    For calculating feature importances
    :param parameters: request parameters
    :return: result (info) of calculating fi
    """
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    if 'fi_parameters' not in parameters['model']:
        raise ParameterNotFoundException('Parameter "fi_parameters" not found in model parameters')

    if 'fi_parameters' in parameters['model']:
        if 'filter' in parameters['model']['fi_parameters']:
            input_filter = parameters['model']['fi_parameters']['filter']
            filter_obj = get_fitting_filter_class()(input_filter)
            parameters['model']['fi_parameters']['filter'] = filter_obj.get_value_as_model_parameter()

        if 'job_id' in parameters:
            parameters['model']['fi_parameters']['job_id'] = parameters['job_id']

    model = VbmModel(parameters['model']['id'])

    result = model.calculate_feature_importances(parameters['model']['fi_parameters'])

    return result


def _get_feature_importances(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    For getting calculated feature importances
    :param parameters: request parameters
    :return: fi data
    """
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    model = VbmModel(parameters['model']['id'])

    result = model.get_feature_importances()

    return result


def _drop_fi_calculation(parameters: dict[str, Any]) -> str:
    """
    To drop calculated feature importances data
    :param parameters: request parameters
    :return: result of dropping
    """
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    model = VbmModel(parameters['model']['id'])

    result = model.drop_fi_calculation()

    return result


def _get_sensitivity_analysis(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    For calculating and getting sensitivity analysis data
    :param parameters: request parameters
    :return: calculated sa data
    """
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')


    check_fields = ['inputs_0', 'inputs_plus', 'inputs_minus']

    for field in check_fields:
        if field not in parameters:
            raise ParameterNotFoundException('Parameter "{}" is not found in request parameters'.format(field))

    model = VbmModel(parameters['model']['id'])

    result = model.get_sensitivity_analysis(parameters['inputs_0'],
                                            parameters['inputs_plus'],
                                            parameters['inputs_minus'])

    return result


def _get_factor_analysis_data(parameters: dict[str, Any]) -> dict[str, Any]:
    """
    For calculating and getting factor analysis data
    :param parameters: request parameters
    :return: calculated fa data
    """
    if not parameters.get('model'):
        raise ParameterNotFoundException('Parameter "model" is not found in request parameters')

    check_fields = ['inputs', 'input_indicators', 'outputs', 'output_indicator']

    for field in check_fields:
        if field not in parameters:
            raise ParameterNotFoundException('Parameter "{}" is not found in request parameters'.format(field))

    model = VbmModel(parameters['model']['id'])

    result = model.get_factor_analysis(parameters['inputs'],
                                        parameters['outputs'],
                                        parameters['input_indicators'],
                                        parameters['output_indicator'],
                                        parameters.get('get_graph')
                                        )

    return result