""" VBM (Vector Budget model)
    Contains model class for VBM
    Classes:
        VbmModel - main model class. Compared with base class it provides feature importances calculation,
            sensitivity analysis calculation and factor analysis calculation
"""

from typing import Any, Callable, ClassVar, Optional
import numpy as np
import pandas as pd
import math

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from eli5.sklearn import PermutationImportance
import plotly.graph_objects as go

from .base_model import Model
from vm_logging.exceptions import ModelException
from ..engines import get_engine_class
from vm_background_jobs.decorators import execute_in_background
from vm_background_jobs.controller import set_background_job_interrupted
from data_processing.loading_engines import get_engine_class as get_loading_engine_class
from ..data_transformers import get_transformer_class
from ..model_types import DataTransformersTypes, FittingStatuses, ModelTypes
from data_processing import api_types as data_api_types
from .. import api_types
import api_types as general_api_types

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

    def predict(self, x: list[dict[str, Any]]) -> dict[str, Any]:
        """
        For predicting data with model
        :param x: input data for predicting
        :return: predicted output data
        """
        self._check_before_predicting(x)

        result_data = self._predict_model(x)
        result_data = result_data.drop(self.fitting_parameters.x_columns, axis=1)

        description = self._form_output_columns_description(result_data)

        return {'outputs': result_data.to_dict('records'), 'description': description}

    def _form_output_columns_description(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Forms and gets output columns description
        :return: formed description
        """
        result_description = dict()

        result_description['columns_description'] = self._db_connector.get_lines('column_descriptions',
                                                {'name': {'$in': self.fitting_parameters.y_columns}})

        result_description['organisations'] = self._db_connector.get_lines('organisations',
                                                {'id': {'$in': list(data['organisation'].unique())}})

        result_description['scenarios'] = self._db_connector.get_lines('scenarios',
                                                {'id': {'$in': list(data['scenario'].unique())}})

        indicators = list(set([el['indicator'] for el in result_description['columns_description']]))
        result_description['indicators'] = self._db_connector.get_lines('indicators',
                                                {'id': {'$in': indicators}})

        analytic_keys = list(set([el['analytic_key'] for el in result_description['columns_description']]))

        result_description['analytic_keys'] = self._db_connector.get_lines('analytic_keys',
                                                {'key': {'$in': analytic_keys}})

        analytics = list(set([el['value_id'] for el in result_description['analytic_keys']]))

        result_description['analytics'] = self._db_connector.get_lines('analytics',
                                                                       {'id': {'$in': analytics}})

        return result_description

    def calculate_feature_importances(self, fi_parameters: dict[str, Any], job_id: str = '') -> str:
        """
        For calculating feature importances and saving them to db
        :param fi_parameters: parameters for fi calculation
        :param job_id: id of background job if fi calculation is in background
        :return: result of fi calculation. calculated data is saved in model
        """
        self._check_before_fi_calculating(fi_parameters)

        self.fitting_parameters.set_start_fi_calculation(job_id)
        self._write_to_db()

        try:
            result = self._fi_calculate_model(fi_parameters)
        except Exception as ex:
            self.fitting_parameters.set_error_fi_calculation(str(ex))
            self._write_to_db()
            raise ex

        if self.fitting_parameters.fi_status != FittingStatuses.Error:

            self.fitting_parameters.set_end_fi_calculation()
            self._write_to_db()

        return result

    def drop(self) -> str:
        """ Deletes model from db. sets initializing = False
        :return: result of dropping
        """
        result = super().drop()

        self._write_feature_importances([], [])

        return result

    def drop_fi_calculation(self) -> str:
        """
        Deletes calculated feature importances from db
        :return: result of deleting
        """

        if self.fitting_parameters.fi_status not in (FittingStatuses.Fit,
                                                     FittingStatuses.Started,
                                                     FittingStatuses.PreStarted,
                                                     FittingStatuses.Error):
            raise ModelException('Can not drop fi calculation. FI is not calculated')

        self._interrupt_fi_calculation_job()
        self.fitting_parameters.set_drop_fi_calculation()
        self._write_feature_importances([], [])
        self._write_to_db()

        return 'Model "{}" id "{}" fi calculation is dropped'.format(self.parameters.name, self.id)

    def get_feature_importances(self, extended: bool = False) -> dict[str, Any]:
        """
        Deletes calculated feature importances from db
        :return: result of deleting
        """

        if self.fitting_parameters.fi_status != FittingStatuses.Fit:
            raise ModelException('Can not get feature importances. FI is not calculated')

        result = self._read_feature_importances(extended=extended)

        return result

    def get_sensitivity_analysis(self, inputs: list[dict[str, Any]],
                                 input_indicators: list[str],
                                 output_indicator: str,
                                 deviations: list[float],
                                 get_graph: bool = False,
                                 auto_selection_number: int = 0,
                                 value_name: str = 'sum',
                                 expand_by_periods: bool = False) -> dict[str, Any]:
        """
        For calculating and getting sensitivity analysis data
        :param inputs: main input data
        :param input_indicators: list input indicators to use
        :param output_indicator: output indicator parameters
        :param deviations: list of deviations of data
        :param get_graph: returns sensitivity analysis graph if True
        :param auto_selection_number: if > 0 sets, hom many indicators will be returned in sa
        :param value_name: name of value, sum or qty
        :param expand_by_periods: return result expanded by periods if true else grouped result
        :return: calculated sensitivity analysis data
        """

        loading_engine = get_loading_engine_class()()

        loading_engine.check_data(inputs, checking_parameter_name='inputs')

        pd_all_input_data = self._prepare_input_data_for_sa(inputs, input_indicators, deviations)

        pd_all_input_data['variant'] = pd_all_input_data['scenario'].apply(self._get_sa_variant_from_scenario)

        data_all = self._predict_model(pd_all_input_data)
        data_all['variant'] = data_all['scenario'].apply(self._get_sa_variant_from_scenario)
        data_all['scenario'] = data_all['scenario'].apply(self._clear_suffix_from_scenario_id)

        y_columns = self._get_sa_output_columns(self.fitting_parameters.y_columns, output_indicator, value_name)

        data_all['y'] = data_all[y_columns].apply(sum, axis=1)
        data_all = data_all.drop(self.fitting_parameters.x_columns + self.fitting_parameters.y_columns, axis=1)
        data_all['indicator'] = data_all['variant'].apply(lambda el: '' if el == 'base' else el.split('_')[1])
        data_all['deviation'] = data_all['variant'].apply(lambda el: 0 if el == 'base'
                                else float(el.split('_')[3])*(-1 if el.split('_')[2] == 'minus' else 1))

        data_all = data_all.drop('variant', axis=1)

        data_base = data_all.loc[data_all['deviation'] == 0].copy()

        data_all = data_all.merge(data_base[['organisation', 'scenario', 'period', 'y']].rename({'y': 'y_0'}, axis=1),
                                  on=('organisation', 'scenario', 'period'), how='left')

        data_all['delta'] = data_all['y'] - data_all['y_0']
        data_all['relative_delta'] = data_all[['delta', 'y_0']].apply(lambda ss: ss['delta'] / ss['y_0']
                             if ss['y_0'] else 0, axis=1)

        data_all_grouped = data_all[['organisation', 'scenario', 'indicator', 'deviation', 'y', 'y_0', 'delta',
                                     'relative_delta']].loc[data_all['indicator'] != '']
        data_all_grouped = data_all_grouped.groupby(['organisation', 'scenario',
                                                     'indicator', 'deviation'], axis=0, as_index=False).sum()

        data_all_grouped['relative_delta'] = data_all_grouped[['delta', 'y_0']].apply(lambda ss: ss['delta'] / ss['y_0']
                             if ss['y_0'] else 0, axis=1)

        if auto_selection_number:

            data_all_grouped_ind = data_all_grouped[['indicator', 'y', 'y_0', 'delta',
                                         'relative_delta']].loc[data_all['indicator'] != '']
            data_all_grouped_ind = data_all_grouped_ind.groupby(['indicator'], axis=0, as_index=False).sum()

            data_all_grouped_ind['abs_relative_delta'] = data_all_grouped_ind['relative_delta'].apply(abs)
            data_all_grouped_ind.sort_values('abs_relative_delta', ascending=False, inplace=True)

            data_all_grouped_ind['relative_delta'] = data_all_grouped_ind[['delta', 'y_0']].apply(
                    lambda ss: ss['delta'] / ss['y_0']
                    if ss['y_0'] else 0, axis=1)

            data_all_grouped_ind = data_all_grouped_ind.drop('abs_relative_delta', axis=1)

            data_all_grouped_ind = data_all_grouped_ind.iloc[0:auto_selection_number]

            filtered_indicators = list(data_all_grouped_ind['indicator'].unique())

            data_all_grouped = data_all_grouped.loc[data_all_grouped['indicator'].isin(filtered_indicators)]
            data_all = data_all.loc[data_all['indicator'].isin(filtered_indicators)]

        if get_graph:
            graph_string = self._get_sa_graph_html(data_all_grouped)
        else:
            graph_string = ''

        description = self._form_sa_output_description(data_all)

        result = dict()
        result['outputs'] = (data_all.to_dict(orient='records') if expand_by_periods
                             else data_all_grouped.to_dict(orient='records'))

        result['description'] = description
        result['graph_data'] = graph_string

        return result

    def _form_sa_output_description(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Forms and gets output columns description
        :return: formed description
        """
        result_description = dict()

        result_description['organisations'] = self._db_connector.get_lines('organisations',
                                                {'id': {'$in': list(data['organisation'].unique())}})

        result_description['scenarios'] = self._db_connector.get_lines('scenarios',
                                                {'id': {'$in': list(data['scenario'].unique())}})

        result_description['indicators'] = self._db_connector.get_lines('indicators',
                                                {'id': {'$in': list(data['indicator'].unique())}})

        return result_description

    def _get_sa_graph_html(self, graph_data: pd.DataFrame) -> str:
        """
        Forms sensitivity analysis graph html
        @param graph_data: sensitivity analysis data
        @return: string of html graph
        """
        x = [el*100 for el in graph_data['deviation'].unique()]
        x.append(0)

        x.sort()

        y_list = []
        indicator_names = []

        indicators = list(graph_data['indicator'].unique())
        indicators_descr = self._db_connector.get_lines('indicators', {'id': {'$in': indicators}})

        for ind_id in indicators:

            indicator_names.append([el['name'] for el in indicators_descr if el['id'] == ind_id][0])

            element_data = graph_data.loc[graph_data['indicator'] == ind_id].copy()

            element_data_0 = element_data.iloc[[0]].copy()
            element_data_0['deviation'] = 0
            element_data_0['y'] = element_data_0['y_0']
            element_data_0['delta'] = 0
            element_data_0['relative_delta'] = 0

            element_data = pd.concat((element_data, element_data_0), axis=0)

            element_data.sort_values('deviation', inplace=True)

            y_c = list(element_data['relative_delta'].to_numpy())
            y_c = list(map(lambda z: 100*z, y_c))

            y_list.append(y_c)

        fig = go.Figure()

        for ind, y in enumerate(y_list):
            fig.add_trace(go.Scatter(x=x, y=y, name=indicator_names[ind]))

        font_size = 10

        fig.update_layout(title=dict(text='Анализ на чувствительность', font=dict(size=font_size+1)), showlegend=True,
                          xaxis_title=dict(text="Отклонения входного показателя, %", font=dict(size=font_size)),
                          yaxis_title=dict(text="Отклонения выходного показателя, %", font=dict(size=font_size)),
                          paper_bgcolor='White',
                          plot_bgcolor='White')

        fig.update_layout(legend=dict(
                                    x=0,
                                    y=-0.2,
                                    traceorder="normal",
                                    orientation='h',
                                    font=dict(size=font_size),
                                ))

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', zerolinecolor='Grey', tickvals=x)

        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey', zerolinecolor='Grey')

        graph_html = fig.to_html()

        return graph_html

    def get_action_before_background_job(self, func_name: str,
                                         args: tuple[Any],
                                         kwargs: dict[str, Any]) -> Optional[Callable]:
        """
        Returns function which will be executed before model fi calculating
        @param func_name: name of fi calculating function
        @param args: positional arguments of fi calculating function
        @param kwargs: keyword arguments of fi calculating function.
        @return: function to execute before fi calculating
        """

        if func_name == 'calculate_fi':
            result = self.do_before_calculating_fi
        else:
            result = super().get_action_before_background_job(func_name, args, kwargs)

        return result

    # noinspection PyUnusedLocal
    def do_before_calculating_fi(self, args: list[Any], kwargs: dict[str, Any]) -> None:
        """
        Function will be executed before fi calculation
        Sets pre-start background job status
        @param args: positional arguments of fi calculating function
        @param kwargs: keyword arguments of fi calculating function.
        @return: None
        """
        self.fitting_parameters.set_pre_start_fi_calculation(kwargs.get('job_id', ''))
        self._write_to_db()

    def _prepare_input_data_for_sa(self, inputs_base: list[dict[str, Any]],
                                 ind_ids: list[str],
                                 deviations: list[int | float]) -> pd.DataFrame:
        """
        Transforms input data to calculate sensitivity analysis
        @param inputs_base: base input data
        @param ind_ids: input indicator ids
        @param deviations: list of derivations
        @return: data prepared to sa calculating
        """

        data_list_to_concat = []

        pd_input_base = pd.DataFrame(inputs_base)

        scenarios_description = self._db_connector.get_lines('scenarios')

        pd_input_base['periodicity'] = pd_input_base['scenario'].apply(lambda arg: [el['periodicity']
                                                                          for el in scenarios_description
                                                                          if el['id'] == arg][0])
        pd_input_base['scenario'] = pd_input_base['scenario'].apply(lambda el: '_'.join((el, 'base')))

        data_list_to_concat.append(pd_input_base)

        for ind_id in ind_ids:

            for dev in deviations:

                pd_inputs_plus = self._set_coefficient_to_data_by_ind(inputs_base, ind_id, 1+dev)
                suffix = 'ind_{}_plus_{}'.format(ind_id, str(dev).replace(' ', ''))
                pd_inputs_plus['scenario'] = pd_inputs_plus['scenario'].apply(lambda el: '_'.join((el, suffix)))
                data_list_to_concat.append(pd_inputs_plus)

                pd_inputs_minus = self._set_coefficient_to_data_by_ind(inputs_base, ind_id, 1-dev)
                suffix = 'ind_{}_minus_{}'.format(ind_id, str(dev).replace(' ', ''))
                pd_inputs_minus['scenario'] = pd_inputs_minus['scenario'].apply(lambda el: '_'.join((el, suffix)))

                data_list_to_concat.append(pd_inputs_minus)

        data_result = pd.concat(data_list_to_concat, axis=0, ignore_index=True)

        return data_result

    @staticmethod
    def _get_sa_variant_from_scenario(series: pd.Series) -> str:
        """
        Returns variant from scenario
        @param series: input data series
        @return: sa scenario variant
        """
        result = '_'.join(series.split('_')[1:])

        return result

    @staticmethod
    def _clear_suffix_from_scenario_id(series: pd.Series) -> dict[str]:
        """
        Clear suffix from scenario
        @param series: data series
        @return: scenario without suffix
        """
        result = series.split('_')[0]

        return result

    def get_factor_analysis(self, inputs: list[dict[str, Any]],
                            output_data_structure: dict[str, Any],
                            input_indicators: list[str],
                            output_indicator: str,
                            get_graph: bool = False) -> dict[str, Any]:
        """
        For calculating and getting factor analysis data
        :param inputs: input data
        :param output_data_structure: output data sum and qty
        :param input_indicators: input indicators list
        :param output_indicator: output indicator for fa calculation
        :param get_graph: true if we want to return also graph data
        :return: calculated fa data
        """
        self._check_before_fa_calculation(inputs, output_data_structure, input_indicators, output_indicator)

        # method of chain substitutions
        used_indicator_ids = []

        model_x_indicators = [el['id'] for el in self.parameters.x_indicators]
        model_y_indicators = [el['id'] for el in self.parameters.y_indicators]

        for input_ind in input_indicators:
            if input_ind not in model_x_indicators:
                raise ModelException('Input indicator id "{}" is not in model'.format(input_ind))

        if output_indicator not in model_y_indicators:
            raise ModelException('Output indicator id "{}" is not in model'.format(output_indicator))

        all_indicators = input_indicators + [output_indicator]

        columns_descr = self._db_connector.get_lines('column_descriptions',
                                                     {'indicator': {'$in': all_indicators}})

        output_columns_all = [el['name'] for el in columns_descr if (el['indicator'] == output_indicator
                          and el['value_name'] == output_data_structure['output_value_name'])]

        output_columns = [el for el in self.fitting_parameters.y_columns if el in output_columns_all]

        input_data = pd.DataFrame(inputs)

        main_periods = list(input_data[['period',
                                   'is_main_period']].loc[input_data['is_main_period'] == True]['period'].unique())

        output_base = self._get_output_value_for_fa(input_data, output_data_structure, used_indicator_ids,
                                                    main_periods, output_columns)

        result_data = []

        for indicator_element in input_indicators:

            output_value = self._get_output_value_for_fa(input_data, output_data_structure, used_indicator_ids,
                                                         main_periods,
                                                         output_columns, indicator_element)

            result_element = {'indicator': indicator_element, 'value': output_value - output_base}

            result_data.append(result_element)
            used_indicator_ids.append(indicator_element)

            output_base = output_value

        graph_string = ''

        if get_graph:
            based_scenario = input_data.loc[input_data['scenario_type'] ==
                                            data_api_types.ScenarioTypes.BASE]['scenario'].unique()[0]
            calculated_scenario = input_data.loc[input_data['scenario_type'] ==
                                            data_api_types.ScenarioTypes.CALCULATED]['scenario'].unique()[0]
            graph_data = self._get_data_for_fa_graph(result_data,
                                                     output_data_structure,
                                                     based_scenario,
                                                     calculated_scenario,
                                                     output_data_structure['output_value_name'])
            graph_string = self._get_fa_graph_bin(graph_data, output_indicator)

        indicators_descr = self._db_connector.get_lines('indicators',
                                                        {'id': {'$in': input_indicators}})

        return {'data': result_data, 'description': {'indicators': indicators_descr}, 'graph_data': graph_string}

    # noinspection PyMethodMayBeStatic
    def _set_coefficient_to_data_by_ind(self, input_data: list[dict[str, Any]], indicator: str,
                                        coefficient: float) -> pd.DataFrame:
        """
        Multiplies data by the coefficient
        @param input_data: input data array
        @param indicator: indicator to multiply its data
        @param coefficient: coefficient to multiply
        @return: result data
        """
        c_data = pd.DataFrame(input_data)

        strs_to_change = c_data.loc[c_data['indicator'] == indicator].copy()
        strs_not_to_change = c_data.loc[c_data['indicator'] != indicator].copy()

        for value_name in ('sum', 'qty'):
            strs_to_change[value_name] = strs_to_change[value_name]*coefficient

        result_data = pd.concat([strs_to_change, strs_not_to_change], axis=0, ignore_index=True)

        return result_data

    # noinspection PyMethodMayBeStatic
    def _get_sa_output_columns(self, y_columns: list[str], output_indicator: str, value_name: str) -> list[str]:
        """
        Returns list of output columns of sensitivity analysis calculation
        @param y_columns: result columns list
        @param output_indicator: output indicator id
        @param value_name: sum or qty
        @return: list of output data
        """
        result = []

        columns_descr = self._db_connector.get_lines('column_descriptions',
                                                    {'name': {'$in': y_columns}})
        for column in y_columns:

            descr = [el for el in columns_descr if el['name'] == column][0]

            if descr['indicator'] == output_indicator and descr['value_name'] == value_name:
                result.append(column)

        return result

    def _check_before_fi_calculating(self, fi_parameters:  dict[str, Any]) -> None:
        """
        For checking parameters before fi calculation
        :param fi_parameters: parameters to check
        """
        if not self._initialized:
            raise ModelException('Model is not initialized. Check model id before')

        if self.fitting_parameters.fitting_status != FittingStatuses.Fit:
            raise ModelException('Model is not fit. Fit model before fi calculation')

        if fi_parameters.get('job_id'):
            if self.fitting_parameters.fi_status != FittingStatuses.PreStarted:
                raise ModelException('Model is not prepared for feature importances calculation in background. ' +
                                     'Drop feature importances calculation and execute another fi calculation job')
        else:
            if self.fitting_parameters.fi_status == FittingStatuses.Started:
                raise ModelException('Model is not prepared for feature importances calculation. ' +
                                     'Drop feature importances calculation and execute another fi calculation')

    # noinspection PyUnusedLocal
    def _fi_calculate_model(self, fi_parameters):
        """
        For fi calculation after prepare and check parameters. Method - permutation importances
        :param fi_parameters: parameters to calculate fi
        :return: info of calculating
        """

        result = self._fit_model(fi_parameters, for_fi=True)

        self._calculate_fi_from_model(result['engine'], result['x'], result['y'])

        result = 'FI is calculated'

        return result

    def _write_feature_importances(self, fi_extended: list[dict[str, Any]], fi_grouped: list[dict[str, Any]]) -> None:
        """
        Writes feature importances to db
        @param fi_extended: feature importances extended data
        @param fi_grouped: feature importances grouped data
        @return: None
        """
        if fi_extended:
            for fi_line in fi_extended:
                fi_line['model_id'] = self.id

            self._db_connector.set_lines('feature_impotances_extended',
                                         fi_extended, {'model_id': self.id})
        else:
            self._db_connector.delete_lines('feature_impotances_extended', {'model_id': self.id})

        if fi_grouped:
            for fi_line in fi_grouped:
                fi_line['model_id'] = self.id

            self._db_connector.set_lines('feature_impotances_grouped',
                                         fi_grouped, {'model_id': self.id})
        else:
            self._db_connector.delete_lines('feature_impotances_grouped', {'model_id': self.id})

    def _read_feature_importances(self, extended: bool = False) -> dict[str, Any]:
        """
        Reads feature importances from db
        @param extended: True if it needs to read extended data fi else grouped
        @return: feature importances data
        """
        collection_name = 'feature_impotances_extended' if extended else 'feature_impotances_grouped'
        fi = self._db_connector.get_lines(collection_name, {'model_id': self.id})
        description = self._get_fi_description(fi, extended)
        result = {'outputs': fi, 'description': description}

        return result

    def _get_fi_description(self, fi: list[dict[str, Any]], extended: bool = False) -> dict[str, Any]:
        """
        Returns description of feature importances data
        @param fi: feature importances data
        @param extended: True if it is extended fi data else group data
        @return: description of data
        """
        pd_fi = pd.DataFrame(fi)

        result_description = dict()

        fi_columns = list(pd_fi['feature'].unique())

        if extended:
            result_description['columns_description'] = self._db_connector.get_lines('column_descriptions',
                                                                                     {'name': {
                                                                                         '$in': fi_columns}})

            analytic_keys = list(set([el['analytic_key'] for el in result_description['columns_description']]))

            result_description['analytic_keys'] = self._db_connector.get_lines('analytic_keys',
                                                                               {'key': {'$in': analytic_keys}})

            analytics = list(set([el['value_id'] for el in result_description['analytic_keys']]))

            result_description['analytics'] = self._db_connector.get_lines('analytics',
                                                                           {'id': {'$in': analytics}})
        else:

            result_description['indicators'] = self._db_connector.get_lines('indicators',
                                                                            {'id': {'$in': fi_columns}})

        return result_description

    def _fit_model(self, fitting_parameters: Optional[dict[str, Any]] = None, for_fi: bool = False) -> dict[str, Any]:
        """
        For fitting model after checking, and preparing parameters
        :param fitting_parameters: additional fitting parameters
        :return: fitting history
        """

        # noinspection PyTypeChecker
        self._scaler = get_transformer_class(DataTransformersTypes.SCALER, self.parameters.type)(self.parameters,
                                                    self.fitting_parameters, model_id=self._id,
                                                    new_scaler=self.fitting_parameters.is_first_fitting())

        pipeline = self._get_model_pipeline(for_predicting=False, fitting_parameters=fitting_parameters)
        data = pipeline.fit_transform(None)

        x, y = self._data_to_x_y(data)
        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)

        self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number,
                                                              self.fitting_parameters.is_first_fitting() or for_fi)

        self._scaler = pipeline.named_steps['scaler']

        if for_fi:
            validation_split = fitting_parameters.get('validation_split') or 0.2 if fitting_parameters else 0.2
            fi_engine = self._get_engine_for_fi(fitting_parameters, validation_split)

            fitting_result = fi_engine.fit(x, y)
            result_history = fitting_result.history if hasattr(fitting_result, 'history') else []
            result_engine = fi_engine
        else:

            fitting_result = self._engine.fit(x, y, fitting_parameters)
            result_history = fitting_result['history']

            y_pred = self._engine.predict(x)

            data_predicted = data.copy()
            data_predicted[self.fitting_parameters.y_columns] = y_pred

            data = self._scaler.inverse_transform(data)
            data_predicted = self._scaler.inverse_transform(data_predicted)

            y = data[self.fitting_parameters.y_columns].to_numpy()
            y_pred = data_predicted[self.fitting_parameters.y_columns].to_numpy()

            self.fitting_parameters.metrics = self._get_metrics(y, y_pred)

            result_engine = self._engine

        return {'history': result_history, 'x': x, 'y': y, 'engine': result_engine}

    # noinspection PyUnresolvedReferences
    def _get_engine_fn_for_fi(self) -> Sequential:
        """
        Returns inner engine for calculating fi
        :return: keras sequential engine
        """

        engine_for_fi = self._engine.get_engine_for_fi()

        return engine_for_fi

    def _get_engine_for_fi(self, fitting_parameters: dict[str, Any],
                           validation_split: float) -> [KerasRegressor | LinearRegression]:
        """
        Returns special variant of model engine to fi calculating
        @param fitting_parameters: dict of fitting parameters
        @param validation_split: validation split coefficient
        @return:
        """

        if self._engine.model_type == ModelTypes.NeuralNetwork:
            engine = KerasRegressor(model=self._get_engine_fn_for_fi(),
                                    build_fn=self._get_engine_fn_for_fi,
                           epochs=fitting_parameters['epochs'],
                           verbose=2,
                           validation_split=validation_split)
        else:
            # noinspection PyUnresolvedReferences
            engine = self._get_engine_fn_for_fi()

        return engine

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

        fi = fi.loc[~fi['feature'].isin(self.fitting_parameters.categorical_columns)]

        columns_descr = self._db_connector.get_lines('column_descriptions',
                                                     {'name': {'$in': list(fi['feature'].unique())}})
        columns_descr = pd.DataFrame(columns_descr)
        columns_descr.rename({'name': 'feature'}, axis=1, inplace=True)

        fi = fi.merge(columns_descr[['feature', 'value_name', 'indicator']], on='feature', how='left')

        fi = fi.sort_values(by='error_delta', ascending=False)

        fi['influence_factor'] = fi['error_delta'].apply(lambda error_delta: math.log(error_delta + 1)
                                    if error_delta > 0 else 0)

        if_sum = fi['influence_factor'].sum()
        fi['influence_factor'] = fi['influence_factor'] / if_sum

        fi_ind = fi.copy()

        fi_ind = fi_ind[['indicator',
                         'error_delta',
                         'influence_factor',
                         'value_name']].groupby(['indicator', 'value_name'], as_index=False).sum()

        fi_ind.rename({'indicator': 'feature'}, axis=1, inplace=True)

        fi_ind = fi_ind.sort_values(by='error_delta', ascending=False)

        fi = fi.to_dict('records')
        fi_ind = fi_ind.to_dict('records')

        self._write_feature_importances(fi, fi_ind)

    def _get_indicator_from_column_name(self, column_name: str) -> dict[str, Any]:
        """
        Gets dict of indicator from column name.
        :param column_name: name of columns
        :return: indicator dict
        """
        short_id = column_name.split('_')[1]

        return self._get_indicator_from_short_id(short_id)

    def _get_indicator_from_short_id(self, short_id: str) -> dict[str, Any]:
        """
        Gets dict of indicator from its short id.
        :param short_id: short id if indicator.
        :return: indicator dict
        """
        indicators = [ind for ind in (self.parameters.x_indicators + self.parameters.y_indicators)
                      if ind['short_id'] == short_id]

        return indicators[0]

    def _get_analytics_from_column_name(self, column_name: str) -> list[dict[str, Any]]:
        """
        Gets analytic list from column name.
        :param column_name: name of column
        :return: analytic list
        """
        column_list = column_name.split('_')

        if 'an' in column_list:
            result = [el['analytics'] for el
                      in (self.fitting_parameters.x_analytic_keys + self.fitting_parameters.y_analytic_keys)
                      if el['short_id'] == column_list[5]][0]
        else:
            result = []

        return result

    def _interrupt_fi_calculation_job(self) -> None:
        """
        Interrupt process of fi calculation if it launched in subprocess
        """
        if self.fitting_parameters.fi_status == FittingStatuses.Started:
            set_background_job_interrupted(self.fitting_parameters.fi_calculation_job_id)

    # noinspection PyUnusedLocal
    def _check_before_fa_calculation(self, data: list[dict[str, Any]], output_data_structure: dict[str, Any],
                            input_indicators: list[str], output_indicator: str) -> None:
        """
        Checks parameters before fa calculation. Raises ModelException if checking is failed
        """
        if not self._initialized:
            raise ModelException('Error of calculating factor analysis data. Model is not initialized')

        if not self.fitting_parameters.fitting_status == FittingStatuses.Fit:
            raise ModelException('Error of calculating factor analysis data. Model is not fit. ' +
                                 'Train the model before calculating')

        loading_engine = get_loading_engine_class()()

        loading_engine.check_data(data, checking_parameter_name='data', for_fa=True)

    # noinspection PyUnusedLocal
    def _get_output_value_for_fa(self, input_data: pd.DataFrame,
                                 output_data_structure: dict[str, Any],
                                 used_indicators: list[str],
                                 main_periods: list[str],
                                 output_columns: list[str],
                                 current_ind: str = '') -> float:
        """
        Forms output data according to one indicator while fa calculating
        :param input_data: main input data
        :param output_data_structure: based value, calculated value and value type
        :param used_indicators: ids of indicators, which is previously used
        :param main_periods: periods of fa calculating
        :param output_columns: list of columns of output indicator
        :param current_ind: id of current input indicator
        :return: one indicator fa data
        """

        c_input_data_pre = input_data.loc[(input_data['indicator'].isin(used_indicators)) &
                                          (input_data['scenario_type'] == data_api_types.ScenarioTypes.CALCULATED)]

        c_input_data_current = input_data.loc[(input_data['indicator'] == current_ind) &
                                          (input_data['scenario_type'] == data_api_types.ScenarioTypes.CALCULATED)]

        c_input_data_post = input_data.loc[(~input_data['indicator'].isin(used_indicators + [current_ind])) &
                                          (input_data['scenario_type'] == data_api_types.ScenarioTypes.BASE)]

        c_input_data = pd.concat([c_input_data_pre, c_input_data_current, c_input_data_post], axis=0,
                                 ignore_index=True)

        scenario = input_data['scenario'].unique()[0]
        c_input_data['scenario'] = scenario

        output_data = self._predict_model(c_input_data.to_dict('records'))

        output_data = output_data.loc[output_data['period'].isin(main_periods)].copy()

        output_data = output_data[output_columns]
        output_data['value'] = output_data.apply(sum, axis=1)

        output_value = output_data['value'].sum()

        return output_value

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _get_data_for_fa_graph(self, result_data: list[dict[str, Any]],
                               output_structure: dict[str, Any],
                               scenario_based: str,
                               scenario_calculated: str,
                               output_value_name: str) -> pd.DataFrame:
        """
        Forms dataframe using to create fa graph html
        :param result_data: fa data to form fa graph data
        :param output_structure: output data, calculated and based values
        :return: prepared dataframe
        """
        result_data = pd.DataFrame(result_data)

        all_indicators = list(result_data['indicator'].unique())
        indicators_descr = self._db_connector.get_lines('indicators',
                                                        {'id': {'$in': all_indicators}})
        indicators_dict = {el['id']: el['name'] for el in indicators_descr}

        result_data['title'] = result_data['indicator'].apply(lambda x: indicators_dict[x])

        need_to_add_other_line = len(result_data['indicator'].unique()) < len(self.parameters.x_indicators)
        result_data['order'] = list(range(2, result_data.shape[0]+2))

        result_data.drop(['indicator'], axis=1, inplace=True)

        based_scenario_descr = self._db_connector.get_line('scenarios', {'id': scenario_based})
        calculated_scenario_descr = self._db_connector.get_line('scenarios',
                                                                {'id': scenario_calculated})

        base_line = {'title': based_scenario_descr['name'], 'value': output_structure['output_value_based'], 'order': 1}

        lines_to_add = [base_line]

        order_of_calculated = result_data.shape[0] + 2
        if need_to_add_other_line:

            sum_all = float(result_data[['value']].apply(sum, axis=0))
            other_value = output_structure['output_value_calculated'] - sum_all - output_structure['output_value_based']

            if abs(other_value) >= 10:

                other_line = {'title': 'Прочие факторы', 'value': other_value, 'order': order_of_calculated}
                order_of_calculated += 1

                lines_to_add.append(other_line)

        calculated_line = {'title': calculated_scenario_descr['name'],
                           'value': output_structure['output_value_calculated'],
                           'order': order_of_calculated}

        lines_to_add.append(calculated_line)

        result_data = pd.concat([result_data, pd.DataFrame(lines_to_add)])

        result_data = result_data.sort_values('order')

        return result_data

    # noinspection PyMethodMayBeStatic
    def _get_fa_graph_bin(self, values: pd.DataFrame, out_indicator: str) -> str:
        """
        Forms fa graph html str
        :param values: values to form html
        :param out_indicator: id of output indicator
        :return: formed html str
        """

        output_indicator_descr = self._db_connector.get_line('indicators', {'id': out_indicator})

        x_list = list(values['title'])
        y_list = list(values['value'])

        text_list = []
        hover_text_list = []

        initial_value = 0
        for index, item in enumerate(y_list):
            if item > 0 and index != 0 and index != len(y_list) - 1:
                text_list.append('+{0:,.0f}'.format(y_list[index]).replace(',', ' '))
            else:
                text_list.append('{0:,.0f}'.format(y_list[index]).replace(',', ' '))

            hover_value = '{0:,.0f}'.format(item).replace(',', ' ')

            if index in (0, len(y_list)-1):
                hover_text_list.append('{}'.format(hover_value))
            else:
                if item > 0:
                    hover_value += ' &#9650;'
                elif item < 0:
                    hover_value += ' &#9660;'

                hover_text_list.append('{}<br>Предыдущее: {}'.format(hover_value,
                                                                '{0:,.0f}'.format(initial_value).replace(',', ' ')))

            initial_value += item

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
            title={'text': '<b>Факторный анализ</b><br><span style="color:#666666">{}</span>'.format(
                output_indicator_descr['name'])},
            showlegend=False,
            height=650,
            font={
                'family': 'Open Sans, light',
                'color': 'black',
                'size': 14
            },
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(tickformat=",.0f"),
            yaxis_title="руб.",
            shapes=dict_list
        )

        fig.update_xaxes(tickangle=-45, tickfont=dict(family='Open Sans, light', color='black', size=14))

        y_tick_vals, y_tick_texts = self._get_y_vals_texts_for_fa_graph(y_list)

        fig.update_yaxes(tickangle=0, tickfont=dict(family='Open Sans, light', color='black', size=14),
                         tickvals=y_tick_vals, ticktext=y_tick_texts)

        fig.update_traces(hoverinfo='text', hovertext=hover_text_list)

        graph_str = fig.to_html()

        return graph_str

    @staticmethod
    def _get_y_vals_texts_for_fa_graph(y_values: list[int | float]) -> [list[int | float], list[str]]:
        """
        Returns text of y-axis of fa graph
        @param y_values: values of y-axis
        @return: list of y-texts
        """
        max_value = 0
        current_value = 0

        for index, y_value in enumerate(y_values):
            if index == 0:
                current_value = 0
                max_value = y_value
            elif index == len(y_values):
                current_value = y_value
            else:
                current_value += y_value

            if current_value > max_value:
                max_value = current_value

        max_value = 1.5*max_value

        step = max_value/10

        step_pow = 0
        c_step = step

        while c_step > 10:
            c_step = c_step // 10
            step_pow += 1

        step = float('5e{}'.format(step_pow))
        step = int(step)

        value = 0

        result_values = []
        result_texts = []

        while value < max_value:
            result_values.append(value)
            result_texts.append('{0:,.0f}'.format(value).replace(',', ' '))
            value += step

        return result_values, result_texts

    def _get_metrics(self, y: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Returns value of model quality metrics
        @param y: True output values
        @param y_pred: Predicted output values
        @return: dict of metrics
        """
        metrics = dict()
        metrics['rsme'] = self._calculate_rsme(y, y_pred)
        metrics['mspe'] = self._calculate_mspe(y, y_pred)

        return metrics

    @staticmethod
    def _calculate_mspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates mean squared percentage error metric
        :param y_true: real output data
        :param y_pred: predicted output data
        :return: value of calculated metric
        """
        eps = np.zeros(y_true.shape)
        eps[:] = 0.0001
        y_p = np.c_[abs(y_true), abs(y_pred), eps]
        y_p = np.max(y_p, axis=1).reshape(-1, 1)

        return np.sqrt(np.nanmean(np.square(((y_true - y_pred) / y_p))))

    @staticmethod
    def _calculate_rsme(y_true, y_pred) -> float:
        """
        Calculates root mean squared error metric
        :param y_true: real output data
        :param y_pred: predicted output data
        :return: value of calculated metric
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))


def get_additional_actions() -> list[dict[str, Callable]]:
    """
    Formed additional actions of vbm model module
    :return: list of actions dict (functions)
    """

    result = list()

    result.append({'name': 'model_get_sensitivity_analysis', 'path': 'model/get_sensitivity_analysis',
                   'func': _get_sensitivity_analysis,
                   'http_method': 'post', 'requires_db': True})

    result.append({'name': 'model_calculate_feature_importances', 'path': 'model/calculate_feature_importances',
                   'func': _calculate_feature_importances,
                   'http_method': 'post', 'requires_db': True})

    result.append({'name': 'model_get_feature_importances', 'path': 'model/get_feature_importances',
                   'func': _get_feature_importances,
                   'http_method': 'get', 'requires_db': True})

    result.append({'name': 'model_drop_fi_calculation', 'path': 'model/drop_fi_calculation',
                   'func': _drop_fi_calculation,
                   'http_method': 'get', 'requires_db': True})

    result.append({'name': 'model_get_factor_analysis', 'path': 'model/get_factor_analysis',
                   'func': _get_factor_analysis,
                   'http_method': 'post', 'requires_db': True})

    return result


# noinspection PyShadowingNames
def _calculate_feature_importances(id: str, fi_parameters: api_types.FittingParameters,
                                   background_job: bool = False) -> general_api_types.BackgroundJobResponse:
    """
    For calculating feature importances
    :param id: id of model
    :param fi_parameters: parameters of fi calculating
    :param background_job: True if fi calculating is in background
    :return: result (info) of calculating fi
    """

    result = calculate_fi(id, fi_parameters.model_dump(), background_job=background_job)

    return general_api_types.BackgroundJobResponse.model_validate(result)


# noinspection PyShadowingNames
@execute_in_background
def calculate_fi(model_id: str, fi_parameters: dict[str, Any], job_id: str = '') -> dict[str, Any]:
    """
    For calculating feature importances
    :param model_id: id of model
    :param fi_parameters: parameters of fi calculating
    :param job_id: id of background job if fi calculating is in background
    :return: result (info) of calculating fi
    """

    model = VbmModel(model_id)
    result = model.calculate_feature_importances(fi_parameters, job_id=job_id)

    return {'description': result, 'mode': general_api_types.ExecutionModes.DIRECTLY, 'pid': 0}


def _drop_fi_calculation(id: str) -> str:
    """
    To drop calculated feature importances data
    :param id: model id to drop fi calculation
    :return: result of dropping
    """

    model = VbmModel(id)
    result = model.drop_fi_calculation()

    return result


def _get_sensitivity_analysis(id: str,
                input_data: data_api_types.SensitivityAnalysisInputData,
                              expand_by_periods: bool = False) -> data_api_types.SensitivityAnalysisOutputData:
    """
    For calculating and getting sensitivity analysis data
    :param id: id of model
    :param input_data: input data to calculate sensitivity analysis
    :param expand_by_periods: return result expanded by periods if true else grouped result
    :return: calculated sa data
    """

    input_data_dict = input_data.model_dump()

    model = VbmModel(id)
    result = model.get_sensitivity_analysis(input_data_dict['inputs'],
                                            input_data_dict['input_indicators'],
                                            input_data_dict['output_indicator'],
                                            input_data_dict['deviations'],
                                            input_data_dict['get_graph'],
                                            input_data_dict['auto_selection_number'],
                                            input_data_dict['value_name'], expand_by_periods)

    return data_api_types.SensitivityAnalysisOutputData.model_validate(result)


def _get_feature_importances(id: str, extended: bool = False) -> data_api_types.FeatureImportancesOutputData:
    """
    For getting feature importances data
    :param id: id of model
    :param extended: input data to calculate sensitivity analysis
    :return: calculated fi data
    """

    model = VbmModel(id)
    result = model.get_feature_importances(extended=extended)

    return data_api_types.FeatureImportancesOutputData.model_validate(result)


def _get_factor_analysis(id: str,  input_data: data_api_types.FactorAnalysisInputData,
                         get_graph: bool = False) -> data_api_types.FactorAnalysisOutputData:
    """
    For calculating and getting factor analysis data
    :param id: id of model
    :param input_data: input data to calculate factor analysis
    :param get_graph: returns fa graph html
    :return: calculated fa data
    """
    model = VbmModel(id)

    input_data_converted = input_data.model_dump()

    output_data_structure = dict()
    output_data_structure['output_value_based'] = input_data_converted['output_value_based']
    output_data_structure['output_value_calculated'] = input_data_converted['output_value_calculated']
    output_data_structure['output_value_name'] = input_data_converted['output_value_name']

    result = model.get_factor_analysis(input_data_converted['data'],
                                       output_data_structure,
                                       input_data_converted['input_indicators'],
                                       input_data_converted['output_indicator'],
                                       get_graph)

    return data_api_types.FactorAnalysisOutputData.model_validate(result)


def get_action_before_background_job(func_name: str, args: tuple[Any], kwargs: dict[str, Any]) -> Optional[Callable]:
    """Returns function which will be executed before model fi calculating
    @param func_name: name of fi calculating function
    @param args: positional arguments of fi calculating function
    @param kwargs: keyword arguments of fi calculating function.
    @return: function to execute before fi calculating
    """
    model = VbmModel(args[0])
    result = model.get_action_before_background_job(func_name, args, kwargs)

    return result
