
from typing import Any, Optional
import pandas as pd

from vm_logging.exceptions import ModelException

from .base_transformer import RowColumnTransformer, Checker


class VbmChecker(Checker):
    service_name = 'vbm'
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        if x.empty:
            raise ModelException('There are no data to fit model. Check loading data, '
                                 'model settings and fitting filter')

        indicator_ids =list(x['indicator_short_id'].unique())

        if self._fitting_mode and self._fitting_parameters.is_first_fitting():
            model_indicators = self._model_parameters.x_indicators + self._model_parameters.y_indicators

            model_indicator_ids = [el['short_id'] for el in model_indicators]

            model_indicator_ids = set(model_indicator_ids)

            error_ids =  []

            for el in model_indicator_ids:
                if el not in indicator_ids:
                    error_ids.append(el)

            if error_ids:
                error_names = [el['name'] for el in self._model_parameters.x_indicators +
                                   self._model_parameters.x_indicators if el['short_id'] in error_ids]
                error_names = ['"{}"'.format(el) for el in error_names]

                raise ModelException('Indicator(s) {} are not in fitting data'.format(', '.join(error_names)))

        return x


class VbmRowColumnTransformer(RowColumnTransformer):
    service_name = 'vbm'

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        raw_data = self._get_raw_data_from_x(x)

        data_result = self._get_grouped_raw_data(raw_data)

        x_columns = []
        y_columns = []

        for ind_parameters in self._model_parameters.x_indicators + self._model_parameters.y_indicators:

            if ind_parameters.get('period_shift'):
                continue

            ind_data = self._get_raw_data_by_indicator(raw_data, ind_parameters)

            analytic_keys, analytic_ids = self._get_analytic_parameters_from_data(ind_data, ind_parameters)

            if not analytic_ids:
                analytic_ids.append('')

            for analytic_id in analytic_ids:

                column_name = self._get_column_name(ind_parameters['short_id'], analytic_id, ind_parameters)

                self._check_append_column_names(column_name, x_columns, y_columns, ind_parameters)

                an_data = self._get_raw_data_by_analytics(ind_data, analytic_id)

                data_result = data_result.merge(an_data, on=['organisation', 'scenario', 'period'], how='left')
                data_result = data_result.rename({'value': column_name}, axis=1)

        if self._fitting_mode and self._fitting_parameters.is_first_fitting():
            self._fitting_parameters.x_columns = x_columns
            self._fitting_parameters.y_columns = y_columns

        return data_result

    @staticmethod
    def _get_raw_data_from_x(x: pd.DataFrame) -> pd.DataFrame:
        raw_data = x
        raw_data.rename({'organisation': 'organisation_struct', 'scenario': 'scenario_struct', 'period': 'period_str',
                         'indicator': 'indicator_struct'}, axis=1, inplace=True)

        raw_data.rename({'organisation_id': 'organisation', 'scenario_id': 'scenario', 'period_date': 'period',
                         'indicator_short_id': 'indicator'}, axis=1, inplace=True)

        return raw_data

    @staticmethod
    def _get_grouped_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
        return raw_data[['organisation', 'scenario',
                                 'period']].groupby(by=['organisation', 'scenario', 'period'], as_index=False).sum()

    def _get_raw_data_by_indicator(self, data: pd.DataFrame, indicator_parameters: dict[str, Any]) -> pd.DataFrame:

        fields = ['organisation', 'scenario', 'period', 'analytics_key_id','analytics', 'value']
        ind_data = data[fields].loc[data['indicator'] == indicator_parameters['short_id']]

        return ind_data

    def _get_raw_data_by_analytics(self, data: pd.DataFrame, analytic_id: str) -> pd.DataFrame:

        fields = ['organisation', 'scenario', 'period', 'analytics_key_id', 'value']
        an_data = data[fields].loc[data['analytics_key_id'] == analytic_id]

        fields = ['organisation', 'scenario', 'period']
        an_data = an_data[fields + ['value']].groupby(fields, as_index=False).sum()

        return an_data

    def _get_analytic_parameters_from_data(self, data: pd.DataFrame,
                                    indicator_parameters: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:

        if indicator_parameters['use_analytics']:

            if self._fitting_mode and self._fitting_parameters.is_first_fitting():

                keys, ids = self._get_analytic_parameters_for_new_fitting(data, indicator_parameters)

            else:
                keys = indicator_parameters.get('analytic_keys') or []
                ids = [el['short_id'] for el in keys]

        else:

            keys, ids = [], []

        return keys, ids


    def _get_analytic_parameters_for_new_fitting(self, data: pd.DataFrame,
                                    indicator_parameters: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:

        ids = list(data['analytics_key_id'].unique())

        data['number'] = data.index
        fields_to_group = ['analytics_key_id']
        fields = fields_to_group + ['number']
        grouped_ind_data = data[fields].groupby(by=fields_to_group, as_index=False).min()

        ind_data = grouped_ind_data.merge(data, on=['analytics_key_id', 'number'], how='left')

        analytics_data = ind_data[['analytics_key_id', 'analytics']].to_dict('records')

        keys = []
        for analytic_el in analytics_data:
            key = {'short_id': analytic_el['analytics_key_id'], 'analytics': analytic_el['analytics']}
            keys.append(key)

            model_keys = (self._fitting_parameters.y_analytic_keys if self._is_y_indicator(indicator_parameters)
                          else self._fitting_parameters.x_analytic_keys)

            model_key_ids = [el['short_id'] for el in model_keys]

            if analytic_el['analytics_key_id'] not in model_key_ids:
                model_keys.append(key)

            model_analytics = (self._fitting_parameters.y_analytics if self._is_y_indicator(indicator_parameters)
                               else self._fitting_parameters.x_analytics)

            model_analytic_ids = [el['short_id'] for el in model_analytics]

            for an_el in analytic_el['analytics']:
                if an_el['short_id'] not in model_analytic_ids:
                    model_analytics.append(an_el)

        indicator_parameters['analytic_keys'] = keys

        return keys, ids


    def _is_y_indicator(self, indicator_parameters: dict[str, Any]) -> bool:
        y_ids = [el['short_id'] for el in self._model_parameters.y_indicators]

        return indicator_parameters['short_id'] in y_ids


    def _get_column_name(self, indicator_id, analytic_id, indicator_parameters):

        if indicator_parameters['use_analytics']:
            result = 'ind_{}_an_{}'.format(indicator_id, analytic_id)
        else:
            result = 'ind_{}'.format(indicator_id)

        if indicator_parameters.get('period_shift'):
            if indicator_parameters['period_shift'] < 0:
                result += '_p_m{}'.format(indicator_parameters['period_shift'])
            else:
                result += '_p_p{}'.format(indicator_parameters['period_shift'])
        elif indicator_parameters.get('period_number'):
            result += '_p_n{}'.format(indicator_parameters['period_number'])
        elif indicator_parameters.get('period_accumulation'):
            result += '_p_a{}'.format(indicator_parameters['period_accumulation'])

        return result

    def _check_append_column_names(self, column_name: str,
                                   x_columns: list[str],
                                   y_columns: list[str],
                                   indicator_parameters: dict[str, Any]) -> None:

        if self._fitting_mode and self._fitting_parameters.is_first_fitting():
            if self._is_y_indicator(indicator_parameters):
                y_columns.append(column_name)
            else:
                x_columns.append(column_name)
        else:
            if self._is_y_indicator(indicator_parameters):
                if column_name not in self._fitting_parameters.y_columns:
                    raise ModelException('Column name "{}" not in y columns'.format(column_name))
            else:
                if column_name not in self._fitting_parameters.x_columns:
                    raise ModelException('Column name "{}" not in x columns'.format(column_name))

