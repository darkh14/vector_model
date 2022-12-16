
from typing import Any, Optional, TypeVar
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import MinMaxScaler
import pickle

from vm_logging.exceptions import ModelException
from ..model_parameters.base_parameters import ModelParameters, FittingParameters
from .base_transformer import Reader, RowColumnTransformer, Checker, CategoricalEncoder, NanProcessor, Scaler
from data_processing.loading_engines.vbm_engine import VbmEngine

VbmScalerClass = TypeVar('VbmScalerClass', bound='VbmScaler')


class VbmReader(Reader):
    service_name = 'vbm'
    def _read_while_predicting(self, data: list[dict[str, Any]]) -> pd.DataFrame:

        loading_engine = VbmEngine()
        return loading_engine.preprocess_data(data)


class VbmChecker(Checker):
    service_name = 'vbm'
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        if x.empty:
            raise ModelException('There are no data to {} model. Check loading data, '.format('fit'
                                if self._fitting_mode else 'predict') + 'model settings and filter')


        if self._fitting_mode and self._fitting_parameters.is_first_fitting():

            indicator_ids = list(x['indicator_short_id'].unique())

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

            ind_data = self._get_raw_data_by_indicator(raw_data, ind_parameters)

            ind_data = self._process_data_periods(ind_data, ind_parameters)

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

        fields = ['organisation', 'scenario', 'period', 'periodicity', 'analytics_key_id','analytics', 'value']
        ind_data = data[fields].loc[data['indicator'] == indicator_parameters['short_id']]

        return ind_data

    def _get_raw_data_by_analytics(self, data: pd.DataFrame, analytic_id: str) -> pd.DataFrame:

        fields = ['organisation', 'scenario', 'period', 'periodicity', 'analytics_key_id', 'value']
        if analytic_id:
            an_data = data[fields].loc[data['analytics_key_id'] == analytic_id]
        else:
            an_data = data[fields]

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
                result += '_p_m{}'.format(-indicator_parameters['period_shift'])
            else:
                result += '_p_p{}'.format(indicator_parameters['period_shift'])
        elif indicator_parameters.get('period_number'):
            result += '_p_n{}'.format(indicator_parameters['period_number'])
        elif indicator_parameters.get('period_accumulation'):
            result += '_p_a'

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


    def _process_data_periods(self, data: pd.DataFrame, indicator_parameters: dict[str, Any]) -> pd.DataFrame:

        result_data = data.copy()

        if indicator_parameters.get('period_shift'):
            result_data = self._process_data_periods_shift(result_data, indicator_parameters)
        elif indicator_parameters.get('period_number'):
            result_data = self._process_data_periods_number(result_data, indicator_parameters)
        elif indicator_parameters.get('period_accumulation'):
            result_data = self._process_data_periods_accumulation(result_data)

        return result_data

    def _process_data_periods_shift(self, data: pd.DataFrame, indicator_parameters: dict[str, Any]) -> pd.DataFrame:
        data['period_shift'] = indicator_parameters['period_shift']
        data['temp_period'] = data[['period', 'periodicity', 'period_shift']].apply(self._shift_data_period, axis=1)

        data = data.drop(['period_shift', 'period'], axis=1)
        data = data.rename({'temp_period': 'period'}, axis=1)

        return data

    def _process_data_periods_number(self, data: pd.DataFrame, indicator_parameters: dict[str, Any]) -> pd.DataFrame:
        data['period_number'] = data[['period', 'periodicity']].apply(self._get_period_number, axis=1)
        data['year'] = data['period'].apply(lambda x: x.year)

        data = data.loc[data['period_number'] == indicator_parameters['period_number']]

        years_scenarios = data[['year', 'scenario', 'periodicity']].groupby(['year',
                                       'scenario', 'periodicity'], as_index=False).sum().to_dict('records')

        all_scenarios_periods = pd.DataFrame(columns=['period_temp', 'scenario'])

        scenarios_periodicity = []
        scenarios = []
        for el in years_scenarios:
            if el['scenario'] not in scenarios:
                scenarios.append(el['scenario'])
                scenarios_periodicity.append({'scenario': el['scenario'], 'periodicity': el['periodicity']})

        for sc_p in scenarios_periodicity:
            years = [el['year'] for el in years_scenarios if el['scenario'] == sc_p['scenario']]
            sc_periods = self._form_all_periods_by_years(years, sc_p['periodicity'])

            periods_df = pd.DataFrame(sc_periods, columns=['period_temp'])
            periods_df['scenario'] = sc_p['scenario']

            all_scenarios_periods = pd.concat([all_scenarios_periods, periods_df])

        data = data.merge(all_scenarios_periods, on=['scenario'], how='left')
        data = data.drop(['period_number', 'period', 'year'], axis=1)
        data = data.rename({'period_temp': 'period'}, axis=1)

        return data

    def _process_data_periods_accumulation(self, data: pd.DataFrame) -> pd.DataFrame:

        temp_data = data.copy()

        temp_data['period_number'] =  temp_data[['period', 'periodicity']].apply(self._get_period_number, axis=1)

        scenarios_periodicity = data[['scenario', 'periodicity']].groupby(['scenario',
                                'periodicity'], as_index=False).sum().to_dict('records')

        result_data = pd.DataFrame(columns=['organisation', 'scenario', 'period', 'value'])

        for sc_p in scenarios_periodicity:
            period_numbers = self._get_period_numbers_in_year(sc_p['periodicity'])

            for number in period_numbers:

                num_data = temp_data[['organisation', 'scenario', 'period', 'period_number', 'periodicity',
                                      'analytics_key_id', 'value']].loc[(temp_data['scenario'] == sc_p['scenario'])
                                      & (temp_data['period_number'] <= number)]

                num_data = num_data[['organisation', 'scenario', 'period', 'periodicity', 'analytics_key_id',
                                     'value']].groupby(['organisation', 'periodicity', 'analytics_key_id',
                                     'scenario'], as_index=False).agg({'period': 'max', 'value': 'sum'})

                result_data = pd.concat([result_data, num_data])

        return result_data

    def _shift_data_period(self, data:pd.DataFrame) -> datetime:

        result =  self._add_to_period(data['period'], -data['period_shift'], data['periodicity'])

        return result

    def _form_all_periods_by_years(self, years: list[int], periodicity: str) -> list[datetime]:

        all_periods = []

        for year in years:
            current_period = datetime(year=year, month=1, day=1)
            end_of_year = datetime(year=year, month=12, day=31)

            while current_period <= end_of_year:
                all_periods.append(current_period)
                current_period = self._add_to_period(current_period, 1, periodicity)

        return all_periods

    @staticmethod
    def _add_to_period(period: datetime, shift: int, periodicity: str) -> datetime:

        if periodicity not in ['day', 'week', 'decade', 'month', 'quarter', 'half_year', 'year']:
            raise ModelException('Unknown periodicity "{}"'.format(periodicity))

        if periodicity == 'decade':
            shifting_parameters ={'days': 10*shift}
        elif periodicity == 'quarter':
            shifting_parameters ={'months': 3*shift}
        elif periodicity == 'half_year':
            shifting_parameters ={'months': 6*shift}
        else:
            shifting_parameters ={periodicity + 's': shift}

        return period + relativedelta(**shifting_parameters)

    def _get_period_number(self, data: pd.Series) -> int:

        if data['periodicity'] == 'month':
            result = data['period'].month
        else:
            current_date = datetime(year=data['period'].year, month=1, day=1)

            result = 1
            while current_date < data['period']:
                current_date = self._add_to_period(current_date, 1, data['periodicity'])
                result += 1

        return result

    def _get_period_numbers_in_year(self, periodicity: str) -> list[int]:

        if periodicity == 'day':
            result = 365
        elif periodicity == 'month':
            result = 12
        elif periodicity == 'quarter':
            result = 4
        elif periodicity == 'half_year':
            result = 2
        elif periodicity == 'year':
            result = 1
        else:
            current_year = datetime.now().year
            current_date = datetime(year=current_year, month=1, day=1)
            end_of_year = datetime(year=current_year, month=12, day=31)
            result = 1
            while current_date < end_of_year:
                current_date = self._add_to_period(current_date, 1, periodicity)
                result += 1

        result = list(range(1, result+1))

        return result


class VbmCategoricalEncoder(CategoricalEncoder):
    service_name = 'vbm'

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, db_path: str, **kwargs):
        super().__init__(model_parameters, fitting_parameters, db_path, **kwargs)
        self._fields = ['scenario']

        if kwargs.get('fields'):
            self._fields = kwargs['fields']

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        added_fields = []

        for field in self._fields:

            values = list(x[field].unique())


            for value in values:
                model_field = '{}_{}'.format(field, value)
                need_to_add = self._check_model_field(model_field)
                if need_to_add:
                    x[model_field] = x[field].apply(lambda l: 1 if l == value else 0)
                    added_fields.append(model_field)

        if not self._fitting_mode or not self._fitting_parameters.is_first_fitting():
            for col in self._fitting_parameters.categorical_columns:
                if col not in added_fields:
                    x[col] = 0

        return x

    def _check_model_field(self, field) -> bool:

        result = False
        if self._fitting_mode and self._fitting_parameters.is_first_fitting():
            if field not in self._fitting_parameters.x_columns:
                self._fitting_parameters.x_columns.append(field)
                self._fitting_parameters.categorical_columns.append(field)
                result = True
        else:
            result = field in self._fitting_parameters.x_columns

        return result


class VbmNanProcessor(NanProcessor):
    service_name = 'vbm'
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        if self._fitting_mode:

            x_digit_columns = self._fitting_parameters.x_columns

            y_digit_columns = self._fitting_parameters.y_columns

            x_not_del = x[x_digit_columns].isnull().all(axis=1)
            y_not_del = x[y_digit_columns].isnull().all(axis=1)

            x_not_del = x_not_del
            y_not_del = y_not_del

            x['x_del'] = x_not_del
            x['y_del'] = y_not_del

            x['not_del'] = x[['x_del', 'y_del']].apply(lambda l: not l['x_del'] and not l['y_del'], axis=1)

            x = x.loc[x['not_del'] == True]

            x = x.drop(['not_del', 'x_del', 'y_del'], axis=1)

            if x[self._fitting_parameters.x_columns].empty:
                raise ModelException('All data are empty. Fitting is impossible')

            if x[self._fitting_parameters.y_columns].empty:
                raise ModelException('All labels are empty. Fitting is impossible')

            if self._fitting_parameters.is_first_fitting():

                all_columns = self._fitting_parameters.x_columns + self._fitting_parameters.y_columns

                cols_to_delete = []
                for d_col in all_columns:
                    if x[d_col].isnull().all():
                        cols_to_delete.append(d_col)
                        if d_col in self._fitting_parameters.x_columns:
                            self._fitting_parameters.x_columns.remove(d_col)

                        if d_col in self._fitting_parameters.y_columns:
                            self._fitting_parameters.y_columns.remove(d_col)

                if not self._fitting_parameters.x_columns:
                    raise ModelException('All x columns are empty. Fitting is impossible')

                if not self._fitting_parameters.y_columns:
                    raise ModelException('All y columns are empty. Fitting is impossible')

                x= x.drop(cols_to_delete, axis=1)

        x = x.fillna(0)

        return x


class VbmScaler(Scaler):
    service_name = 'vbm'

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, db_path: str, **kwargs):

        super().__init__(model_parameters, fitting_parameters, db_path, **kwargs)

        if 'model_id' not in kwargs:
            raise ModelException('Parameter "model_id" not found in additional parameter for VbmScaler object')

        self._model_id = kwargs['model_id']

        self._scaler_engine = MinMaxScaler()

        self._model_id = ''

        self._read_from_db()

    def fit(self, x: Optional[list[dict[str, Any]] | pd.DataFrame] = None,
            y: Optional[list[dict[str, Any]] | pd.DataFrame] = None) -> VbmScalerClass:

        non_categorical_columns = [el for el in self._fitting_parameters.x_columns
                                   if el not in self._fitting_parameters.categorical_columns]

        self._scaler_engine.fit(x[non_categorical_columns])

        self._write_to_db()

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        non_categorical_columns = [el for el in self._fitting_parameters.x_columns
                                   if el not in self._fitting_parameters.categorical_columns]

        result = x.copy()
        result[non_categorical_columns] = self._scaler_engine.transform(x[non_categorical_columns])

        return result

    def drop(self):
        self._db_connector.delete_lines('scalers', {'model_id': self._model_id})

    def _write_to_db(self):
        line_to_db = {'model_id': self._model_id, 'engine': pickle.dumps(self._scaler_engine)}

        self._db_connector.set_line('scalers', line_to_db, {'model_id': self._model_id})

    def _read_from_db(self):
        line_from_db = self._db_connector.get_line('scalers', {'model_id': self._model_id})

        if line_from_db:
            self._scaler_engine = pickle.loads(line_from_db['engine'])
