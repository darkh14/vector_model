"""
    VBM (Vector budget model)
    Module for transformer classes.
    Classes:
        VbmReader - for reading data from db
        VbmChecker - for checking data (fail if data is empty)
        VbmRowColumnTransformer - for forming data structure (indicator-analytics-period columns)
        VbmCategoricalEncoder - for forming categorical fields (now is not categorical fields)
        VbmNanProcessor - for working with nan values (deletes nan rows, columns, fills 0 to na values)
        VbmScaler - for data scaling (min-max scaling)

"""

from typing import Any, Optional, TypeVar, ClassVar
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import MinMaxScaler
import pickle

from vm_logging.exceptions import ModelException
from ..model_parameters.base_parameters import ModelParameters, FittingParameters
from .base_transformer import Reader, RowColumnTransformer, Checker, CategoricalEncoder, NanProcessor, Scaler
from data_processing.data_preprocessors import get_data_preprocessing_class

VbmScalerClass = TypeVar('VbmScalerClass', bound='VbmScaler')


class VbmReader(Reader):
    """ For reading data from db. Data preprocessor added """
    service_name: ClassVar[str] = 'vbm'

    def _read_while_predicting(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Added data preprocessor
        :param data: input data
        :return: preprocessed data
        """
        data_preprocessor = get_data_preprocessing_class()()
        return data_preprocessor.preprocess_data_for_predicting(data)


class VbmChecker(Checker):
    """ For data checking. Fail if data is empty """
    service_name: ClassVar[str] = 'vbm'

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Checking indicator columns, checking data emptiness. Raises ModelException if failed
        :param x: input data.
        :return: input data without transforming
        """
        if x.empty:
            raise ModelException('There are no data to {} model. Check loading data, '.format('fit'
                                  if self._fitting_mode else 'predict') + 'model settings and filter')

        if self._fitting_mode and self._fitting_parameters.is_first_fitting():

            indicator_ids = list(x['indicator_short_id'].unique())

            model_indicators = self._model_parameters.x_indicators + self._model_parameters.y_indicators

            model_indicator_ids = [el['short_id'] for el in model_indicators]

            model_indicator_ids = set(model_indicator_ids)

            error_ids = []

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
    """ Transformer for forming data structure (indicator-analytics-period columns) """
    service_name: ClassVar[str] = 'vbm'

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Main method for row-column data transformation
        :param x: input data.
        :return: data after transforming
        """
        raw_data = self._get_raw_data_from_x(x)

        data_result = self._get_grouped_raw_data(raw_data)

        x_columns = []
        y_columns = []

        all_indicators = self._model_parameters.x_indicators
        if self._fitting_mode:
            all_indicators = all_indicators + self._model_parameters.y_indicators

        for ind_parameters in all_indicators:

            for value_name in ind_parameters['values']:

                ind_data = self._get_raw_data_by_indicator(raw_data, ind_parameters, value_name)

                ind_data = self._process_data_periods(ind_data, ind_parameters)

                analytic_keys, analytic_ids = self._get_analytic_parameters_from_data(ind_data, ind_parameters)

                if not analytic_ids:
                    analytic_ids.append('')

                for analytic_id in analytic_ids:

                    column_name = self._get_column_name(ind_parameters['short_id'], analytic_id, value_name,
                                                        ind_parameters)

                    self._check_append_column_names(column_name, x_columns, y_columns, ind_parameters)

                    an_data = self._get_raw_data_by_analytics(ind_data, analytic_id)

                    data_result = data_result.merge(an_data, on=['organisation', 'scenario', 'period'], how='left')
                    data_result = data_result.rename({'value': column_name}, axis=1)

        if self._fitting_mode and self._fitting_parameters.is_first_fitting():
            self._fitting_parameters.x_columns = x_columns
            self._fitting_parameters.y_columns = y_columns
        else:

            all_columns = self._fitting_parameters.x_columns
            if self._fitting_mode:
                all_columns = all_columns + self._fitting_parameters.y_columns

            for col in all_columns:
                if col not in x_columns + y_columns and col not in self._fitting_parameters.categorical_columns:
                    data_result[col] = 0

        if not self._fitting_mode:
            data_result = self._add_data_while_predicting(data_result, raw_data)

        return data_result

    def _add_data_while_predicting(self, data_result: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds special fields to data after transforming while predicting
        :param data_result: data after transforming
        :param raw_data: data before transforming
        :return: data after fields added
        """
        merge_data_columns = ['organisation', 'organisation_struct', 'scenario', 'scenario_struct', 'index']
        data_to_merge = raw_data[merge_data_columns]

        merge_columns = ['organisation', 'scenario', 'index']

        data_result = data_result.merge(data_to_merge, on=merge_columns, how='left')

        data_result['period'] = data_result['period'].apply(lambda x: x.strftime('%d.%m.%Y'))

        return data_result

    @staticmethod
    def _get_raw_data_from_x(x: pd.DataFrame) -> pd.DataFrame:
        """
        transforms data list of dicts to pd.DataFrame
        :param x: input data (list of dicts)
        :return: data (pd.DataFrame)
        """
        raw_data = x

        raw_data['index'] = raw_data.index

        raw_data.rename({'organisation': 'organisation_struct', 'scenario': 'scenario_struct', 'period': 'period_str',
                         'indicator': 'indicator_struct'}, axis=1, inplace=True)

        raw_data.rename({'organisation_id': 'organisation', 'scenario_id': 'scenario', 'period_date': 'period',
                         'indicator_short_id': 'indicator'}, axis=1, inplace=True)

        return raw_data

    @staticmethod
    def _get_grouped_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Groups raw data to use in transforming
        :param raw_data: all raw data before transforming
        :return: grouped data
        """
        return raw_data[['organisation', 'scenario',
                        'period', 'index']].groupby(by=['organisation', 'scenario', 'period'], as_index=False).min()

    def _get_raw_data_by_indicator(self, data: pd.DataFrame, indicator_parameters: dict[str, Any],
                                   value_name: str) -> pd.DataFrame:
        """
        Gets raw_data where indicator == required indicator
        :param data: all data
        :param indicator_parameters: parameters of required indicator
        :return: data with one indicator
        """
        fields = ['organisation', 'scenario', 'period', 'periodicity', 'analytics_key_id', 'analytics', value_name]
        ind_data = data[fields].loc[data['indicator'] == indicator_parameters['short_id']]

        ind_data = ind_data.rename({value_name: 'value'}, axis=1)

        return ind_data

    def _get_raw_data_by_analytics(self, data: pd.DataFrame, analytic_id: str) -> pd.DataFrame:
        """
        Gets raw indicator data by analytic key
        :param data: data with one indicator
        :param analytic_id: id of required analytic key
        :return: data with one indicator and one analytic key
        """
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
        """
        Gets all analytic keys and ids in data (for predicting and secondary fitting)
        :param data: data with one indicator
        :param indicator_parameters:
        :return: analytic keys and ids
        """
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
        """
        Gets analytic keys and id for new fitting
        :param data: data with one indicator
        :param indicator_parameters: parameters of current indicator
        :return: all analytic keys and ids found in data
        """
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
        """
        Returns True if current indicator is in y indicators, else False
        :param indicator_parameters: parameters of current indicators
        :return: True if current indicator is in y indicators, else False
        """
        y_ids = [el['short_id'] for el in self._model_parameters.y_indicators]

        return indicator_parameters['short_id'] in y_ids

    def _get_column_name(self, indicator_id: str, analytic_id: str, value_name: str,
                         indicator_parameters: dict[str, Any]) -> str:
        """
        Form column name from indicator id, analytic key id and period parameters
        :param indicator_id: id of current indicator
        :param analytic_id: id of current analytic key
        :param value_name: name of value ex. "value" or "value_quantity"
        :param indicator_parameters: parameters of current indicator
        :return: column name
        """

        if indicator_parameters['use_analytics']:
            result = 'ind_{}_val_{}_an_{}'.format(indicator_id, value_name, analytic_id)
        else:
            result = 'ind_{}_val_{}'.format(indicator_id, value_name)

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
        """
        Checks columns name and append it to all columns if it needs
        :param column_name: name of current column
        :param x_columns: x_columns array
        :param y_columns: y_columns array
        :param indicator_parameters: parameters of current indicator
        """
        if self._is_y_indicator(indicator_parameters):
            y_columns.append(column_name)
        else:
            x_columns.append(column_name)

        if not self._fitting_mode or not self._fitting_parameters.is_first_fitting():

            if self._is_y_indicator(indicator_parameters):
                if column_name not in self._fitting_parameters.y_columns:
                    raise ModelException('Column name "{}" not in y columns'.format(column_name))
            else:
                if column_name not in self._fitting_parameters.x_columns:
                    raise ModelException('Column name "{}" not in x columns'.format(column_name))

    def _process_data_periods(self, data: pd.DataFrame, indicator_parameters: dict[str, Any]) -> pd.DataFrame:
        """
        Forms period columns in data.
        :param data: data with one indicator
        :param indicator_parameters: parameters of current indicator
        :return: data with required period column
        """
        result_data = data.copy()

        if indicator_parameters.get('period_shift'):
            result_data = self._process_data_periods_shift(result_data, indicator_parameters)
        elif indicator_parameters.get('period_number'):
            result_data = self._process_data_periods_number(result_data, indicator_parameters)
        elif indicator_parameters.get('period_accumulation'):
            result_data = self._process_data_periods_accumulation(result_data)

        return result_data

    def _process_data_periods_shift(self, data: pd.DataFrame, indicator_parameters: dict[str, Any]) -> pd.DataFrame:
        """
        Forms period columns in data when period shift
        :param data: data with one indicator
        :param indicator_parameters: parameters of current indicator
        :return: data with required period column
        """
        data['period_shift'] = indicator_parameters['period_shift']
        data['temp_period'] = data[['period', 'periodicity', 'period_shift']].apply(self._shift_data_period, axis=1)

        data = data.drop(['period_shift', 'period'], axis=1)
        data = data.rename({'temp_period': 'period'}, axis=1)

        return data

    def _process_data_periods_number(self, data: pd.DataFrame, indicator_parameters: dict[str, Any]) -> pd.DataFrame:
        """
        Forms period columns in data when period number is fixed
        :param data: data with one indicator
        :param indicator_parameters: parameters of current indicator
        :return: data with required period column
        """

        data_result = data.copy()

        data_result['period_number'] = data_result[['period', 'periodicity']].apply(self._get_period_number, axis=1)
        data_result['year'] = data_result['period'].apply(lambda x: x.year)

        data_values = data_result.loc[data_result['period_number'] == indicator_parameters['period_number']]
        data_values = data_values[['organisation', 'scenario', 'analytics_key_id', 'year', 'value']]

        data_result = data_result.drop(['value', 'period'], axis=1)

        years_scenarios = data_result[['year', 'scenario', 'periodicity']].groupby(['year',
                                       'scenario', 'periodicity'], as_index=False).sum().to_dict('records')

        all_scenarios_periods = pd.DataFrame(columns=['period', 'scenario', 'year'])

        scenarios_periodicity = []
        scenarios = []
        for el in years_scenarios:
            if el['scenario'] not in scenarios:
                scenarios.append(el['scenario'])
                scenarios_periodicity.append({'scenario': el['scenario'], 'periodicity': el['periodicity']})

        for sc_p in scenarios_periodicity:
            years = [el['year'] for el in years_scenarios if el['scenario'] == sc_p['scenario']]
            sc_periods = self._form_all_periods_by_years(years, sc_p['periodicity'])

            periods_df = pd.DataFrame(sc_periods, columns=['period'])
            periods_df['scenario'] = sc_p['scenario']
            periods_df['year'] = periods_df['period'].apply(lambda x: x.year)

            all_scenarios_periods = pd.concat([all_scenarios_periods, periods_df])

        data_result = data_result[['organisation', 'scenario', 'periodicity', 'analytics_key_id',
        'year']].groupby(['organisation', 'scenario', 'year', 'periodicity', 'analytics_key_id'], as_index=False).sum()

        data_result = data_result.merge(all_scenarios_periods, on=['scenario', 'year'], how='left')

        data_result = data_result.merge(data_values, on=['organisation',
                                                         'analytics_key_id', 'scenario', 'year'], how='left')

        data_result = data_result.drop(['year'], axis=1)

        return data_result

    def _process_data_periods_accumulation(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Forms period columns in data when period is accumulated
        :param data: data with one indicator
        :return: data with required period column
        """
        temp_data = data.copy()

        temp_data['period_number'] = temp_data[['period', 'periodicity']].apply(self._get_period_number, axis=1)

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

    def _shift_data_period(self, data: pd.DataFrame) -> datetime:
        """
        Shifts period column in data.
        :param data: data before shifting
        :return: data after shifting
        """
        result = self._add_to_period(data['period'], -data['period_shift'], data['periodicity'])

        return result

    def _form_all_periods_by_years(self, years: list[int], periodicity: str) -> list[datetime]:
        """
        returns all periods belonging to years in years array
        :param years: list of years
        :param periodicity: periodicity of periods
        :return: list of periods
        """
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
        """
        Adds some periods to period according to periodicity
        :param period: period to add
        :param shift: how many periods need to add
        :param periodicity: periodicity of adding
        :return: period after adding
        """
        if periodicity not in ['day', 'week', 'decade', 'month', 'quarter', 'half_year', 'year']:
            raise ModelException('Unknown periodicity "{}"'.format(periodicity))

        if periodicity == 'decade':
            shifting_parameters = {'days': 10*shift}
        elif periodicity == 'quarter':
            shifting_parameters = {'months': 3*shift}
        elif periodicity == 'half_year':
            shifting_parameters = {'months': 6*shift}
        else:
            shifting_parameters = {periodicity + 's': shift}

        return period + relativedelta(**shifting_parameters)

    def _get_period_number(self, data: pd.Series) -> int:
        """
        Returns number of period in year
        :param data: series of period
        :return: number in year
        """
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
        """
        Gets all number of periods in year according to periodicity.
        :param periodicity: periodicity of period
        :return: list of numbers in year
        """
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
    """ Transformer for forming categorical fields (now is not categorical fields) """
    service_name: ClassVar[str] = 'vbm'

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, **kwargs) -> None:
        """
        Defines fields parameter (fields to be encoded)
        :param model_parameters: model parameters object
        :param fitting_parameters: fitting parameters object
        :param kwargs: additional parameters
        """
        super().__init__(model_parameters, fitting_parameters, **kwargs)
        self._fields = []

        if kwargs.get('fields'):
            self._fields = kwargs['fields']

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Adds encoding fields
        :param x: data before encoding
        :return: data after encoding
        """
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
        """
        Check is categorical field in fields.
        :param field: field to check
        :return: result of checking
        """
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
    """ Transformer for working with nan values (deletes nan rows, columns, fills 0 to na values) """
    service_name: ClassVar[str] = 'vbm'

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Process nan values: removes all nan rows and columns, fills 0 instead single nan values
        :param x: data before nan processing
        :return: data after na  processing
        """
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

                x = x.drop(cols_to_delete, axis=1)

        x = x.fillna(0)

        return x


class VbmScaler(Scaler):
    """ Transformer for data scaling (min-max scaling) """
    service_name: ClassVar[str] = 'vbm'

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, **kwargs):
        """
        Defines model id, scaler engine. Reads from db if it is not new
        :param model_parameters: model parameters object
        :param fitting_parameters: fitting parameters object
        :param kwargs: additional parameters
        """
        super().__init__(model_parameters, fitting_parameters, **kwargs)

        if 'model_id' not in kwargs:
            raise ModelException('Parameter "model_id" not found in additional parameter for VbmScaler object')

        self._model_id = kwargs['model_id']

        self._scaler_engine = MinMaxScaler()

        self._model_id = ''

        self._read_from_db()

    def fit(self, x: Optional[list[dict[str, Any]] | pd.DataFrame] = None,
            y: Optional[list[dict[str, Any]] | pd.DataFrame] = None) -> VbmScalerClass:
        """
        Saves engine parameters to scale data
        :param x: data to scale
        :param y: None
        :return: self scaling object
        """
        non_categorical_columns = [el for el in self._fitting_parameters.x_columns
                                   if el not in self._fitting_parameters.categorical_columns]

        self._scaler_engine.fit(x[non_categorical_columns])

        self._write_to_db()

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data after saving scaler parameters
        :param x: data before scaling
        :return: data after scaling
        """
        non_categorical_columns = [el for el in self._fitting_parameters.x_columns
                                   if el not in self._fitting_parameters.categorical_columns]

        result = x.copy()
        result[non_categorical_columns] = self._scaler_engine.transform(x[non_categorical_columns])

        return result

    def drop(self) -> None:
        """ For deleting current scaler from db """
        self._db_connector.delete_lines('scalers', {'model_id': self._model_id})

    def _write_to_db(self) -> None:
        """ For writing current scaler to db """
        line_to_db = {'model_id': self._model_id, 'engine': pickle.dumps(self._scaler_engine)}

        self._db_connector.set_line('scalers', line_to_db, {'model_id': self._model_id})

    def _read_from_db(self) -> None:
        """ For reading current scaler from db """
        line_from_db = self._db_connector.get_line('scalers', {'model_id': self._model_id})

        if line_from_db:
            self._scaler_engine = pickle.loads(line_from_db['engine'])
