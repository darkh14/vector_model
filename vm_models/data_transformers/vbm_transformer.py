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

from vm_logging.exceptions import ModelException
from ..model_parameters.base_parameters import ModelParameters, FittingParameters
from .base_transformer import Reader, RowColumnTransformer, Checker, CategoricalEncoder, NanProcessor, Scaler, Shuffler
from data_processing.data_preprocessors import get_data_preprocessing_class
from id_generator import IdGenerator

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

            indicator_ids = list(x['indicator'].unique())

            model_indicators = self._model_parameters.x_indicators + self._model_parameters.y_indicators

            model_indicator_ids = [el['id'] for el in model_indicators]

            model_indicator_ids = set(model_indicator_ids)

            error_ids = []

            for el in model_indicator_ids:
                if el not in indicator_ids:
                    error_ids.append(el)

            if error_ids:
                error_names = [el['name'] for el in self._model_parameters.x_indicators +
                               self._model_parameters.y_indicators if el['id'] in error_ids]
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

        analytic_bound_ids = self._get_analytic_bound_ids()

        all_indicators = self._add_field_to_indicators(self._model_parameters.x_indicators, 'is_y', False)
        if self._fitting_mode:
            all_indicators = all_indicators + self._add_field_to_indicators(self._model_parameters.y_indicators,
                                                                            'is_y', True)

        need_to_update_columns_descr = self._fitting_mode and self._fitting_parameters.is_first_fitting()

        column_descriptions = list()

        for ind_parameters in all_indicators:

            ind_data = self._get_raw_data_by_indicator(raw_data, ind_parameters)

            ind_data = self._process_data_periods(ind_data, ind_parameters)

            analytic_keys = self._get_analytic_parameters_from_data(ind_data, ind_parameters, analytic_bound_ids)

            ind_parameters['analytic_keys'] = analytic_keys.copy()

            if not analytic_keys and not ind_parameters['use_analytics']:
                analytic_keys.append('')

            for analytic_key in analytic_keys:

                for value_name in ('sum', 'qty'):

                    if (value_name == 'sum' and not ind_parameters['use_sum']
                            or value_name == 'qty' and not ind_parameters['use_qty']):
                        continue

                    column_descr = self._get_column_description(ind_parameters['id'], analytic_key, value_name,
                                                        ind_parameters)

                    if need_to_update_columns_descr:
                        column_descriptions.append(column_descr)

                    self._check_append_column_names(column_descr['name'], x_columns, y_columns, ind_parameters)

                    an_data = self._get_raw_data_by_analytics(ind_data, analytic_key, value_name)
                    an_data.rename({value_name: column_descr['name']}, axis=1, inplace=True)

                    data_result = data_result.merge(an_data, on=['organisation', 'scenario', 'period'], how='left')

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

        if need_to_update_columns_descr:
            filter_names = {'name': {'$in': [el['name'] for el in column_descriptions]}}
            self._db_connector.set_lines('column_descriptions', column_descriptions, filter_names)

        return data_result

    def _get_analytic_bound_ids(self) -> dict[str, list[str]]:
        
        bound = {}

        for ind in self._model_parameters.x_indicators + self._model_parameters.y_indicators:
            if ind.get('analytics_bound'):
                bound[ind['id']] = ind['analytics_bound']

        return bound

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def _add_data_while_predicting(self, data_result: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds special fields to data after transforming while predicting
        :param data_result: data after transforming
        :param raw_data: data before transforming
        :return: data after fields added
        """

        return data_result

    @staticmethod
    def _add_field_to_indicators(indicators: list[dict[str, Any]], field_name: str,
                                 field_value: Any) -> list[dict[str, Any]]:

        for ind in indicators:
            ind[field_name] = field_value

        return indicators

    def _get_raw_data_from_x(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        transforms data list of dicts to pd.DataFrame
        :param x: input data (list of dicts)
        :return: data (pd.DataFrame)
        """
        raw_data = x

        if 'periodicity' not in raw_data.columns:
            scenarios_description = self._db_connector.get_lines('scenarios')

            raw_data['periodicity'] = raw_data['scenario'].apply(lambda arg: [el['periodicity']
                                                                              for el in scenarios_description
                                                                              if el['id'] == arg][0])

        return raw_data

    @staticmethod
    def _get_grouped_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Groups raw data to use in transforming
        :param raw_data: all raw data before transforming
        :return: grouped data
        """
        return raw_data[['organisation', 'scenario',
                        'period']].groupby(by=['organisation', 'scenario', 'period'], as_index=False).min()

    # noinspection PyMethodMayBeStatic
    def _get_raw_data_by_indicator(self, data: pd.DataFrame, indicator_parameters: dict[str, Any]) -> pd.DataFrame:
        """
        Gets raw_data where indicator == required indicator
        :param data: all data
        :param indicator_parameters: parameters of required indicator
        :return: data with one indicator
        """
        fields = ['organisation', 'scenario', 'period', 'periodicity', 'analytic_key', 'sum', 'qty']
        ind_data = data[fields].loc[data['indicator'] == indicator_parameters['id']]

        return ind_data

    # noinspection PyMethodMayBeStatic
    def _get_raw_data_by_analytics(self, data: pd.DataFrame, analytic_key: str, value_name: str) -> pd.DataFrame:
        """
        Gets raw indicator data by analytic key
        :param data: data with one indicator
        :param analytic_key: id of required analytic key
        :return: data with one indicator and one analytic key
        """
        fields = ['organisation', 'scenario', 'period', 'periodicity', 'analytic_key', 'sum', 'qty']
        if analytic_key:
            an_data = data[fields].loc[data['analytic_key'] == analytic_key]
        else:
            an_data = data[fields]

        fields = ['organisation', 'scenario', 'period']
        an_data = an_data[fields + [value_name]].groupby(fields, as_index=False).sum()

        return an_data

    def _get_analytic_parameters_from_data(self, data: pd.DataFrame,
                                    indicator_parameters: dict[str, Any],
                                    analytic_bound_ids: dict[str, list[str]]) -> list[str]:
        """
        Gets all analytic keys and ids in data (for predicting and secondary fitting)
        :param data: data with one indicator
        :param indicator_parameters:
        :return: analytic keys and ids
        """
        if indicator_parameters['use_analytics']:

            if self._fitting_mode and self._fitting_parameters.is_first_fitting():
                keys = self._get_analytic_parameters_for_new_fitting(data, indicator_parameters, analytic_bound_ids)
            else:
                keys = indicator_parameters.get('analytic_keys') or []
        else:
            keys = []

        return keys

    # noinspection PyMethodMayBeStatic
    def _get_analytic_parameters_for_new_fitting(self, data: pd.DataFrame, indicator_parameters: dict[str, Any],
                                    analytic_bound_ids: dict[str, list[str]]) -> list[str]:
        """
        Gets analytic keys and id for new fitting
        :param data: data with one indicator
        :param indicator_parameters: parameters of current indicator
        :return: all analytic keys and ids found in data
        """

        result = list(data['analytic_key'].unique())

        use_bound = indicator_parameters.get('analytics_bound')
        ind_bound_ids = [el for el in indicator_parameters['analytics_bound']] if use_bound else []
        use_inv_bound = analytic_bound_ids.get(indicator_parameters['id'])
        inv_ind_bound_ids = analytic_bound_ids.get(indicator_parameters['id']) if use_inv_bound else []

        if use_bound:
            result = [el for el in result if el in ind_bound_ids]
        elif use_inv_bound:
            result = [el for el in result if el not in inv_ind_bound_ids]

        return result

    @staticmethod
    def _is_y_indicator(indicator_parameters: dict[str, Any]) -> bool:
        """
        Returns True if current indicator is in y indicators, else False
        :param indicator_parameters: parameters of current indicators
        :return: True if current indicator is in y indicators, else False
        """

        return indicator_parameters['is_y']

    # noinspection PyMethodMayBeStatic
    def _get_column_description(self, indicator: str, analytic_key: str, value_name: str,
                         indicator_parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Form column name from indicator id, analytic key id and period parameters
        :param indicator: id of current indicator
        :param analytic_key: id of current analytic key
        :param value_name: name of value ex. "value" or "value_quantity"
        :param indicator_parameters: parameters of current indicator
        :return: column name
        """
        result = {'indicator': indicator,
                  'use_analytics': indicator_parameters['use_analytics'],
                  'analytic_key': analytic_key,
                  'value_name': value_name,
                  'period_shift': indicator_parameters['period_shift'],
                  'period_number': indicator_parameters['period_number'],
                  'period_accumulation': indicator_parameters['period_accumulation']}

        if indicator_parameters['use_analytics']:
            result_name = 'ind_{}_val_{}_an_{}'.format(indicator, value_name, analytic_key)
        else:
            result_name = 'ind_{}_val_{}'.format(indicator, value_name)

        if indicator_parameters.get('period_shift'):
            if indicator_parameters['period_shift'] < 0:
                result_name += '_p_m{}'.format(-indicator_parameters['period_shift'])
            else:
                result_name += '_p_p{}'.format(indicator_parameters['period_shift'])
        elif indicator_parameters.get('period_number'):
            result_name += '_p_n{}'.format(indicator_parameters['period_number'])
        elif indicator_parameters.get('period_accumulation'):
            result_name += '_p_a'

        result['name'] = IdGenerator.get_id_by_name(result_name)

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
        """
        super().__init__(model_parameters, fitting_parameters, **kwargs)
        self._fields: list = model_parameters.categorical_features or []

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Adds encoding fields
        :param x: data before encoding
        :return: data after encoding
        """
        added_fields = []

        # noinspection PyTypeChecker
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

    def fit(self, x: Optional[list[dict[str, Any]] | pd.DataFrame] = None,
            y: Optional[list[dict[str, Any]] | pd.DataFrame] = None) -> VbmScalerClass:
        """
        Saves engine parameters to scale data
        :param x: data to scale
        :param y: None
        :return: self scaling object
        """

        self._fitting_mode = True

        if self._new_scaler:

            non_categorical_columns = [el for el in self._fitting_parameters.x_columns +
                                       self._fitting_parameters.y_columns
                                       if el not in self._fitting_parameters.categorical_columns]

            data = x[non_categorical_columns]

            self._scaler_engine.fit(data)

            self._write_to_db()

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms data after saving scaler parameters
        :param x: data before scaling
        :return: data after scaling
        """

        result = x.copy()

        if not self._fitting_mode:
            result[self._fitting_parameters.y_columns] = 0

        non_categorical_columns = [el for el in self._fitting_parameters.x_columns + self._fitting_parameters.y_columns
                                   if el not in self._fitting_parameters.categorical_columns]

        result[non_categorical_columns] = self._scaler_engine.transform(result[non_categorical_columns])

        if not self._fitting_mode:
            result = result.drop(self._fitting_parameters.y_columns, axis=1)

        return result

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transforms data after predicting to get real (unscaled) result
        :param x: data before unscaling
        :return: data after unscaling
        """

        result = x.copy()

        non_categorical_columns = [el for el in self._fitting_parameters.x_columns + self._fitting_parameters.y_columns
                                   if el not in self._fitting_parameters.categorical_columns]

        result[non_categorical_columns] = self._scaler_engine.inverse_transform(result[non_categorical_columns])

        return result

    def _get_scaler_engine(self) -> object:
        """
        For getting scaler object of right type
        :return: inner scaler object
        """
        return MinMaxScaler()


class VbmShuffler(Shuffler):
    service_name: ClassVar[str] = 'vbm'
