""" VBM (Vector budget model)
    Module contains VBM classes for saving and getting fitting parameters of model
    Classes:
        VbmModelParameters - for model parameters
        VbmFittingParameters - for fitting parameters
"""

from typing import Any, Optional, ClassVar
from dataclasses import dataclass, fields
from copy import deepcopy
from datetime import datetime
import os

from vm_logging.exceptions import ModelException, ParametersFormatError
from .base_parameters import ModelParameters, FittingParameters
from data_processing.data_preprocessors import get_data_preprocessing_class

from id_generator import IdGenerator

__all__ = ['VbmModelParameters', 'VbmFittingParameters']


@dataclass
class VbmModelParameters(ModelParameters):
    """ Dataclass for storing, saving and getting model parameters. Inherited by ModelParameters class
        Methods:
            set_all - to set all input parameters in object
            get_all - to get all parameters
            _check_new_parameters - checks parameters
        Properties:
            x_indicators
            y_indicators
    """
    service_name: ClassVar[str] = 'vbm'

    _x_indicators: list[dict[str, Any]] = None
    _y_indicators: list[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """
        Defines x, y indicators as empty lists
        """
        super().__post_init__()
        self._x_indicators = []
        self._y_indicators = []

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        """
        For setting all parameters, defined in "parameters" parameter
        :param parameters: input parameters to set
        :param without_processing: no need to convert parameters if True
        """
        super().set_all(parameters, without_processing)

        self._check_new_parameters(parameters)

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields = [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_parameters = [el[1:] if el.startswith('_') else el for el in self_fields]

        for count, par_name in enumerate(self_parameters):
            name = self_fields[count] if without_processing else par_name
            if par_name in parameters:
                setattr(self, name, parameters[par_name])

    def get_all(self) -> dict[str, Any]:
        """
        For getting values of all parameters
        :return: dict of values of all parameters
        """
        result = super().get_all()

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields = [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_fields = [el[1:] if el.startswith('_') else el for el in self_fields]

        parameters_to_add = {name: getattr(self, name) for name in self_fields}

        result.update(parameters_to_add)

        return result

    def _check_new_parameters(self, parameters: dict[str, Any]) -> None:
        """
        For checking parameters. Raises ModelException if checking is failed
        :param parameters: parameters to check
        """

        super()._check_new_parameters(parameters)

        match parameters:
            case {'x_indicators': list(x_indicators), 'y_indicators': list(y_indicators)}:
                for _ind in x_indicators:
                    match _ind:
                        case {'id': str(),
                              'name': str(),
                              'use_analytics': bool(),
                              'period_shift': int(),
                              'period_number': int(),
                              'period_accumulation': bool(),
                              'values': list(values)}:
                            for _el in values:
                                if not isinstance(_el, str):
                                    raise ParametersFormatError('Wrong request parameters format. '
                                                                'Check "x_indicators" parameter')
                        case _:
                            raise ParametersFormatError('Wrong request parameters format. '
                                                        'Check "x_indicators" parameter')

                for _ind in y_indicators:
                    match _ind:
                        case {'id': str(),
                              'name': str(),
                              'use_analytics': bool(),
                              'values': list(values)}:
                            for _el in values:
                                if not isinstance(_el, str):
                                    raise ParametersFormatError('Wrong request parameters format. '
                                                                'Check "y_indicators" parameter')
                        case _:
                            raise ParametersFormatError('Wrong request parameters format. '
                                                        'Check "y_indicators" parameter')

            case _:
                raise ParametersFormatError('Wrong request parameters format. Check "model" parameter')

    @staticmethod
    def _add_short_ids_to_analytic_bounds(analytic_bounds: list[list[dict[str, Any]]]) \
            -> list[dict[str, list[str, Any]]]:

        data_processor = get_data_preprocessing_class()()

        bounds = []

        for initial_bound in analytic_bounds:
            bound_el = {}
            # noinspection PyUnresolvedReferences
            analytics = data_processor.add_short_id_to_analytics(initial_bound)
            # noinspection PyUnresolvedReferences
            bound_el['short_id'] = data_processor.get_short_id_for_analytics(analytics)
            bound_el['analytics'] = analytics

            bounds.append(bound_el)

        return bounds

    @property
    def x_indicators(self) -> list[dict[str, Any]]:
        """
        Property x_indicators getter. Returns self._x_indicators value
        :return: property value
        """
        return self._x_indicators

    @property
    def y_indicators(self):
        """
        Property y_indicators getter. Returns self._y_indicators value
        :return: property value
        """
        return self._y_indicators

    @x_indicators.setter
    def x_indicators(self, value) -> None:
        """
        Property x_indicators setter. Sets self._x_indicators value.
        :param value: value to set
        """
        x_indicators = deepcopy(value)

        for el in x_indicators:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

            if 'analytic_bounds' in el:
                el['analytic_bounds'] = self._add_short_ids_to_analytic_bounds(el['analytic_bounds'])

        self._x_indicators = x_indicators

    @y_indicators.setter
    def y_indicators(self, value):
        """
        Property y_indicators setter. Sets self._y_indicators value.
        :param value: value to set
        """
        y_indicators = deepcopy(value)

        for el in y_indicators:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

            if 'analytics_bound' in el:
                el['analytics_bound'] = self._add_short_ids_to_analytic_bounds(el['analytics_bound'])

        self._y_indicators = y_indicators


@dataclass
class VbmFittingParameters(FittingParameters):
    """ Dataclass for storing, saving and getting fitting parameters
        Methods:
            set_all - to set all input parameters in object
            get_all - to get all parameters
            set_start_fitting - to set statuses and other parameters before starting fitting
            set_drop_fitting - to set statuses and other parameters when dropping fitting
            set_error_fitting - to set statuses and other parameters when error is while fitting
            set_start_fi_calculation - to set statuses and other parameters before starting fi calculation
            set_end_fi_calculation - to set statuses and other parameters after finishing fi calculation
            set_error_fi_calculation - to set statuses and other parameters when error is while fi calculating
            set_drop_fi_calculation - to set statuses and other parameters when dropping fi calculation
        Properties:
            x_analytics
            y_analytics
            x_analytic_keys
            y_analytic_keys
     """
    service_name: ClassVar[str] = 'vbm'

    _x_analytics: Optional[list[dict[str, Any]]] = None
    _y_analytics: Optional[list[dict[str, Any]]] = None

    _x_analytic_keys: Optional[list[dict[str, Any]]] = None
    _y_analytic_keys: Optional[list[dict[str, Any]]] = None

    fi_is_calculated: bool = False
    fi_calculation_is_started: bool = False
    fi_calculation_is_pre_started: bool = False
    fi_calculation_is_error: bool = False
    fi_calculation_date: Optional[datetime] = None
    fi_calculation_start_date: Optional[datetime] = None
    fi_calculation_error_date: Optional[datetime] = None

    fi_calculation_error_text: str = ''

    fi_calculation_job_id: str = ''
    fi_calculation_job_pid: int = 0
    feature_importances: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        """
        Converts _x_analytics, _y_analytics, _x_analytic_keys, _y_analytic_keys to empty lists, feature_importances to
        empty dict
        """
        super().__post_init__()

        self._x_analytics = []
        self._y_analytics = []

        self._x_analytic_keys = []
        self._y_analytic_keys = []

        self.feature_importances = {}

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        """
        For setting all parameters, defined in "parameters" parameter
        :param parameters: input parameters to set
        :param without_processing: no need to convert parameters if True
        """
        super().set_all(parameters, without_processing=without_processing)

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields = [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_parameters = [el[1:] if el.startswith('_') else el for el in self_fields]

        fi_parameters = ['fi_is_calculated', 'fi_calculation_is_started', 'fi_calculation_is_error',
                         'fi_calculation_date', 'fi_calculation_start_date', 'fi_calculation_error_date',
                         'fi_calculation_error_text', 'fi_calculation_job_id', 'fi_calculation_job_pid',
                         'feature_importances']
        self_parameters = [el for el in self_parameters if el not in fi_parameters]

        for count, par_name in enumerate(self_parameters):
            name = self_fields[count] if without_processing else par_name
            if par_name in parameters:
                setattr(self, name, parameters[par_name])

        self.fi_is_calculated = parameters.get('fi_is_calculated') or False
        self.fi_calculation_is_started = parameters.get('fi_calculation_is_started') or False
        self.fi_calculation_is_pre_started = parameters.get('fi_calculation_is_pre_started') or False
        self.fi_calculation_is_error = parameters.get('fi_calculation_is_error') or False

        if without_processing:
            self.fi_calculation_date = parameters['fi_calculation_date']
            self.fi_calculation_start_date = parameters['fi_calculation_start_date']
            self.fi_calculation_error_date = parameters['fi_calculation_error_date']
        else:
            self.fi_calculation_date = (datetime.strptime(parameters['fi_calculation_date'], '%d.%m.%Y %H:%M:%S')
                                 if parameters.get('fi_calculation_date') else None)
            self.fi_calculation_start_date = (datetime.strptime(parameters['fi_calculation_start_date'],
                                 '%d.%m.%Y %H:%M:%S') if parameters.get('fi_calculation_start_date') else None)
            self.fi_calculation_error_date = (datetime.strptime(parameters['fi_calculation_error_date'],
                                 '%d.%m.%Y %H:%M:%S') if parameters.get('fi_calculation_error_date') else None)

        self.fi_calculation_error_text = parameters.get('fi_calculation_error_text') or ''

        self.fi_calculation_job_id = parameters.get('fi_calculation_job_id') or ''
        self.fi_calculation_job_pid = parameters.get('fi_calculation_job_pid') or 0

        self.feature_importances = parameters.get('feature_importances') or {}

    def get_all(self, for_db: bool = False) -> dict[str, Any]:
        """
        For getting values of all parameters
        :param for_db: True if we need to get parameters for writing them to db
        :return: dict of values of all parameters
        """
        result = super().get_all(for_db=for_db)

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields = [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_fields = [el[1:] if el.startswith('_') else el for el in self_fields]

        parameters_to_add = {name: getattr(self, name) for name in self_fields}

        result.update(parameters_to_add)

        fi_parameters = {
            'fi_is_calculated': self.fi_is_calculated,
            'fi_calculation_is_started': self.fi_calculation_is_started,
            'fi_calculation_is_pre_started': self.fi_calculation_is_pre_started,
            'fi_calculation_is_error': self.fi_calculation_is_error,

            'fi_calculation_date': self.fi_calculation_date,
            'fi_calculation_start_date': self.fi_calculation_start_date,
            'fi_calculation_error_date': self.fi_calculation_error_date,

            'fi_calculation_error_text': self.fi_calculation_error_text,
            'fi_calculation_job_id': self.fi_calculation_job_id,
            'fi_calculation_job_pid': self.fi_calculation_job_pid,
            'feature_importances': self.feature_importances
        }

        if not for_db:
            fi_parameters['fi_calculation_date'] = (fi_parameters[
                                                        'fi_calculation_date'].strftime('%d.%m.%Y %H:%M:%S')
                                                    if fi_parameters.get('fi_calculation_date') else None)
            fi_parameters['fi_calculation_start_date'] = (fi_parameters[
                                                        'fi_calculation_start_date'].strftime('%d.%m.%Y %H:%M:%S')
                                                    if fi_parameters.get('fi_calculation_start_date') else None)
            fi_parameters['fi_calculation_error_date'] = (fi_parameters[
                                                        'fi_calculation_error_date'].strftime('%d.%m.%Y %H:%M:%S')
                                                    if fi_parameters.get('fi_calculation_error_date') else None)

        result.update(fi_parameters)

        return result

    def set_start_fitting(self, fitting_parameters: dict[str, Any]) -> None:
        """
        For setting statuses and other parameters before starting fitting
        :param fitting_parameters: parameters of fitting, which will be started
        """
        super().set_start_fitting(fitting_parameters)

        if self.fi_is_calculated or self.fi_calculation_is_started:
            self.set_drop_fi_calculation()

    def set_drop_fitting(self, model_id='') -> None:
        """
        For setting statuses and other parameters when dropping fitting
        """
        super().set_drop_fitting()

        self._x_analytics = []
        self._y_analytics = []

        self._x_analytic_keys = []
        self._y_analytic_keys = []

        if self.fi_is_calculated or self.fi_calculation_is_started or self.fi_calculation_is_error:
            self.set_drop_fi_calculation()

    def set_error_fitting(self, error_text: str = '') -> None:
        """
        For setting statuses and other parameters when error is while fitting
        :param error_text: text of fitting error
        """
        super().set_error_fitting(error_text)

        if self._first_fitting:
            self._x_analytics = []
            self._y_analytics = []

            self._x_analytic_keys = []
            self._y_analytic_keys = []

    def set_start_fi_calculation(self, fi_parameters: dict[str, Any]) -> None:
        """
        For setting statuses and other parameters before starting fi calculation
        :param fi_parameters: parameters of fi calculation, which will be started
        """
        self.fi_is_calculated = False
        self.fi_calculation_is_started = True
        self.fi_calculation_is_pre_started = False
        self.fi_calculation_is_error = False

        self.fi_calculation_date = None
        self.fi_calculation_start_date = datetime.utcnow()
        self.fi_calculation_error_date = None

        self.fi_calculation_error_text = ''

        self.fi_calculation_job_pid = os.getpid()

        if fi_parameters.get('job_id'):
            self.fi_calculation_job_id = fi_parameters['job_id']

    def set_pre_start_fi_calculation(self, fi_parameters: dict[str, Any]) -> None:
        """
        For setting statuses and other parameters before starting fi calculation
        :param fi_parameters: parameters of fi calculation, which will be started
        """
        self.fi_is_calculated = False
        self.fi_calculation_is_started = False
        self.fi_calculation_is_pre_started = True

        self.fi_calculation_is_error = False

        self.fi_calculation_date = None
        self.fi_calculation_start_date = datetime.utcnow()
        self.fi_calculation_error_date = None

        self.fi_calculation_error_text = ''

        self.fi_calculation_job_pid = os.getpid()

        if fi_parameters.get('job_id'):
            self.fi_calculation_job_id = fi_parameters['job_id']

    def set_end_fi_calculation(self) -> None:
        """
        For setting statuses and other parameters after finishing fi calculation
        """
        if not self.fi_calculation_is_started:
            raise ModelException('Can not finish fi calculation. Fi calculation is not started. ' +
                                 'Start fi calculation before')

        self.fi_is_calculated = True
        self.fi_calculation_is_started = False
        self.fi_calculation_is_pre_started = False
        self.fi_calculation_is_error = False

        self.fi_calculation_date = datetime.utcnow()
        self.fi_calculation_error_date = None

    def set_error_fi_calculation(self, error_text) -> None:
        """
        For setting statuses and other parameters when error is while fi calculating
        :param error_text: text of fitting error
        """
        self.fi_is_calculated = False
        self.fi_calculation_is_started = False
        self.fi_calculation_is_pre_started = False
        self.fi_calculation_is_error = True

        self.fi_calculation_date = None
        self.fi_calculation_error_date = datetime.utcnow()

        self.fi_calculation_error_text = error_text

        self.feature_importances = {}

    def set_drop_fi_calculation(self) -> None:
        """
        For setting statuses and other parameters when dropping fi calculation
        """
        if not self.fi_is_calculated and not self.fi_calculation_is_started and not self.fi_calculation_is_error:
            raise ModelException('Can not drop fi calculation. FI is not calculated')

        self.fi_is_calculated = False
        self.fi_calculation_is_started = False
        self.fi_calculation_is_pre_started = False
        self.fi_calculation_is_error = False

        self.fi_calculation_date = None
        self.fi_calculation_start_date = None
        self.fi_calculation_error_date = None

        self.fi_calculation_error_text = ''

        self.fi_calculation_job_id = ''
        self.fi_calculation_job_pid = 0

        self.feature_importances = {}

    @property
    def x_analytics(self) -> list[dict[str, Any]]:
        """
        Property x_analytics getter. Returns self._x_analytics value
        :return: property value
        """
        return self._x_analytics

    @property
    def y_analytics(self):
        """
        Property y_analytics getter. Returns self._y_analytics value
        :return: property value
        """
        return self._y_analytics

    @x_analytics.setter
    def x_analytics(self, value: list[dict[str, Any]]) -> None:
        """
        Property x_analytics setter. Sets self._x_analytics value.
        :param value: value to set
        """
        x_analytics = deepcopy(value)

        for el in x_analytics:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._x_analytics = x_analytics

    @y_analytics.setter
    def y_analytics(self, value: list[dict[str, Any]]) -> None:
        """
        Property y_analytics setter. Sets self._y_analytics value.
        :param value: value to set
        """
        y_analytics = deepcopy(value)

        for el in y_analytics:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._y_analytics = y_analytics

    @property
    def x_analytic_keys(self) -> list[dict[str, Any]]:
        """
        Property x_analytic_keys getter. Returns self.x_analytic_keys value
        :return: property value
        """
        return self._x_analytic_keys

    @property
    def y_analytic_keys(self) -> list[dict[str, Any]]:
        """
        Property y_analytic_keys getter. Returns self.y_analytic_keys value
        :return: property value
        """
        return self._y_analytic_keys

    @x_analytic_keys.setter
    def x_analytic_keys(self, value: list[dict[str, Any]]) -> None:
        """
        Property x_analytic_keys setter. Sets self.x_analytic_keys value.
        :param value: value to set
        """
        x_analytic_keys = deepcopy(value)

        for el in x_analytic_keys:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_list_of_dict_short_id(el['analytics'])

        self._x_analytic_keys = x_analytic_keys

    @y_analytic_keys.setter
    def y_analytic_keys(self, value: list[dict[str, Any]]) -> None:
        """
        Property y_analytic_keys setter. Sets self.y_analytic_keys value.
        :param value: value to set
        """
        y_analytic_keys = deepcopy(value)

        for el in y_analytic_keys:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_list_of_dict_short_id(el['analytics'])

        self._y_analytic_keys = y_analytic_keys
