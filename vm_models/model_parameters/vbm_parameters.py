""" Module contains VBM class for saving and getting fitting parameters of model """

from typing import Any, Optional
from dataclasses import dataclass, fields
from copy import deepcopy
from datetime import datetime
import os

from vm_logging.exceptions import ModelException
from .base_parameters import ModelParameters, FittingParameters
from id_generator import IdGenerator

__all__ = ['VbmModelParameters', 'VbmFittingParameters']

@dataclass
class VbmModelParameters(ModelParameters):

    service_name: str = 'vbm'
    _x_indicators: list[dict[str, Any]] = None
    _y_indicators: list[dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        self._x_indicators = []
        self._y_indicators = []

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:

        super().set_all(parameters)

        self._check_new_parameters(parameters)

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields =  [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_parameters = [el[1:] if el.startswith('_') else el for el in self_fields]

        for count, par_name in enumerate(self_parameters):
            name = self_fields[count] if without_processing else par_name
            if par_name in parameters:
                setattr(self, name, parameters[par_name])

    def get_all(self) -> dict[str, Any]:
        result = super().get_all()

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields = [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_fields = [el[1:] if el.startswith('_') else el for el in self_fields]

        parameters_to_add = {name: getattr(self, name) for name in self_fields}

        result.update(parameters_to_add)

        return result

    def _check_new_parameters(self, parameters: dict[str, Any], checking_names: Optional[list] = None) -> None:

        super()._check_new_parameters(parameters, checking_names)
        if not checking_names:
            super()._check_new_parameters(parameters, ['x_indicators', 'y_indicators'])

    @property
    def x_indicators(self):
        return self._x_indicators

    @property
    def y_indicators(self):
        return self._y_indicators

    @x_indicators.setter
    def x_indicators(self, value):
        x_indicators = deepcopy(value)

        for el in x_indicators:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._x_indicators = x_indicators

    @y_indicators.setter
    def y_indicators(self, value):
        y_indicators = deepcopy(value)

        for el in y_indicators:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._y_indicators = y_indicators


@dataclass
class VbmFittingParameters(FittingParameters):
    service_name = 'vbm'

    _x_analytics: Optional[list[dict[str, Any]]] = None
    _y_analytics: Optional[list[dict[str, Any]]] = None

    _x_analytic_keys: Optional[list[dict[str, Any]]] = None
    _y_analytic_keys: Optional[list[dict[str, Any]]] = None

    fi_is_calculated: bool = False
    fi_calculation_is_started: bool = False
    fi_calculation_is_error: bool = False
    fi_calculation_date: Optional[datetime] = None
    fi_calculation_start_date: Optional[datetime] = None
    fi_calculation_error_date: Optional[datetime] = None

    fi_calculation_error_text: str = ''

    fi_calculation_job_id: str = ''
    fi_calculation_job_pid: int = 0
    feature_importances: Optional[dict[str, Any]] = None

    def __post_init__(self):

        super().__post_init__()

        self._x_analytics = []
        self._y_analytics = []

        self._x_analytic_keys = []
        self._y_analytic_keys = []

        self.feature_importances = {}

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        super().set_all(parameters, without_processing=without_processing)

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields =  [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_parameters = [el[1:] if el.startswith('_') else el for el in self_fields]

        fi_parameters  = ['fi_is_calculated', 'fi_calculation_is_started', 'fi_calculation_is_error',
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
        self.fi_calculation_is_error = parameters.get('fi_calculation_is_error') or False

        if without_processing:
            self.fi_calculation_date = parameters['fi_calculation_date']
            self.fi_calculation_start_date = parameters['fi_calculation_start_date']
            self.fi_calculation_error_date = parameters['fi_calculation_error_date']
        else:
            self.fi_calculation_date = (datetime.strptime(parameters['fi_calculation_date'], '%d.%m.%Y %H:%M:%S')
                                 if parameters.get('fi_calculation_date') else None)
            self.fi_calculation_start_date = (datetime.strptime(parameters['fi_calculation_start_date'], '%d.%m.%Y %H:%M:%S')
                                       if parameters.get('fi_calculation_start_date') else None)
            self.fi_calculation_error_date = (datetime.strptime(parameters['fi_calculation_error_date'], '%d.%m.%Y %H:%M:%S')
                                       if parameters.get('fi_calculation_error_date') else None)

        self.fi_calculation_error_text = parameters.get('fi_calculation_error_text') or ''

        self.fi_calculation_job_id = parameters.get('fi_calculation_job_id') or ''
        self.fi_calculation_job_pid = parameters.get('fi_calculation_job_pid') or 0

        self.feature_importances = parameters.get('feature_importances') or {}

    def get_all(self, for_db: bool = False) -> dict[str, Any]:
        result = super().get_all(for_db=for_db)

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields = [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_fields = [el[1:] if el.startswith('_') else el for el in self_fields]

        parameters_to_add = {name: getattr(self, name) for name in self_fields}

        result.update(parameters_to_add)

        fi_parameters = {
            'fi_is_calculated': self.fi_is_calculated,
            'fi_calculation_is_started': self.fi_calculation_is_started,
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
            fi_parameters['fi_calculation_date'] = (fi_parameters['fi_calculation_date'].strftime('%d.%m.%Y %H:%M:%S')
                                                    if fi_parameters['fi_calculation_date'] else None)
            fi_parameters['fi_calculation_start_date'] = (fi_parameters['fi_calculation_start_date'].strftime('%d.%m.%Y %H:%M:%S')
                                                    if fi_parameters['fi_calculation_start_date'] else None)
            fi_parameters['fi_calculation_error_date']  = (fi_parameters['fi_calculation_error_date'].strftime('%d.%m.%Y %H:%M:%S')
                                                    if fi_parameters['fi_calculation_error_date'] else None)

        result.update(fi_parameters)

        return result

    def set_start_fitting(self, fitting_parameters: dict[str, Any]) -> None:
        super().set_start_fitting(fitting_parameters)

        if self.fi_is_calculated or self.fi_calculation_is_started:
            self.set_drop_fi_calculation()

    def set_drop_fitting(self):
        super().set_drop_fitting()

        self._x_analytics = []
        self._y_analytics = []

        self._x_analytic_keys = []
        self._y_analytic_keys = []

        if self.fi_is_calculated or self.fi_calculation_is_started:
            self.set_drop_fi_calculation()

    def set_error_fitting(self, error_text: str = '') -> None:
        super().set_error_fitting(error_text)

        if self._first_fitting:
            self._x_analytics = []
            self._y_analytics = []

            self._x_analytic_keys = []
            self._y_analytic_keys = []

    def set_start_fi_calculation(self, fi_parameters: dict[str, Any]) -> None:

        self.fi_is_calculated = False
        self.fi_calculation_is_started = True
        self.fi_calculation_is_error = False

        self.fi_calculation_date = None
        self.fi_calculation_start_date = datetime.utcnow()
        self.fi_calculation_error_date = None

        self.fi_calculation_error_text = ''

        self.fi_calculation_job_pid = os.getpid()

        if fi_parameters.get('job_id'):
            self.fitting_job_id = fi_parameters['job_id']

    def set_end_fi_calculation(self):

        if not self.fi_calculation_is_started:
            raise ModelException('Can not finish fi calculation. Fi calculation is not started. ' +
                                 'Start fi calculation before')

        self.fi_is_calculated = True
        self.fi_calculation_is_started = False
        self.fi_calculation_is_error = False

        self.fi_calculation_date = datetime.utcnow()
        self.fi_calculation_error_date = None

    def set_error_fi_calculation(self, error_text):

        self.fi_is_calculated = False
        self.fi_calculation_is_started = False
        self.fi_calculation_is_error = True

        self.fi_calculation_date = None
        self.fi_calculation_error_date = datetime.utcnow()

        self.fi_calculation_error_text = error_text

        self.feature_importances = {}

    def set_drop_fi_calculation(self):

        if not self.fi_is_calculated and not self.fi_calculation_is_started and not self.fi_calculation_is_error:
            raise ModelException('Can not drop fi calculation. FI is not calculated')

        self.fi_is_calculated = False
        self.fi_calculation_is_started = False
        self.fi_calculation_is_error = False

        self.fi_calculation_date = None
        self.fi_calculation_start_date = None
        self.fi_calculation_error_date = None

        self.fi_calculation_error_text = ''

        self.fi_calculation_job_id = ''
        self.fi_calculation_job_pid = 0

        self.feature_importances = {}

    @property
    def x_analytics(self):
        return self._x_analytics

    @property
    def y_analytics(self):
        return self._y_analytics

    @x_analytics.setter
    def x_analytics(self, value):
        x_analytics = deepcopy(value)

        for el in x_analytics:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._x_analytics = x_analytics

    @y_analytics.setter
    def y_analytics(self, value):
        y_analytics = deepcopy(value)

        for el in y_analytics:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._y_analytics = y_analytics

    @property
    def x_analytic_keys(self):
        return self._x_analytic_keys

    @property
    def y_analytic_keys(self):
        return self._y_analytic_keys

    @x_analytic_keys.setter
    def x_analytic_keys(self, value):
        x_analytic_keys = deepcopy(value)

        for el in x_analytic_keys:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_list_of_dict_short_id(el['analytics'])

        self._x_analytic_keys = x_analytic_keys

    @y_analytic_keys.setter
    def y_analytic_keys(self, value):
        y_analytic_keys = deepcopy(value)

        for el in y_analytic_keys:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_list_of_dict_short_id(el['analytics'])

        self._y_analytic_keys = y_analytic_keys