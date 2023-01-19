""" Module contains base class for saving and getting fitting parameters of model
    Classes:
        ModelParameters - base dataclass for storing, saving and getting model parameters
        FittingParameters - base dataclass for storing, saving and getting fitting parameters
"""

from typing import Any, Optional, ClassVar
from dataclasses import dataclass
from datetime import datetime
import os

from ..model_filters import base_filter, get_fitting_filter_class
from vm_logging.exceptions import ModelException, ParametersFormatError


@dataclass
class ModelParameters:
    """ Base dataclass for storing, saving and getting model parameters
        Methods:
            set_all - to set all input parameters in object
            get_all - to get all parameters
            get_data_filter_for_db - to get data filter for using as sa db filter
            _check_new_parameters - checks parameters
    """
    service_name: ClassVar[str] = ''

    name: str = ''
    type: str = ''
    data_filter: Optional[base_filter.FittingFilter] = None

    def __post_init__(self) -> None:
        """
        Defines data filter object
        """
        self.data_filter = get_fitting_filter_class()({})

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        """
        For setting all parameters, defined in "parameters" parameter
        :param parameters: input parameters to set
        :param without_processing: no need to convert parameters if True
        """
        self._check_new_parameters(parameters)

        self.name = parameters['name']
        self.type = parameters['type']

        self.data_filter = get_fitting_filter_class()(parameters.get('filter'))

    def _check_new_parameters(self, parameters: dict[str, Any]) -> None:
        """
        For checking parameters. Raises ModelException if checking is failed
        :param parameters: parameters to check
        """

        match parameters:
            case {'id': str(model_id), 'name': str(name), 'type': str(model_type)}:
                pass
            case _:
                raise ParametersFormatError('Wrong request parameters format. Check "model" parameter')

    def get_data_filter_for_db(self) -> dict[str, Any]:
        """
        Gets data filter value to use it as a db filter
        :return: db filter value
        """
        return self.data_filter.get_value_as_db_filter()

    def get_all(self) -> dict[str, Any]:
        """
        For getting values of all parameters
        :return: dict of values of all parameters
        """
        parameters = {
            'name': self.name,
            'type': self.type,
            'filter': self.data_filter
        }

        return parameters


@dataclass
class FittingParameters:
    """ Base dataclass for storing, saving and getting fitting parameters
        Methods:
            set_all - to set all input parameters in object
            get_all - to get all parameters
            set_start_fitting - to set statuses and other parameters before starting fitting
            set_end_fitting - to set statuses and other parameters after finishing fitting
            set_drop_fitting - to set statuses and other parameters when dropping fitting
            set_error_fitting - to set statuses and other parameters when error is while fitting
            is_first_fitting - returns True if it is first fitting after dropping or when model is new
    """
    service_name: ClassVar[str] = ''

    is_fit: bool = False
    fitting_is_started: bool = False
    fitting_is_error: bool = False
    fitting_date: Optional[datetime] = None
    fitting_start_date: Optional[datetime] = None
    fitting_error_date: Optional[datetime] = None

    fitting_error_text: str = ''

    x_columns: Optional[list[str]] = None
    y_columns: Optional[list[str]] = None

    categorical_columns: Optional[list[str]] = None

    fitting_job_id: str = ''
    fitting_job_pid: int = 0

    metrics: Optional[dict[str, Any]] = None

    _first_fitting: bool = False

    def __post_init__(self) -> None:
        """
        Converts x, y categorical columns values to empty lists, metrics value to empty dict
        """
        self.x_columns = []
        self.y_columns = []

        self.categorical_columns = []
        self.metrics = {}

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        """
        For setting all parameters, defined in "parameters" parameter
        :param parameters: input parameters to set
        :param without_processing: no need to convert parameters if True
        """
        self.is_fit = parameters.get('is_fit') or False
        self.fitting_is_started = parameters.get('fitting_is_started') or False
        self.fitting_is_error = parameters.get('fitting_is_error') or False

        if without_processing:
            self.fitting_date = parameters['fitting_date']
            self.fitting_start_date = parameters['fitting_start_date']
            self.fitting_error_date = parameters['fitting_error_date']
        else:
            self.fitting_date = (datetime.strptime(parameters['fitting_date'], '%d.%m.%Y %H:%M:%S')
                                 if parameters.get('fitting_date') else None)
            self.fitting_start_date = (datetime.strptime(parameters['fitting_start_date'], '%d.%m.%Y %H:%M:%S')
                                       if parameters.get('fitting_start_date') else None)
            self.fitting_error_date = (datetime.strptime(parameters['fitting_error_date'], '%d.%m.%Y %H:%M:%S')
                                       if parameters.get('fitting_error_date') else None)

        self.fitting_error_text = parameters.get('fitting_error_text') or ''

        self.fitting_job_id = parameters.get('fitting_job_id') or ''
        self.fitting_job_pid = parameters.get('fitting_job_pid') or 0

        self.x_columns = parameters.get('x_columns') or []
        self.y_columns = parameters.get('y_columns') or []

        self.categorical_columns = parameters.get('categorical_columns') or []

        self.metrics = parameters.get('metrics') or {}

    def get_all(self, for_db: bool = False) -> dict[str, Any]:
        """
        For getting values of all parameters
        :param for_db: True if we need to get parameters for writing them to db
        :return: dict of values of all parameters
        """
        parameters = {
            'is_fit': self.is_fit,
            'fitting_is_started': self.fitting_is_started,
            'fitting_is_error': self.fitting_is_error,

            'fitting_date': self.fitting_date,
            'fitting_start_date': self.fitting_start_date,
            'fitting_error_date': self.fitting_error_date,

            'fitting_error_text': self.fitting_error_text,
            'fitting_job_id': self.fitting_job_id,
            'fitting_job_pid': self.fitting_job_pid,

            'x_columns': self.x_columns,
            'y_columns': self.y_columns,
            'categorical_columns': self.categorical_columns,
            'metrics': self.metrics
        }

        if not for_db:
            parameters['fitting_date'] = (parameters['fitting_date'].strftime('%d.%m.%Y %H:%M:%S')
                                    if parameters.get('fitting_date') else None)
            parameters['fitting_start_date'] = (parameters['fitting_start_date'].strftime('%d.%m.%Y %H:%M:%S')
                                    if parameters.get('fitting_start_date') else None)
            parameters['fitting_error_date'] = (parameters['fitting_error_date'].strftime('%d.%m.%Y %H:%M:%S')
                                    if parameters.get('fitting_error_date') else None)

        return parameters

    def set_start_fitting(self, fitting_parameters: dict[str, Any]) -> None:
        """
        For setting statuses and other parameters before starting fitting
        :param fitting_parameters: parameters of fitting, which will be started
        """
        self.is_fit = False
        self.fitting_is_started = True
        self.fitting_is_error = False

        self.fitting_start_date = datetime.utcnow()
        self.fitting_date = None
        self.fitting_error_date = None

        self.fitting_error_text = ''

        self.fitting_job_pid = os.getpid()

        if fitting_parameters.get('job_id'):
            self.fitting_job_id = fitting_parameters['job_id']

        self._first_fitting = not self.x_columns and not self.y_columns

    def set_end_fitting(self) -> None:
        """
        For setting statuses and other parameters after finishing fitting
        """
        if not self.fitting_is_started:
            raise ModelException('Can not finish fitting. Fitting is not started. Start fitting before')

        self.is_fit = True
        self.fitting_is_started = False
        self.fitting_is_error = False

        self.fitting_date = datetime.utcnow()
        self.fitting_error_date = None

        self._first_fitting = False

    def set_drop_fitting(self) -> None:
        """
        For setting statuses and other parameters when dropping fitting
        """
        if not self.is_fit and not self.fitting_is_started and not self.fitting_is_error:
            raise ModelException('Can not drop fitting. Model is not fit')

        self.is_fit = False
        self.fitting_is_started = False
        self.fitting_is_error = False

        self.fitting_start_date = None
        self.fitting_date = None
        self.fitting_error_date = None

        self.fitting_error_text = ''

        self.x_columns = []
        self.y_columns = []

        self.categorical_columns = []

        self._first_fitting = True

    def set_error_fitting(self, error_text: str = '') -> None:
        """
        For setting statuses and other parameters when error is while fitting
        :param error_text: text of fitting error
        """
        self.is_fit = False
        self.fitting_is_started = False
        self.fitting_is_error = True

        self.fitting_date = None
        self.fitting_error_date = datetime.utcnow()

        self.fitting_error_text = error_text

        if self._first_fitting:
            self.x_columns = []
            self.y_columns = []

            self.categorical_columns = []

    def is_first_fitting(self):
        """
        Returns True if it is first fitting after dropping or when model is new else False
        :return: bool result
        """
        return self._first_fitting
