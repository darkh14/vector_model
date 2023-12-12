""" Module contains base class for saving and getting fitting parameters of model
    Classes:
        ModelParameters - base dataclass for storing, saving and getting model parameters
        FittingParameters - base dataclass for storing, saving and getting fitting parameters
"""

from typing import Any, Optional, ClassVar, Type
from dataclasses import dataclass
from datetime import datetime
import os
import sys, inspect

from ..model_filters import base_filter, get_fitting_filter_class
from .. import model_types
from vm_logging.exceptions import ModelException


@dataclass
class BaseModelStructure:
    type: ClassVar[Optional[model_types.ModelTypes]] = None

    def set_all(self, parameters: dict[str, Any]) -> None:

        self._check_parameters(parameters)

        for attr_name in self.__dict__:
            if attr_name in parameters:
                setattr(self, attr_name, parameters[attr_name])

    def get_all(self):

        return self.__dict__

    def _check_parameters(self, parameters: dict[str, Any]) -> None:
        ...


@dataclass
class PolynomialRegressionStructure(BaseModelStructure):
    type: ClassVar[Optional[model_types.ModelTypes]] = model_types.ModelTypes.PolynomialRegression

    power: int = 2

    def _check_parameters(self, parameters: dict[str, Any]) -> None:

        if parameters.get('power'):
            if not isinstance(parameters['power'], int):
                raise ModelException('Parameter "power" must be int from 2 to 5')
            elif parameters['power'] < 2 or parameters['power'] > 5:
                raise ModelException('Parameter "power" must be int from 2 to 5')


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
    type: model_types.ModelTypes = model_types.ModelTypes.NeuralNetwork
    data_filter: Optional[base_filter.FittingFilter] = None

    model_structure: Optional[BaseModelStructure] = None

    def __post_init__(self) -> None:
        """
        Defines data filter object
        """
        self.data_filter = get_fitting_filter_class()({})

    def set_all(self, parameters: dict[str, Any]) -> None:
        """
        For setting all parameters, defined in "parameters" parameter
        :param parameters: input parameters to set
        """

        self.name = parameters['name']
        self.type = parameters['type']

        self.data_filter = get_fitting_filter_class()(parameters.get('filter'))

        if not self.model_structure:
            model_structure_class = self._get_model_structure_class()

            if model_structure_class:
                self.model_structure = model_structure_class()
                self.model_structure.set_all(parameters)
        else:
            self.model_structure.set_all(parameters)

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

        if self.model_structure:
            parameters.update(self.model_structure.get_all())

        return parameters

    def _get_model_structure_class(self) -> Optional[Type[BaseModelStructure]]:

        result = None

        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj) and obj in BaseModelStructure.__subclasses__() and obj.type == self.type:
                result = obj

        return result

    def __getattr__(self, item):
        if self.model_structure and item in self.model_structure.__dict__:
            return getattr(self.model_structure, item)
        else:
            raise AttributeError("{} object has no attribute {}".format(self.__name__, item))


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

    fitting_status: model_types.FittingStatuses = model_types.FittingStatuses.NotFit

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

    def set_all(self, parameters: dict[str, Any]) -> None:
        """
        For setting all parameters, defined in "parameters" parameter
        :param parameters: input parameters to set
        """
        self.fitting_status = parameters.get('fitting_status') or model_types.FittingStatuses.NotFit

        self.fitting_date = parameters.get('fitting_date')
        self.fitting_start_date = parameters.get('fitting_start_date')
        self.fitting_error_date = parameters.get('fitting_error_date')

        self.fitting_error_text = parameters.get('fitting_error_text', '')

        self.fitting_job_id = parameters.get('fitting_job_id', '')
        self.fitting_job_pid = parameters.get('fitting_job_pid', 0)

        self.x_columns = parameters.get('x_columns', [])
        self.y_columns = parameters.get('y_columns', [])

        self.categorical_columns = parameters.get('categorical_columns', [])

        self.metrics = parameters.get('metrics', {})

    def get_all(self) -> dict[str, Any]:
        """
        For getting values of all parameters
        :return: dict of values of all parameters
        """
        parameters = {
            'fitting_status': self.fitting_status,

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

        return parameters

    def set_pre_start_fitting(self, job_id: str = '') -> None:
        """
        For setting statuses and other parameters before starting fitting
        :param job_id: id of ob if fitting is background
        """
        self.fitting_status = model_types.FittingStatuses.PreStarted

        self.fitting_start_date = datetime.utcnow()
        self.fitting_date = None
        self.fitting_error_date = None

        self.fitting_error_text = ''

        self.fitting_job_pid = os.getpid()

        self.metrics = {}

        if job_id:
            self.fitting_job_id = job_id

        self._first_fitting = not self.x_columns and not self.y_columns

    def set_start_fitting(self, job_id: str = '') -> None:
        """
        For setting statuses and other parameters before starting fitting
        :param job_id: id of job if fitting in background
        """
        self.fitting_status = model_types.FittingStatuses.Started

        self.fitting_start_date = datetime.utcnow()
        self.fitting_date = None
        self.fitting_error_date = None

        self.fitting_error_text = ''

        self.fitting_job_pid = os.getpid()

        self.metrics = {}

        self._first_fitting = not self.x_columns and not self.y_columns

        if job_id:
            self.fitting_job_id = job_id

    def set_end_fitting(self) -> None:
        """
        For setting statuses and other parameters after finishing fitting
        """
        if self.fitting_status != model_types.FittingStatuses.Started:
            raise ModelException('Can not finish fitting. Fitting is not started. Start fitting before')

        self.fitting_status = model_types.FittingStatuses.Fit

        self.fitting_date = datetime.utcnow()
        self.fitting_error_date = None

        self._first_fitting = False

    def set_drop_fitting(self, model_id='') -> None:
        """
        For setting statuses and other parameters when dropping fitting
        """
        if self.fitting_status not in (model_types.FittingStatuses.Fit,
                                       model_types.FittingStatuses.Started,
                                       model_types.FittingStatuses.PreStarted):
            raise ModelException('Can not drop fitting. Model is not fit')

        self.fitting_status = model_types.FittingStatuses.NotFit

        self.fitting_start_date = None
        self.fitting_date = None
        self.fitting_error_date = None

        self.fitting_error_text = ''

        self.fitting_job_pid = 0
        self.fitting_job_id = ''

        self.x_columns = []
        self.y_columns = []

        self.categorical_columns = []

        self.metrics = {}

        self._first_fitting = True

    def set_error_fitting(self, error_text: str = '') -> None:
        """
        For setting statuses and other parameters when error is while fitting
        :param error_text: text of fitting error
        """
        self.fitting_status = model_types.FittingStatuses.Error

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
