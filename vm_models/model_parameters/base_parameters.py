""" Module contains base class for saving and getting fitting parameters of model """

from typing import Any, Optional, Type
from dataclasses import dataclass, fields
from datetime import datetime
from ..model_filters import base_filter, get_fitting_filter_class

from vm_logging.exceptions import ModelException


@dataclass
class ModelParameters:

    service_name: str = ''

    name: str = ''
    type: str = ''
    data_filter: Optional[base_filter.FittingFilter] = None

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        self._check_new_parameters(parameters)

        self.name = parameters['name']
        self.type = parameters['type']

        self.data_filter = get_fitting_filter_class()(parameters.get('filter'), for_model=True)

    def _check_new_parameters(self, parameters: dict[str, Any], checking_names:Optional[list] = None) -> None:

        if not checking_names:
            checking_names = ['name', 'type']

        error_names = [el for el in checking_names if el not in parameters]

        if not parameters.get('name'):
            ModelException('Parameter(s) {} not found in model parameters'.format(', '.join("{}".format(error_names))))

    def get_data_filter_for_db(self) -> dict[str, Any]:
        return self.data_filter.get_value_for_db()

    def get_all(self) -> dict[str, Any]:
        parameters = {
            'name': self.name,
            'type': self.type,
            'filter': self.data_filter.get_value()
        }

        return parameters


@dataclass
class FittingParameters:

    service_name: str = ''

    is_fit: bool = False
    fitting_is_started: bool = False
    fitting_is_error: bool = False
    fitting_date: Optional[datetime] = None
    fitting_start_date: Optional[datetime] = None
    fitting_error_date: Optional[datetime] = None

    x_columns: Optional[list[str]] = None
    y_columns: Optional[list[str]] = None

    fitting_job_id: str = ''
    fitting_job_pid: int = 0


    def __post_init__(self):
        self.x_columns = []
        self.y_columns = []

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:

        par_fields = [el.name for el in fields(self) if el.name not in ['service_name']]

        for field_name in par_fields:
            if field_name in parameters:
                setattr(self, field_name, parameters[field_name])

    def get_all(self) -> dict[str, Any]:
        parameters = {
            'is_fit': self.is_fit,
            'fitting_is_started': self.fitting_is_started,
            'fitting_is_error': self.fitting_is_error,
            'fitting_date': self.fitting_date.strftime('%d.%m.%Y %H:%M:%S') if self.fitting_date else None,
            'fitting_start_date': self.fitting_start_date.strftime('%d.%m.%Y %H:%M:%S') if self.fitting_date else None,
            'fitting_error_date': self.fitting_start_date.strftime('%d.%m.%Y %H:%M:%S') if self.fitting_date else None,
            'fitting_job_id': self.fitting_job_id,
            'fitting_job_pid': self.fitting_job_pid,
            'x_columns': self.x_columns,
            'y_columns': self.y_columns
        }

        return parameters

    def set_start_fitting(self):
        self.is_fit = False
        self.fitting_is_started = True
        self.fitting_is_error = False

        self.fitting_start_date = datetime.utcnow()
        self.fitting_date = None
        self.fitting_error_date = None

    def set_end_fitting(self):

        if not self.fitting_is_started:
            raise ModelException('Can not finish fitting. Fitting is not started. Start fitting before')

        self.is_fit = True
        self.fitting_is_started = False
        self.fitting_is_error = False

        self.fitting_date = datetime.utcnow()
        self.fitting_error_date = None

    def set_drop_fitting(self):

        if not self.is_fit:
            raise ModelException('Can not drop fitting. Model is not fit')

        self.is_fit = False
        self.fitting_is_started = False
        self.fitting_is_error = False

        self.fitting_start_date = None
        self.fitting_date = None
        self.fitting_error_date = None

    def set_error_fitting(self):

        self.is_fit = False
        self.fitting_is_started = False
        self.fitting_is_error = True

        self.fitting_date = None
        self.fitting_error_date = datetime.utcnow()