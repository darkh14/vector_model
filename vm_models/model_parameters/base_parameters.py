""" Module contains base class for saving and getting fitting parameters of model """

from typing import Any, Optional
from dataclasses import dataclass, fields
from datetime import datetime

from vm_logging.exceptions import ModelException


@dataclass
class ModelParameters:

    service_name: str = ''

    name: str = ''
    type: str = ''
    data_filter: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.data_filter = {}

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        self._check_new_parameters(parameters)

        self.name = parameters['name']
        self.type = parameters['type']

        self.data_filter = parameters.get('filter')

    def _check_new_parameters(self, parameters: dict[str, Any], checking_names:Optional[list] = None) -> None:

        if not checking_names:
            checking_names = ['name', 'type']

        error_names = [el for el in checking_names if el not in parameters]

        if not parameters.get('name'):
            ModelException('Parameter(s) {} not found in model parameters'.format(', '.join("{}".format(error_names))))

    def get_data_filter_for_db(self) -> dict[str, Any]:
        return self.data_filter

    def get_all(self) -> dict[str, Any]:
        parameters = {
            'name': self.name,
            'type': self.type,
            'filter': self.data_filter
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
            'fitting_date': self.fitting_date,
            'fitting_start_date': self.fitting_start_date,
            'fitting_job_id': self.fitting_job_id,
            'fitting_job_pid': self.fitting_job_pid,
            'x_columns': self.x_columns,
            'y_columns': self.y_columns
        }

        return parameters

