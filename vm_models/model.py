""" Contains model class, that provides working with models including
fitting and predicting
"""

from typing import Any, Optional
from datetime import datetime

from vm_logging.exceptions import ModelException
from .fitting_parameters import FittingParameters
from db_processing import get_connector as get_db_connector
from db_processing.connectors import base_connector

__all__ = ['Model', 'get_model']


class Model:
    service_name = ''
    def __init__(self, model_id: str, db_path: str):

        self._id: str = model_id
        self._initialized: bool = False
        self._name: str = ''

        self._filter: dict[str, Any] = dict()

        self._is_fit: bool = False
        self._fitting_is_started: bool = False
        self._fitting_start_date: Optional[datetime] = None
        self._fitting_job_id: str = ''
        self._fitting_job_pid: int = 0
        self._fitting_parameters: Optional[FittingParameters] = None
        self._engine = None

        self._db_connector: base_connector.Connector = get_db_connector(db_path)

        self._read_from_db()

    def initialize(self, model_parameters: dict[str, Any]) -> dict[str, Any]:

        if self._initialized:
            raise ModelException('Model "{}" id - "{}" is always initialized'.format(self._name, self._id))

        required_parameter_names = ['name']
        error_names = [el for el in required_parameter_names if el not in model_parameters]

        if error_names:
            error_text_list = ['"{}"'.format(el) for el in error_names]
            raise ModelException('Parameter(s) {} is not found in model parameters'.format(','.join(error_text_list)))

        model_filter = model_parameters.get('filter') or {}

        self._name = model_parameters['name']
        self._filter = model_filter

        self._write_to_db()

        self._initialized = True

        return self.get_info()

    def drop(self) -> dict[str, Any]:
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        self._db_connector.delete_lines('models', {'id': self._id})

        return self.get_info()

    def get_info(self) -> dict[str, Any]:

        model_info = {'id': self._id,
                      'name': self._name,
                      'filter': self._filter,
                      'is_fit': self._is_fit,
                      'fitting_is_started': self._fitting_is_started,
                      'fitting_start_date': self._fitting_start_date.strftime('%d.%m.%Y %H:%M:%S')
                      if self._fitting_start_date else None,
                      'fitting_job_id': self._fitting_job_pid,
                      'fitting_job_pid': self._fitting_job_pid
                       }

        self._add_fields_to_model_info(model_info)

        return model_info

    def _add_fields_to_model_info(self, model_info: dict[str, Any]) -> None: ...

    def _write_to_db(self):
        model_to_db = {'id': self._id,
                       'name': self._name,
                       'filter': self._filter,
                       'is_fit': self._is_fit,
                       'fitting_is_started': self._fitting_is_started,
                       'fitting_start_date': self._fitting_start_date,
                       'fitting_job_id': self._fitting_job_pid,
                       'fitting_job_pid': self._fitting_job_pid
                       }

        self._db_connector.set_line('models', model_to_db, {'id': self._id})

    def _add_fields_to_write_to_db(self, model_to_db: dict[str, Any])-> None: ...

    def _read_from_db(self):
        model_from_db = self._db_connector.get_line('models', {'id': self._id})

        if model_from_db:
            self._name = model_from_db['name']
            self._filter = model_from_db['filter']

            self._is_fit = model_from_db['is_fit']
            self._fitting_is_started = model_from_db['fitting_is_started']
            self._fitting_start_date = model_from_db['fitting_start_date']
            self._fitting_job_id = model_from_db['fitting_job_id']
            self._fitting_job_pid = model_from_db['fitting_job_pid']
            # self._fitting_parameters: Optional[FittingParameters] = model_from_db['is_fit']
            # self._engine = None

            self._initialized = True

    def _read_additional_fields_from_db(self, model_from_db: dict[str, Any]) -> None: ...

    @property
    def id(self) -> str:
        return self._id


def get_model(model_id: str, db_path: str) -> Model:
    return Model(model_id, db_path)
