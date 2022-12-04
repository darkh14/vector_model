""" Contains model class, that provides working with models including
fitting and predicting
"""

from typing import Any, Callable

from vm_logging.exceptions import ModelException
from vm_models.model_parameters import base_parameters, get_model_parameters_class
from db_processing import get_connector as get_db_connector
from db_processing.connectors import base_connector

__all__ = ['Model']


class Model:
    service_name: str = ''
    def __init__(self, model_id: str, db_path: str):

        self._id: str = model_id
        self._initialized: bool = False

        self.parameters: base_parameters.ModelParameters = get_model_parameters_class()()
        self.fitting_parameters:base_parameters.FittingParameters = get_model_parameters_class(fitting=True)()

        self._engine = None

        self._db_connector: base_connector.Connector = get_db_connector(db_path)

        self._read_from_db()

    def initialize(self, model_parameters: dict[str, Any]) -> dict[str, Any]:

        if self._initialized:
            raise ModelException('Model "{}" id - "{}" is always initialized'.format(self.parameters.name, self._id))

        self.parameters.set_all(model_parameters)
        self.fitting_parameters.set_all(model_parameters)

        self._write_to_db()

        self._initialized = True

        return self.get_info()

    def drop(self) -> str:
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        self._db_connector.delete_lines('models', {'id': self._id})

        return 'model id {} is dropped'.format(self._id)

    def get_info(self) -> dict[str, Any]:

        model_info = {'id': self._id}

        model_info.update(self.parameters.get_all())

        return model_info

    def _write_to_db(self):
        model_to_db = {'id': self._id}
        model_to_db.update(self.parameters.get_all())
        model_to_db.update(self.fitting_parameters.get_all())

        self._db_connector.set_line('models', model_to_db, {'id': self._id})

    def _add_fields_to_write_to_db(self, model_to_db: dict[str, Any])-> None: ...

    def _read_from_db(self):
        model_from_db = self._db_connector.get_line('models', {'id': self._id})

        if model_from_db:
            self.parameters.set_all(model_from_db)
            self.fitting_parameters.set_all(model_from_db)

            self._initialized = True

    @property
    def id(self) -> str:
        return self._id


def get_additional_actions() -> dict[str, Callable]:
    return {}


