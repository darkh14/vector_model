""" Contains model class, that provides working with models including
fitting and predicting
"""

from typing import Any, Callable,Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from vm_logging.exceptions import ModelException
from vm_models.model_parameters import base_parameters, get_model_parameters_class
from db_processing import get_connector as get_db_connector
from db_processing.connectors import base_connector
from ..data_transformers import get_transformer_class
from ..engines import get_engine_class
from ..model_types import DataTransformersTypes

__all__ = ['Model']


class Model:
    service_name: str = ''
    def __init__(self, model_id: str, db_path: str):

        self._id: str = model_id
        self._initialized: bool = False
        self._db_path: str = db_path

        self.parameters: base_parameters.ModelParameters = get_model_parameters_class()()
        self.fitting_parameters:base_parameters.FittingParameters = get_model_parameters_class(fitting=True)()

        self._scaler = get_transformer_class(DataTransformersTypes.SCALER)(self.parameters, self.fitting_parameters,
                                                                           self._db_path)
        self._engine = get_engine_class()(self.parameters, self.fitting_parameters, self._db_path)

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
        # self._initialized = False

        return 'model id {} is dropped'.format(self._id)

    def get_info(self) -> dict[str, Any]:
        model_info = {'id': self._id, 'initialized': self._initialized}

        model_info.update(self.parameters.get_all())
        model_info.update(self.fitting_parameters.get_all())

        return model_info

    def fit(self, fitting_parameters: dict[str, Any]) -> dict[str, Any]:

        self._check_before_fitting(fitting_parameters)

        self.fitting_parameters.set_start_fitting()
        self._write_to_db()

        try:
            result = self._fit_model(fitting_parameters['epochs'], fitting_parameters['filter'])
        except Exception as ex:
            self.fitting_parameters.set_error_fitting()
            raise ex

        self.fitting_parameters.set_end_fitting()
        self._write_to_db()

        return {'descr': result}

    def drop_fitting(self) -> str:

        self.fitting_parameters.set_drop_fitting()

        return 'Model "{}" id "{}" fitting is dropped'.format(self.parameters.name, self.id)

    def _fit_model(self, epochs: int, fitting_filter: Optional[dict[str, Any]],
                   fitting_parameters: Optional[dict[str, Any]]) -> Any:
        pipeline = self._get_fitting_pipeline(fitting_filter)
        data = pipeline.fit_transform(None)

        x, y = self._data_to_x_y(data)
        result = self._engine.fit(x, y, epochs, fitting_parameters)

        return result

    def _data_to_x_y(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return data[self.fitting_parameters.x_columns].to_numpy(), data[self.fitting_parameters.y_columns].to_numpy()

    def _check_before_fitting(self, fitting_parameters: dict[str, Any]):
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        if 'epochs' not in fitting_parameters:
            raise ModelException('Parameter "epochs" not found in fitting parameters')

    def _get_fitting_estimators(self, fitting_filter: Optional[dict[str, Any]] = None,
                              fitting_parameters: Optional[dict[str, Any]] = None) -> list[tuple[str, Any]]:

        estimator_types_list = [
                                DataTransformersTypes.READER,
                                DataTransformersTypes.CHECKER,
                                DataTransformersTypes.ROW_COLUMN_TRANSFORMER,
                                DataTransformersTypes.CATEGORICAL_ENCODER,
                                DataTransformersTypes.NAN_PROCESSOR
        ]

        estimators = []
        for estimator_type in estimator_types_list:
            estimator_class = get_transformer_class(estimator_type)
            estimator = estimator_class(self.parameters, self.fitting_parameters,self._db_path)

            estimators.append((estimator_type.value, estimator))

        estimators.append(('scaler', self._scaler))

        return estimators

    def _get_fitting_pipeline(self, fitting_filter: Optional[dict[str, Any]] = None,
                              fitting_parameters: Optional[dict[str, Any]] = None) -> Pipeline:

        estimators = self._get_fitting_estimators(fitting_filter, fitting_parameters)

        return Pipeline(estimators)

    def _write_to_db(self):
        model_to_db = {'id': self._id}
        model_to_db.update(self.parameters.get_all())
        model_to_db.update(self.fitting_parameters.get_all())

        self._db_connector.set_line('models', model_to_db, {'id': self._id})

    def _add_fields_to_write_to_db(self, model_to_db: dict[str, Any])-> None: ...

    def _read_from_db(self):
        model_from_db = self._db_connector.get_line('models', {'id': self._id})

        if model_from_db:
            self.parameters.set_all(model_from_db, without_processing=True)
            self.fitting_parameters.set_all(model_from_db, without_processing=True)

            self._initialized = True
        else:
            self._initialized = False

            self.parameters: base_parameters.ModelParameters = get_model_parameters_class()()
            self.fitting_parameters: base_parameters.FittingParameters = get_model_parameters_class(fitting=True)()

            self._scaler = get_transformer_class(DataTransformersTypes.SCALER)(self.parameters, self.fitting_parameters,
                                                                               self._db_path)
            self._engine = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def initialized(self) -> bool:
        return self._initialized


def get_additional_actions() -> dict[str, Callable]:
    return {}


