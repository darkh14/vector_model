""" Contains model class, that provides working with models including
fitting and predicting
"""

from typing import Any, Callable,Optional
import psutil

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from vm_logging.exceptions import ModelException
from vm_models.model_parameters import base_parameters, get_model_parameters_class
from db_processing import get_connector as get_db_connector
from db_processing.connectors import base_connector
from ..data_transformers import get_transformer_class, base_transformer
from ..engines import get_engine_class, base_engine
from ..model_types import DataTransformersTypes
from vm_background_jobs.controller import set_background_job_interrupted

__all__ = ['Model']


class Model:
    service_name: str = ''
    def __init__(self, model_id: str, db_path: str):

        self._id: str = model_id
        self._initialized: bool = False
        self._db_path: str = db_path

        self.parameters: base_parameters.ModelParameters = get_model_parameters_class()()
        self.fitting_parameters:base_parameters.FittingParameters = get_model_parameters_class(fitting=True)()

        self._engine: Optional[base_engine.BaseEngine] = None

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
        self._initialized = False

        scaler = get_transformer_class(DataTransformersTypes.SCALER)(self.parameters,
                                                                     self.fitting_parameters,
                                                                     self._db_path, model_id=self._id)
        scaler.drop()

        if self._engine:
            self._engine.drop()

        return 'model id {} is dropped'.format(self._id)

    def get_info(self) -> dict[str, Any]:
        model_info = {'id': self._id, 'initialized': self._initialized}

        model_info.update(self.parameters.get_all())
        model_info.update(self.fitting_parameters.get_all())

        model_info['filter'] = model_info['filter'].get_value_as_json_serializable()

        return model_info

    def fit(self, fitting_parameters: dict[str, Any]) -> dict[str, Any]:

        self._check_before_fitting(fitting_parameters)

        self.fitting_parameters.set_start_fitting(fitting_parameters)
        self._write_to_db()

        try:
            result = self._fit_model(fitting_parameters['epochs'], fitting_parameters)
        except Exception as ex:
            self.fitting_parameters.set_error_fitting()
            self._write_to_db()
            raise ex

        if not self.fitting_parameters.fitting_is_error:

            self.fitting_parameters.metrics = self._engine.metrics

            self.fitting_parameters.set_end_fitting()
            self._write_to_db()

        return result

    def predict(self, x: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._check_before_predicting()

        result = self._predict_model(x)

        return result

    def drop_fitting(self) -> str:

        self._interrupt_fitting_job()

        self.fitting_parameters.set_drop_fitting()
        self._write_to_db()

        return 'Model "{}" id "{}" fitting is dropped'.format(self.parameters.name, self.id)

    def _interrupt_fitting_job(self):
        if self.fitting_parameters.fitting_is_started:
            set_background_job_interrupted(self.fitting_parameters.fitting_job_id, self._db_path)

    def _fit_model(self, epochs: int, fitting_parameters: Optional[dict[str, Any]] = None) -> Any:

        pipeline = self._get_model_pipeline(for_predicting=False, fitting_parameters=fitting_parameters)
        data = pipeline.fit_transform(None)

        x, y = self._data_to_x_y(data)
        self._engine = get_engine_class(self.parameters.type)(self.parameters,
                                                              self.fitting_parameters, self._db_path, self._id)
        result = self._engine.fit(x, y, epochs, fitting_parameters)

        return result

    def _data_to_x_y(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return data[self.fitting_parameters.x_columns].to_numpy(), data[self.fitting_parameters.y_columns].to_numpy()

    def _data_to_x(self, data: pd.DataFrame) -> np.ndarray:
        return data[self.fitting_parameters.x_columns].to_numpy()

    def _y_to_data(self, y: np.ndarray) ->  pd.DataFrame:
        return pd.DataFrame(y, columns=self.fitting_parameters.y_columns)

    def _check_before_fitting(self, fitting_parameters: dict[str, Any]):
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        if 'epochs' not in fitting_parameters:
            raise ModelException('Parameter "epochs" not found in fitting parameters')

        if self.fitting_parameters.fitting_is_started:
            raise ModelException('Another fitting is started yet. Wait for end of fitting')

    def _check_before_predicting(self):

        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        if not self.fitting_parameters.is_fit:
            raise ModelException(('Model "{}" id "{}" is not fit. ' +
                                 'Fit model before predicting').format(self.parameters.name, self._id))

    def _get_model_estimators(self, for_predicting: bool = False,
                              fitting_parameters: Optional[dict[str, Any]] = None) -> list[tuple[str, Any]]:

        estimator_types_list = [
                                DataTransformersTypes.READER,
                                DataTransformersTypes.CHECKER,
                                DataTransformersTypes.ROW_COLUMN_TRANSFORMER,
                                DataTransformersTypes.CATEGORICAL_ENCODER,
                                DataTransformersTypes.NAN_PROCESSOR,
                                DataTransformersTypes.SCALER
        ]

        estimators = []
        for estimator_type in estimator_types_list:
            estimator = self._get_estimator(estimator_type, fitting_parameters)
            estimators.append((estimator_type.value, estimator))

        return estimators

    def _get_estimator(self, transformer_type: DataTransformersTypes,
                             fitting_parameters: Optional[dict[str, Any]] = None) -> base_transformer.BaseTransformer:

        estimator_class = get_transformer_class(transformer_type)
        estimator = estimator_class(self.parameters, self.fitting_parameters, self._db_path, model_id=self._id)
        if fitting_parameters:
            estimator.set_additional_parameters(fitting_parameters)

        return estimator

    def _get_model_pipeline(self, for_predicting: bool = False,
                            fitting_parameters: Optional[dict[str, Any]] = None) -> Pipeline:

        estimators = self._get_model_estimators(for_predicting, fitting_parameters)

        return Pipeline(estimators)

    def _write_to_db(self):
        model_to_db = {'id': self._id}
        model_to_db.update(self.parameters.get_all())
        model_to_db.update(self.fitting_parameters.get_all())

        model_to_db['filter'] = model_to_db['filter'].get_value_as_model_parameter()

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

            self._engine = None

    def _predict_model(self, x: list[dict[str, Any]]) -> list[dict[str, Any]]:

        pipeline = self._get_model_pipeline(for_predicting=True)
        data = pipeline.transform(x)

        x = self._data_to_x(data)
        self._engine = get_engine_class(self.parameters.type)(self.parameters,
                                                              self.fitting_parameters, self._db_path, self._id)
        y_pred = self._engine.predict(x)

        result_data = self._y_to_data(y_pred)
        return result_data.to_dict('records')

    @property
    def id(self) -> str:
        return self._id

    @property
    def initialized(self) -> bool:
        return self._initialized


def get_additional_actions() -> dict[str, Callable]:
    return {}


