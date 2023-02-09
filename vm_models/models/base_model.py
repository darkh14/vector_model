""" Contains model class, that provides working with models including
    fitting and predicting
    Classes:
        Model - base model class, provides fitting and predicting. Saves itself to db
"""

from typing import Any, Callable, Optional, ClassVar

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
from data_processing.loading_engines import get_engine_class as get_loading_engine_class

__all__ = ['Model']


class Model:
    """ Base model class, provides fitting and predicting. Saves itself to db
        Methods:
            initialize - to initialize model in db
            drop - to delete model from db
            get_info - gets all model info (statuses, dates, metrics etc.)
            fit - to fit model
            predict - to predict data using this model
            drop_fitting - to drop model fitting
        Properties:
            id
            initialized
    """
    service_name: ClassVar[str] = ''

    def __init__(self, model_id: str) -> None:
        """
        Defines all model parameters and read model from db if it is necessary
        :param model_id: id of model
        """
        self._id: str = model_id
        self._initialized: bool = False

        self.parameters: base_parameters.ModelParameters = get_model_parameters_class()()
        self.fitting_parameters: base_parameters.FittingParameters = get_model_parameters_class(fitting=True)()

        self._engine: Optional[base_engine.BaseEngine] = None

        self._db_connector: base_connector.Connector = get_db_connector()

        # noinspection PyTypeChecker
        self._scaler: base_transformer.Scaler = get_transformer_class(DataTransformersTypes.SCALER,
                                                                      self.parameters.type)(self.parameters,
                                                                                           self.fitting_parameters,
                                                                                           model_id=self._id)
        self._read_from_db()

    def initialize(self, model_parameters: dict[str, Any]) -> dict[str, Any]:
        """
        For initializing model in db
        :param model_parameters:
        :return: model info dict
        """
        if self._initialized:
            raise ModelException('Model "{}" id - "{}" is always initialized'.format(self.parameters.name, self._id))

        self.parameters.set_all(model_parameters)
        self.fitting_parameters.set_all(model_parameters)

        self._write_to_db()

        self._initialized = True

        return self.get_info()

    def drop(self) -> str:
        """ Deletes model from db. sets initializing = False
        :return: result of dropping
        """
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        self._db_connector.delete_lines('models', {'id': self._id})
        self._initialized = False

        if self._engine:
            self._engine.drop()

        self._scaler.drop()

        return 'model id {} is dropped'.format(self._id)

    def get_info(self) -> dict[str, Any]:
        """
        Gest model information - statuses, dates, metrics etc.
        :return: model info dict
        """
        model_info = {'id': self._id, 'initialized': self._initialized}

        model_info.update(self.parameters.get_all())
        model_info.update(self.fitting_parameters.get_all())

        model_info['filter'] = model_info['filter'].get_value_as_json_serializable()

        return model_info

    def fit(self, fitting_parameters: dict[str, Any]) -> dict[str, Any]:
        """
        For fitting model
        :param fitting_parameters: parameters of fitting (ex. epochs)
        :return: fitting history
        """
        self._check_before_fitting(fitting_parameters)

        self.fitting_parameters.set_start_fitting(fitting_parameters)
        self._write_to_db()

        try:
            result = self._fit_model(fitting_parameters['epochs'], fitting_parameters)
        except Exception as ex:
            self.fitting_parameters.set_error_fitting(str(ex))
            self._write_to_db()
            raise ex

        if not self.fitting_parameters.fitting_is_error:

            self.fitting_parameters.set_end_fitting()
            self._write_to_db(write_scaler=True)

        return result

    def predict(self, x: list[dict[str, Any]]) -> dict[str, Any]:
        """
        For predicting data with model
        :param x: input data for predicting
        :return: predicted output data
        """
        self._check_before_predicting(x)

        result = self._predict_model(x)

        return result

    def drop_fitting(self) -> str:
        """
        Deletes results of fitting from db
        :return: resul of dropping
        """
        self._interrupt_fitting_job()

        if self._engine:
            self._engine.drop()

        self._scaler.drop()

        self.fitting_parameters.set_drop_fitting()
        self._write_to_db()

        return 'Model "{}" id "{}" fitting is dropped'.format(self.parameters.name, self.id)

    def _interrupt_fitting_job(self) -> None:
        """
        Interrupts fitting job process when fitting is launched in background mode
        """
        if self.fitting_parameters.fitting_is_started:
            set_background_job_interrupted(self.fitting_parameters.fitting_job_id)

    def _fit_model(self, epochs: int, fitting_parameters: Optional[dict[str, Any]] = None) -> Any:
        """
        For fitting model after checking, and preparing parameters
        :param epochs: number of epochs for fitting
        :param fitting_parameters: additional fitting parameters
        :return: fitting history
        """
        pipeline = self._get_model_pipeline(for_predicting=False, fitting_parameters=fitting_parameters)
        data = pipeline.fit_transform(None)

        x, y = self._data_to_x_y(data)
        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)
        self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number,
                                                              self.fitting_parameters.is_first_fitting())
        result = self._engine.fit(x, y, epochs, fitting_parameters)

        self._scaler = pipeline.named_steps['scaler']

        y_pred = self._engine.predict(x)

        data_predicted = data.copy()
        data_predicted[self.fitting_parameters.y_columns] = y_pred

        data = self._scaler.inverse_transform(data)
        data_predicted = self._scaler.inverse_transform(data_predicted)

        y = data[self.fitting_parameters.y_columns].to_numpy()
        y_pred = data_predicted[self.fitting_parameters.y_columns].to_numpy()

        self.fitting_parameters.metrics = self._get_metrics(y, y_pred)

        return result

    def _data_to_x_y(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts read pd data to np array x and y for fitting
        :param data: input read data
        :return: x, y data tuple
        """
        return data[self.fitting_parameters.x_columns].to_numpy(), data[self.fitting_parameters.y_columns].to_numpy()

    def _data_to_x(self, data: pd.DataFrame) -> np.ndarray:
        """
        Converts input pd data to np array x
        :param data: input data
        :return: x data for predicting
        """
        return data[self.fitting_parameters.x_columns].to_numpy()

    def _y_to_data(self, y: np.ndarray, x_data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts output np array y to output pd. data
        :param y: y output predicted np array
        :param x_data: input x data
        :return: predicted output pd data
        """

        x_data[self.fitting_parameters.y_columns] = y

        numeric_columns = self.fitting_parameters.x_columns + self.fitting_parameters.y_columns

        x_data[numeric_columns] = self._scaler.inverse_transform(x_data[numeric_columns])

        return x_data[self.fitting_parameters.y_columns]

    def _check_before_fitting(self, fitting_parameters: dict[str, Any]) -> None:
        """
        For checking statuses and other parameters before fitting. Raises ModelException if checking is failed
        :param fitting_parameters: parameters to check
        """
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        if 'epochs' not in fitting_parameters:
            raise ModelException('Parameter "epochs" not found in fitting parameters')

        if self.fitting_parameters.fitting_is_started:
            raise ModelException('Another fitting is started yet. Wait for end of fitting')

    def _check_before_predicting(self, inputs: list[dict[str, Any]]) -> None:
        """
        For checking statuses and other parameters before predicting. Raises ModelException if checking is failed
        """
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        if not self.fitting_parameters.is_fit:
            raise ModelException(('Model "{}" id "{}" is not fit. ' +
                                 'Fit model before predicting').format(self.parameters.name, self._id))

        loading_engine = get_loading_engine_class()()

        loading_engine.check_data(inputs)

    def _get_model_estimators(self, for_predicting: bool = False,
                              fitting_parameters: Optional[dict[str, Any]] = None) -> list[tuple[str, Any]]:
        """
        Gets list of estimators for data transforming before fitting or predicting
        :param for_predicting: True is need to form estimators for predicting
        :param fitting_parameters: parameters for fitting
        :return: list of estimators
        """
        estimator_types_list = [
                                DataTransformersTypes.READER,
                                DataTransformersTypes.CHECKER,
                                DataTransformersTypes.ROW_COLUMN_TRANSFORMER,
                                DataTransformersTypes.CATEGORICAL_ENCODER,
                                DataTransformersTypes.NAN_PROCESSOR,
                                DataTransformersTypes.SCALER
        ]

        if not for_predicting:
            estimator_types_list.append(DataTransformersTypes.SHUFFLER)

        estimators = []
        for estimator_type in estimator_types_list:
            estimator = self._get_estimator(estimator_type, fitting_parameters)
            estimators.append((estimator_type.value, estimator))

        return estimators

    def _get_estimator(self, transformer_type: DataTransformersTypes,
                       fitting_parameters: Optional[dict[str, Any]] = None) -> base_transformer.BaseTransformer:
        """
        Gets estimator according to type.
        :param transformer_type: type of estimator
        :param fitting_parameters: parameters of fitting
        :return: required estimator
        """
        estimator_class = get_transformer_class(transformer_type, self.parameters.type)
        estimator = estimator_class(self.parameters, self.fitting_parameters, model_id=self._id)
        if fitting_parameters:
            estimator.set_additional_parameters(fitting_parameters)

        return estimator

    def _get_model_pipeline(self, for_predicting: bool = False,
                            fitting_parameters: Optional[dict[str, Any]] = None) -> Pipeline:
        """
        Gets pipeline of transformers for fitting or predicting
        :param for_predicting: True is need to form pipeline for predicting
        :param fitting_parameters: parameters of fitting
        :return: pipeline of estimators to transform data
        """
        estimators = self._get_model_estimators(for_predicting, fitting_parameters)

        return Pipeline(estimators)

    def _write_to_db(self, write_scaler: bool = False) -> None:
        """
        Writes model to db
        :param write_scaler: also writes scaler to DB if True
        """
        model_to_db = {'id': self._id}
        model_to_db.update(self.parameters.get_all())
        model_to_db.update(self.fitting_parameters.get_all(for_db=True))

        model_to_db['filter'] = model_to_db['filter'].get_value_as_bytes()

        self._db_connector.set_line('models', model_to_db, {'id': self._id})

        if write_scaler:
            self._scaler.write_to_db()

    def _read_from_db(self, read_scaler: bool = False):
        """
        Reads model from db
        :param write_scaler: also reads scaler from DB if True
        """
        model_from_db = self._db_connector.get_line('models', {'id': self._id})

        if model_from_db:
            self.parameters.set_all(model_from_db, without_processing=True)
            self.fitting_parameters.set_all(model_from_db, without_processing=True)

            if read_scaler:
                self._scaler.read_from_db()

            self._initialized = True
        else:
            self._initialized = False

            self.parameters = get_model_parameters_class()()
            self.fitting_parameters = get_model_parameters_class(fitting=True)()

            self._engine = None

    def _predict_model(self, x_input: list[dict[str, Any]]) -> dict[str, Any]:
        """
        For predicting data after check and prepare parameters
        :param x_input: list of input data
        :return: dict of result of predicting
        """
        pipeline = self._get_model_pipeline(for_predicting=True)
        data = pipeline.transform(x_input)

        x = self._data_to_x(data)
        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)
        if not self._engine:
            self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number)

        y_pred = self._engine.predict(x)

        self._scaler.read_from_db()

        result_data = self._y_to_data(y_pred, data)
        return {'output': result_data.to_dict('records'), 'description': self._form_output_columns_description()}

    def _form_output_columns_description(self):
        """
        Creates columns description
        :return: description
        """
        return self.fitting_parameters.y_columns

    def _get_metrics(self, y: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {}

    @property
    def id(self) -> str:
        """
        Property gets value of model id. Cannot be set out of __init__
        :return: value of property
        """
        return self._id

    @property
    def initialized(self) -> bool:
        """
        Property gets value of status initialized. Cannot be set out of __init__
        :return: value of property
        """
        return self._initialized


def get_additional_actions() -> dict[str, Callable]:
    """
    Forms dict of additional actions of models package. In base model result is empty
    :return: dict of actions (functions)
    """
    return {}
