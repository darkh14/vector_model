""" Contains model class, that provides working with models including
    fitting and predicting
    Classes:
        Model - base model class, provides fitting and predicting. Saves itself to db
"""
import enum
from typing import Any, Callable, Optional, ClassVar

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from vm_logging.exceptions import ModelException
from vm_models.model_parameters import get_model_parameters_class
from ..model_parameters import base_parameters
from db_processing import get_connector as get_db_connector
from db_processing.connectors import base_connector
from ..data_transformers import get_transformer_class, base_transformer
from ..engines import get_engine_class, base_engine
from ..model_types import DataTransformersTypes, FittingStatuses
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

        self._scaler: Optional[base_transformer.Scaler] = None

        self._read_from_db()

    def initialize(self, model_parameters: dict[str, Any]) -> str:
        """
        For initializing model in db
        :param model_parameters: data of new model
        :return: result of initializing
        """
        if self._initialized:
            raise ModelException('Model "{}" id - "{}" is always initialized'.format(self.parameters.name,
                                                                                     self._id))

        self.parameters.set_all(model_parameters)
        self.fitting_parameters.set_all(model_parameters)

        self._write_to_db()

        self._initialized = True

        return 'Model is initialized'

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

        if self._scaler:
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

    def fit(self, fitting_parameters: Optional[dict[str, Any]], job_id: str = '') -> str:
        """
        For fitting model
        :param fitting_parameters: parameters of fitting (ex. epochs)
        :param job_id: id of job if fitting is background
        :return: fitting history
        """

        try:
            self._check_before_fitting(job_id)

            if fitting_parameters:
                all_parameters = self.fitting_parameters.get_all()
                all_parameters.update(fitting_parameters)
                self.fitting_parameters.set_all(all_parameters)

            self.fitting_parameters.set_start_fitting(job_id)
            self._write_to_db()

            self._fit_model(fitting_parameters)
        except Exception as ex:
            self.fitting_parameters.set_error_fitting(str(ex))
            self._write_to_db()
            raise ex

        if not self.fitting_parameters.fitting_status == FittingStatuses.Error:

            self.fitting_parameters.set_end_fitting()
            self._write_to_db()

        return 'Model is fit'

    def predict(self, x: list[dict[str, Any]]) -> dict[str, Any]:
        """
        For predicting data with model
        :param x: input data for predicting
        :return: predicted output data
        """
        self._check_before_predicting(x)

        result_data = self._predict_model(x)
        result_data = result_data.drop(self.fitting_parameters.x_columns, axis=1)

        return {'output': result_data.to_dict('records')}

    def drop_fitting(self) -> str:
        """
        Deletes results of fitting from db
        :return: resul of dropping
        """

        if self.fitting_parameters.fitting_status not in (FittingStatuses.Fit,
                                                          FittingStatuses.PreStarted,
                                                          FittingStatuses.Started,
                                                          FittingStatuses.Error):

            raise ModelException('Can not drop fitting. Model is not fit')

        self._interrupt_fitting_job()

        if self._engine:
            self._engine.drop()

        if self._scaler:
            self._scaler.drop()

        self.fitting_parameters.set_drop_fitting()
        self._write_to_db()

        return 'Model "{}" id "{}" fitting is dropped'.format(self.parameters.name, self.id)

    def get_action_before_background_job(self, func_name: str,
                                         args: tuple[Any],
                                         kwargs: dict[str, Any]) -> Optional[Callable]:
        """Returns function which will be executed before model fit
        @param func_name: name of fit function
        @param args: positional arguments of fit function
        @param kwargs: keyword arguments of fit function.
        @return: function to execute before fit
        """
        result = None
        if func_name == 'fit':
            result = self.do_before_fit

        return result

    def get_action_error_background_job(self, func_name: str,
                                        args: tuple[Any],
                                        kwargs: dict[str, Any]) -> Optional[Callable]:
        """Returns function which will be executed while error model fit
        @param func_name: name of fit function
        @param args: positional arguments of fit function
        @param kwargs: keyword arguments of fit function.
        @return: function to execute before fit
        """
        result = None
        if func_name == 'fit':
            result = self.do_error_fit

        return result

    # noinspection PyUnusedLocal
    def do_before_fit(self, args: tuple[Any], kwargs: dict[str, Any]) -> None:

        self.fitting_parameters.set_pre_start_fitting(kwargs.get('job_id', ''))
        self._write_to_db()

    # noinspection PyUnusedLocal
    def do_error_fit(self, args: tuple[Any], kwargs: dict[str, Any]) -> None:

        if self.fitting_parameters.fitting_status != FittingStatuses.Error:
            self.fitting_parameters.set_error_fitting(kwargs.get('error_text', ''))
            self._write_to_db()

    def _interrupt_fitting_job(self) -> None:
        """
        Interrupts fitting job process when fitting is launched in background mode
        """
        if self.fitting_parameters.fitting_status == FittingStatuses.Started:
            set_background_job_interrupted(self.fitting_parameters.fitting_job_id)

    def _fit_model(self, fitting_parameters: Optional[dict[str, Any]], job_id: str = '') -> dict[str, Any]:
        """
        For fitting model after checking, and preparing parameters,
        :param fitting_parameters: parameters of fitting
        :param job_id: id of job if fitting is background
        :return: fitting history
        """

        # noinspection PyTypeChecker
        self._scaler = get_transformer_class(DataTransformersTypes.SCALER, self.parameters.type)(self.parameters,
                                                    self.fitting_parameters, model_id=self._id,
                                                    new_scaler=self.fitting_parameters.is_first_fitting())

        pipeline = self._get_model_pipeline(for_predicting=False, fitting_parameters=fitting_parameters)
        data = pipeline.fit_transform(None)

        x, y = self._data_to_x_y(data)
        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)

        self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number,
                                                            self.fitting_parameters.is_first_fitting(),
                                                            self.parameters)
        fitting_result = self._engine.fit(x, y, fitting_parameters)
        result_history = fitting_result['history']

        self._scaler = pipeline.named_steps['scaler']

        y_pred = self._engine.predict(x)

        data_predicted = data.copy()
        data_predicted[self.fitting_parameters.y_columns] = y_pred

        data = self._scaler.inverse_transform(data)
        data_predicted = self._scaler.inverse_transform(data_predicted)

        y = data[self.fitting_parameters.y_columns].to_numpy()
        y_pred = data_predicted[self.fitting_parameters.y_columns].to_numpy()

        self.fitting_parameters.metrics = self._get_metrics(y, y_pred)

        result_engine = self._engine

        return {'history': result_history, 'x': x, 'y': y, 'engine': result_engine}

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

        result = x_data.copy()

        result[self.fitting_parameters.y_columns] = y

        numeric_columns = self.fitting_parameters.x_columns + self.fitting_parameters.y_columns

        result[numeric_columns] = self._scaler.inverse_transform(result[numeric_columns])

        return result

    def _check_before_fitting(self, job_id: str = '') -> None:
        """
        For checking statuses and other parameters before fitting. Raises ModelException if checking is failed
        :param job_id: if of fitting job if fitting is background
        """
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        if self.fitting_parameters.fitting_status == FittingStatuses.Started:
            raise ModelException('Another fitting is started yet. Wait for end of fitting')

        if job_id:
            if not self.fitting_parameters.fitting_status == FittingStatuses.PreStarted:
                raise ModelException('Model is not prepared for fitting in background. ' +
                                     'Drop fitting and execute another fitting job')
        else:
            if self.fitting_parameters.fitting_status == FittingStatuses.PreStarted:
                raise ModelException('Model is not prepared for fitting. ' +
                                     'Drop fitting and execute another fitting')

    def _check_before_predicting(self, inputs: list[dict[str, Any]]) -> None:
        """
        For checking statuses and other parameters before predicting. Raises ModelException if checking is failed
        """
        if not self._initialized:
            raise ModelException('Model id - {} is not initialized'.format(self._id))

        if not self.fitting_parameters.fitting_status == FittingStatuses.Fit:
            raise ModelException(('Model "{}" id "{}" is not fit. ' +
                                 'Fit model before predicting').format(self.parameters.name, self._id))

        loading_engine = get_loading_engine_class()()

        loading_engine.check_data(inputs)

    def _get_model_estimators(self, for_predicting: bool = False,
                              fitting_parameters: Optional[dict[str, Any]] = None,
                              without_scaling: bool = False) -> list[tuple[str, Any]]:
        """
        Gets list of estimators for data transforming before fitting or predicting
        :param for_predicting: True is need to form estimators for predicting
        :param fitting_parameters: parameters for fitting
        :param without_scaling: without scaler adding if True
        :return: list of estimators
        """
        estimator_types_list = self._get_estimator_types(for_predicting, fitting_parameters)

        estimators = []
        for estimator_type in estimator_types_list:
            if estimator_type == DataTransformersTypes.SCALER:
                if without_scaling:
                    continue
                estimator = self._scaler
            else:
                estimator = self._get_estimator(estimator_type, fitting_parameters)
            estimators.append((estimator_type.value, estimator))

        return estimators

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def _get_estimator_types(self, for_predicting: bool = False,
                             fitting_parameters: Optional[dict[str, Any]] = None) -> list[DataTransformersTypes]:
        """
        Returns list of estimator types
        @param for_predicting: True if it needs to form estimator list to predict data otherwise to fit model
        @param fitting_parameters: dict of fitting parameters
        @return: list of estimator types
        """

        estimator_types = [DataTransformersTypes.READER,
                           DataTransformersTypes.CHECKER,
                           DataTransformersTypes.ROW_COLUMN_TRANSFORMER,
                           DataTransformersTypes.CATEGORICAL_ENCODER,
                           DataTransformersTypes.NAN_PROCESSOR,
                           DataTransformersTypes.SCALER]

        if not for_predicting:
            estimator_types.append(DataTransformersTypes.SHUFFLER)

        return estimator_types

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
                            fitting_parameters: Optional[dict[str, Any]] = None,
                            without_scaling: bool = False) -> Pipeline:
        """
        Gets pipeline of transformers for fitting or predicting
        :param for_predicting: True is need to form pipeline for predicting
        :param fitting_parameters: parameters of fitting
        :return: pipeline of estimators to transform data
        """
        estimators = self._get_model_estimators(for_predicting, fitting_parameters, without_scaling=without_scaling)

        return Pipeline(estimators)

    def _write_to_db(self) -> None:
        """
        Writes model to db
        """
        model_to_db = {'id': self._id}
        model_to_db.update(self.parameters.get_all())
        model_to_db.update(self.fitting_parameters.get_all())

        for key, value in model_to_db.items():
            if isinstance(value, enum.Enum):
                model_to_db[key] = value.value

        model_to_db['filter'] = model_to_db['filter'].get_value_as_bytes()

        self._db_connector.set_line('models', model_to_db, {'id': self._id})

    def _read_from_db(self):
        """
        Reads model from db
        """
        model_from_db = self._db_connector.get_line('models', {'id': self._id})

        if model_from_db:
            for key, value in model_from_db.items():
                if isinstance(self.parameters.__dict__.get(key), enum.Enum):
                    model_from_db[key] = getattr(self.parameters, key).__class__(value)
                if isinstance(self.fitting_parameters.__dict__.get(key), enum.Enum):
                    model_from_db[key] = getattr(self.fitting_parameters, key).__class__(value)

            self.parameters.set_all(model_from_db)
            self.fitting_parameters.set_all(model_from_db)

            self._initialized = True
        else:
            self._initialized = False

            self.parameters = get_model_parameters_class()()
            self.fitting_parameters = get_model_parameters_class(fitting=True)()

            self._engine = None
            self._scaler = None

    def _predict_model(self, x_input: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        """
        For predicting data after check and prepare parameters
        :param x_input: list of input data
        :return: dict of result of predicting
        """

        if not self._scaler:
            self._scaler = get_transformer_class(DataTransformersTypes.SCALER, self.parameters.type)(self.parameters,
                                                                                self.fitting_parameters,
                                                                                model_id=self._id,)

        pipeline = self._get_model_pipeline(for_predicting=True)
        data = pipeline.transform(x_input)

        x = self._data_to_x(data)
        input_number = len(self.fitting_parameters.x_columns)
        output_number = len(self.fitting_parameters.y_columns)
        if not self._engine:
            self._engine = get_engine_class(self.parameters.type)(self._id, input_number, output_number,
                                                                  parameters=self.parameters)

        y_pred = self._engine.predict(x)

        result_data = self._y_to_data(y_pred, data)

        return result_data

    def _get_metrics(self, y: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Returns value of model quality metrics
        @param y: True output values
        @param y_pred: Predicted output values
        @return: dict of metrics
        """
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
