""" Module of base model engine class.
    Classes:
        BaseEngine - main class of ML model engine. supports fit, predict and drop from db
"""


from typing import Any, Optional, ClassVar
import numpy as np
from abc import ABC, abstractmethod

from db_processing import get_connector
from db_processing.connectors.base_connector import Connector
from ..model_types import ModelTypes
from ..model_parameters import base_parameters


class BaseEngine(ABC):
    """ Main class of ML model engine. supports fit, predict and drop from db. Real ML classes
        are inherited from this class
        Methods:
             fit - for fitting engine
             predict - for predicting data
             drop - for deleting engine from db
    """
    service_name: ClassVar[str] = ''
    model_type: ClassVar[ModelTypes] = ModelTypes.NeuralNetwork

    def __init__(self, model_id: str, input_number: int, output_number: int, new_engine: bool = False,
                 parameters: Optional[base_parameters.ModelParameters] = None) -> None:
        """
        Defines model_id, db connector, input, output number, _new_engine and metrics
        :param model_id: id of model class
        :param input_number: number of inputs
        :param output_number: number of outputs
        :param new_engine: is new engine (no need to read from db)
        :param parameters: additional parameters (for subclasses)
        """
        self._model_id: str = model_id

        self._db_connector: Connector = get_connector()

        self._input_number: int = input_number
        self._output_number: int = output_number

        self._new_engine: bool = new_engine

        self._inner_engine = Optional[object]

    def fit(self, x: np.ndarray,  y: np.ndarray, parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        For fitting ML engine
        :param x: inputs
        :param y: outputs (labels)
        :param parameters: additional parameters
        :return: history of fitting
        """

        self._check_fitting_parameters(parameters)

        result = self._fit_engine(x, y, parameters)

        return result

    @abstractmethod
    def _fit_engine(self, x: np.ndarray,  y: np.ndarray,
                    parameters: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """
        For fitting ML engine after checking
        :param x: inputs
        :param y: outputs (labels)
        :param parameters: additional parameters
        :return: history of fitting
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        For predicting data with ML engine
        :param x: inputs
        :return: predicted data
        """

    def drop(self) -> None:
        """
        For deleting engine from DB
        """
        self._db_connector.delete_lines('engines', {'model_id': self._model_id})

    def get_engine_for_fi(self) -> Any:
        """
        Returns special type of model engine to calculate feature importances
        @return: engine for fi
        """
        return self._inner_engine

    @property
    def inner_engine(self) -> object:
        """
        Property for self._inner_engine
        :return: value of self._inner_engine
        """
        return self._inner_engine

    @inner_engine.setter
    def inner_engine(self, value) -> object:
        """
        Property for self._inner_engine
        :return: value of self._inner_engine
        """
        self._inner_engine = value

    # noinspection PyMethodMayBeStatic
    def _check_fitting_parameters(self, parameters: dict[str, Any]) -> None:
        pass
