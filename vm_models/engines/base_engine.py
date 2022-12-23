""" Module of base model engine class.
    Classes:
        BaseEngine - main class of ML model engine. supports fit, predict and drop from db
"""


from typing import Any, Optional, ClassVar
import numpy as np
from abc import ABC, abstractmethod

from db_processing import connectors, get_connector

class BaseEngine(ABC):
    """ Main class of ML model engine. supports fit, predict and drop from db. Real ML classes
        are inherited from this class
        Methods:
             fit - for fitting engine
             predict - for predicting data
             drop - for deleting engine from db
    """
    service_name: ClassVar[str] = ''
    model_type: ClassVar[str] = ''

    def __init__(self, model_id: str, input_number: int, output_number: int, new_engine: bool = False,
                 **kwargs) -> None:
        """
        Defines model_id, db connector, input, output number, _new_engine and metrics
        :param model_id: id of model class
        :param input_number: number of inputs
        :param output_number: number of outputs
        :param new_engine: is new engine (no need to read from db)
        :param kwargs: additional parameters (for subclasses)
        """
        self._model_id: str = model_id

        self._db_connector: connectors.base_connector.Connector  = get_connector()

        self._input_number: int = input_number
        self._output_number: int = output_number

        self._new_engine: bool = new_engine

        self.metrics: dict[str, Any] = {}

    @abstractmethod
    def fit(self, x: np.ndarray,  y: np.ndarray, epochs: int,
            parameters: Optional[dict[str, Any]] = None)-> dict[str, Any]:
        """
        For fitting ML engine
        :param x: inputs
        :param y: outputs (labels)
        :param epochs: nuber of epochs
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