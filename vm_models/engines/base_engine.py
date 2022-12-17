
from typing import Any, Optional
import numpy as np

from db_processing import connectors, get_connector
from ..model_parameters.base_parameters import ModelParameters, FittingParameters

class BaseEngine:
    service_name: str = ''
    model_type = ''

    def __init__(self, model_id: str, input_number: int, output_number: int, db_path: str, new_engine: bool = False,
                 **kwargs):

        self._model_id: str = model_id

        self._db_connector: connectors.base_connector.Connector  = get_connector(db_path)

        self._input_number: int = input_number
        self._output_number: int = output_number

        self._new_engine: bool = new_engine

        self.metrics: dict[str, Any] = {}

    def fit(self, x: np.ndarray,  y: np.ndarray, epochs: int,
            parameters: Optional[dict[str, Any]] = None)-> dict[str, Any]:

        return {'description': 'Fit OK', 'metrics': self.metrics}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x

    def drop(self) -> None:
        self._db_connector.delete_lines('engines', {'model_id': self._model_id})

    @staticmethod
    def _calculate_mspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0

    @staticmethod
    def _calculate_rsme(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return 0