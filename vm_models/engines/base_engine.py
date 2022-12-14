
from typing import Any, Optional
import numpy as np

from db_processing import connectors, get_connector
from ..model_parameters.base_parameters import ModelParameters, FittingParameters

class BaseEngine:
    service_name: str = ''
    model_type = ''

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, db_path: str,
                 model_id: str, **kwargs):

        self._model_id: str = model_id
        self._model_parameters: ModelParameters = model_parameters
        self._fitting_parameters: FittingParameters = fitting_parameters

        self._db_connector: connectors.base_connector.Connector  = get_connector(db_path)

        self._input_number = len(self._fitting_parameters.x_columns)
        self._output_number = len(self._fitting_parameters.y_columns)

    def fit(self, x: np.ndarray,  y: np.ndarray, epochs: int,
            parameters: Optional[dict[str, Any]] = None)-> dict[str, Any]:
        return {'description': 'Fit OK'}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x

    def drop(self) -> None:
        self._db_connector.delete_lines('engines', {'model_id': self._model_id})