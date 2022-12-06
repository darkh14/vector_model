
from typing import Any
import numpy as np

from db_processing import get_connector

class BaseEngine:
    service_name: str = ''

    def __init__(self, model_parameters, fitting_parameters, db_path, **kwargs):
        self._model_parameters = model_parameters
        self._fitting_parameters = fitting_parameters

        self._db_connector = get_connector(db_path)

    def initialize(self, engine_parameters: dict[str, Any]): ...

    def fit(self, x: np.ndarray,  y: np.ndarray)-> dict[str, Any]:
        return {'description': 'Fit OK'}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x