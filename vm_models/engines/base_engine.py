
from typing import Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from ..model_types import DataTransformersTypes
from db_processing import get_connector

class BaseEngine(BaseEstimator, TransformerMixin):
    service_name: str = ''
    transformer_type: DataTransformersTypes  = DataTransformersTypes.ENGINE

    def __init__(self, model_parameters, fitting_parameters, db_path, **kwargs):
        self._model_parameters = model_parameters
        self._fitting_parameters = fitting_parameters

        self._db_connector = get_connector(db_path)

    def fit(self, x: pd.DataFrame,  y: pd.DataFrame)-> dict[str, Any]:
        return {'description': 'Fit OK'}

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x