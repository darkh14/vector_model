""" Module for base transformer classes.
    Classes:
        DataTransformer - to transform data while fitting and predicting
"""

from typing import TypeVar, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


from ..model_parameters.base_parameters import ModelParameters, FittingParameters
from db_processing import get_connector
from ..model_types import DataTransformersTypes

BaseTransformerClass = TypeVar('BaseTransformerClass', bound='BaseTransformer')

__all__ = ['BaseTransformer', 'Reader', 'Checker', 'RowColumnTransformer', 'Scaler']

class BaseTransformer(BaseEstimator, TransformerMixin):
    service_name = ''
    transformer_type = DataTransformersTypes.NONE

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, db_path: str, **kwargs):
        self._model_parameters = model_parameters
        self._fitting_parameters = fitting_parameters

        self._db_connector = get_connector(db_path)

    def fit(self, x: Optional[list[dict[str, Any]]] = None,
            y: Optional[list[dict[str, Any]]] = None)-> BaseTransformerClass:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x


class Reader(BaseTransformer):
    service_name: str = ''
    transformer_type: DataTransformersTypes = DataTransformersTypes.READER

    def transform(self, x: Optional[list[dict[str, Any]]]) -> pd.DataFrame:
        raw_data = self._db_connector.get_lines('raw_data', self._model_parameters.get_data_filter_for_db())

        return pd.DataFrame(raw_data)


class Checker(BaseTransformer):
    service_name: str = ''
    transformer_type: DataTransformersTypes = DataTransformersTypes.CHECKER


class RowColumnTransformer(BaseTransformer):
    service_name: str = ''
    transformer_type: DataTransformersTypes = DataTransformersTypes.ROW_COLUMN_TRANSFORMER


class CategoricalEncoder(BaseTransformer):
    service_name: str = ''
    transformer_type: DataTransformersTypes = DataTransformersTypes.CATEGORICAL_ENCODER


class NanProcessor(BaseTransformer):
    service_name: str = ''
    transformer_type: DataTransformersTypes = DataTransformersTypes.NAN_PROCESSOR


class Scaler(BaseTransformer):
    service_name: str = ''
    transformer_type: DataTransformersTypes = DataTransformersTypes.SCALER
