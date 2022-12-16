""" Module for base transformer classes.
    Classes:
        DataTransformer - to transform data while fitting and predicting
"""

from typing import TypeVar, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


from ..model_parameters.base_parameters import ModelParameters, FittingParameters
from db_processing import get_connector
from db_processing.connectors import base_connector
from ..model_types import DataTransformersTypes
from ..model_filters import get_fitting_filter_class, base_filter

BaseTransformerClass = TypeVar('BaseTransformerClass', bound='BaseTransformer')

__all__ = ['BaseTransformer',
           'Reader',
           'Checker',
           'RowColumnTransformer',
           'Scaler',
           'CategoricalEncoder',
           'NanProcessor']

class BaseTransformer(BaseEstimator, TransformerMixin):
    service_name: str = ''
    transformer_type: str = DataTransformersTypes.NONE

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, db_path: str, **kwargs):
        self._model_parameters: ModelParameters = model_parameters
        self._fitting_parameters: FittingParameters = fitting_parameters

        self._db_connector: base_connector.Connector = get_connector(db_path)

        self._fitting_mode = False

    def fit(self, x: Optional[list[dict[str, Any]] | pd.DataFrame] = None,
            y: Optional[list[dict[str, Any]] | pd.DataFrame] = None)-> BaseTransformerClass:
        self._fitting_mode = True

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x

    def initialize(self, transformer_parameters: dict[str, Any]) -> None: ...

    def set_additional_parameters(self, parameters: dict[str, Any]) -> None: ...


class Reader(BaseTransformer):
    service_name: str = ''
    transformer_type: DataTransformersTypes = DataTransformersTypes.READER

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, db_path: str, **kwargs):
        super().__init__(model_parameters, fitting_parameters, db_path)
        self._fitting_filter: Optional[base_filter.FittingFilter] = None

    def transform(self, x: Optional[list[dict[str, Any]]]) -> pd.DataFrame:

        if self._fitting_mode:
            raw_data = self._read_while_fitting()
        else:
            raw_data = self._read_while_predicting(x)

        return raw_data

    def _read_while_fitting(self) -> pd.DataFrame:

        data_filter = self._model_parameters.get_data_filter_for_db()

        ad_data_filter = self._fitting_filter.get_value_as_db_filter() if self._fitting_filter else None

        if data_filter and ad_data_filter:
            result_filter = {'$and': [data_filter, ad_data_filter]}
        elif ad_data_filter:
            result_filter = ad_data_filter
        elif data_filter:
            result_filter = data_filter
        else:
            result_filter = None

        raw_data = self._db_connector.get_lines('raw_data', result_filter)
        raw_data = pd.DataFrame(raw_data)

        return raw_data

    def _read_while_predicting(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(data)

    def set_additional_parameters(self, parameters: dict[str, Any]) -> None:
        if 'filter' in parameters:
            self._fitting_filter = get_fitting_filter_class()(parameters['filter'])


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
