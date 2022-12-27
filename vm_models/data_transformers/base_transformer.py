""" Module for base transformer classes.
    Classes:
        DataTransformer - to transform data while fitting and predicting
        Reader - for reading data from db
        Checker - for checking data
        RowColumnTransformer - for forming data structure
        CategoricalEncoder - for forming categorical fields
        NanProcessor - for working with nan values
        Scaler - for data scaling
"""

from typing import TypeVar, Any, Optional, ClassVar
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
    """ Base transformer class for transform data. Supports sklearn pipeline, methods fit and transform
        Methods:
             fit - fit transformer parameters to data
             transform - to transform data
             set_additional_parameters - to set additional parameters for some transformer if it is need

    """
    service_name: ClassVar[str] = ''
    model_type: ClassVar[str] = ''
    transformer_type: ClassVar[DataTransformersTypes] = DataTransformersTypes.NONE

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters, **kwargs) -> None:
        """
        Defines model and fitting parameters, create db connector
        :param model_parameters: model parameters object
        :param fitting_parameters: fitting parameters object
        :param kwargs: additional parameters
        """
        self._model_parameters: ModelParameters = model_parameters
        self._fitting_parameters: FittingParameters = fitting_parameters

        self._db_connector: base_connector.Connector = get_connector()

        self._fitting_mode = False

    def fit(self, x: Optional[list[dict[str, Any]] | pd.DataFrame] = None,
            y: Optional[list[dict[str, Any]] | pd.DataFrame] = None) -> BaseTransformerClass:
        """
        For fitting transformer parameters to data
        :param x: input data
        :param y: output data (optional for fitting)
        :return: self (transformer object)
        """
        self._fitting_mode = True

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        For transforming data (main method)
        :param x: input data
        :return: data after transforming
        """
        return x

    def set_additional_parameters(self, parameters: dict[str, Any]) -> None:
        """
        For setting additional parameters (optional for some types of transformers)
        :param parameters: dict of parameters to set
        """


class Reader(BaseTransformer):
    """
        Transformer class for reading data from db
    """
    service_name: ClassVar[str] = ''
    transformer_type: ClassVar[DataTransformersTypes] = DataTransformersTypes.READER

    def __init__(self, model_parameters: ModelParameters, fitting_parameters: FittingParameters):
        """
        Defines fitting filter
        :param model_parameters: model parameters object
        :param fitting_parameters: fitting parameters object
        """
        super().__init__(model_parameters, fitting_parameters)
        self._fitting_filter: Optional[base_filter.FittingFilter] = None

    def transform(self, x: Optional[list[dict[str, Any]]]) -> pd.DataFrame:
        """
        Reads data from db, adds short ids and some other fields
        :param x: None
        :return: data read
        """
        if self._fitting_mode:
            raw_data = self._read_while_fitting()
        else:
            raw_data = self._read_while_predicting(x)

        return raw_data

    def _read_while_fitting(self) -> pd.DataFrame:
        """
        For reading data while fitting (reading from db)
        :return: data read
        """
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
        """
        For reading while predicting. Really data is not read, but transform to pd.DataFrame
        :param data: input data
        :return: data after transforming
        """
        return pd.DataFrame(data)

    def set_additional_parameters(self, parameters: dict[str, Any]) -> None:
        """
        For setting filter
        :param parameters: additional parameters, can contain filter
        """
        if 'filter' in parameters:
            self._fitting_filter = get_fitting_filter_class()(parameters['filter'])


class Checker(BaseTransformer):
    """
        Transformer class for checking data
    """
    service_name: ClassVar[str] = ''
    transformer_type: ClassVar[DataTransformersTypes] = DataTransformersTypes.CHECKER


class RowColumnTransformer(BaseTransformer):
    """
        Transformer class for transform rows to columns (form data structure)
    """
    service_name: ClassVar[str] = ''
    transformer_type: ClassVar[DataTransformersTypes] = DataTransformersTypes.ROW_COLUMN_TRANSFORMER


class CategoricalEncoder(BaseTransformer):
    """
        Transformer class to form categorical fields (one-hot encoding)
    """
    service_name: ClassVar[str] = ''
    transformer_type: ClassVar[DataTransformersTypes] = DataTransformersTypes.CATEGORICAL_ENCODER


class NanProcessor(BaseTransformer):
    """
        Transformer class for working with nan values
    """
    service_name: ClassVar[str] = ''
    transformer_type: ClassVar[DataTransformersTypes] = DataTransformersTypes.NAN_PROCESSOR


class Scaler(BaseTransformer):
    """
        Transformer class to scale data (minmax, standard or other)
    """
    service_name: ClassVar[str] = ''
    transformer_type: ClassVar[DataTransformersTypes] = DataTransformersTypes.SCALER
