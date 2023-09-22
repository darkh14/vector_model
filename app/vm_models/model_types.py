""" Module for defining enums and other types using in models
    Classes:
        DataTransformerTypes - types of data transformers
"""

from enum import Enum

__all__ = ['DataTransformersTypes', 'ModelTypes', 'FittingStatuses']


class DataTransformersTypes(Enum):
    """
    READER - transformer to read data
    CHECKER - transformer to check data
    ROW_COLUMN_TRANSFORMER - transformer to transform data from rows to columns
    CATEGORICAL_ENCODER - transformer to add categorical columns
    NAN_PROCESSOR - transform to delete or replace Nan values
    SCALER - transformer to scale data (minmax or standard)
    SHUFFLER - transformer to shuffle data rows
    NONE - empty transformer
    """
    READER = 'reader'
    CHECKER = 'checker'
    ROW_COLUMN_TRANSFORMER = 'row_column_transformer'
    CATEGORICAL_ENCODER = 'categorical_encoder'
    NAN_PROCESSOR = 'nan_processor'
    SCALER = 'scaler'
    SHUFFLER = 'shuffler'
    NONE = 'none'


class ModelTypes(Enum):
    """
    NeuralNetwork - 3 layer direct distribution NN
    LinearRegression - linear regression based on sklearn.linear_model
    PolynomialRegression - 2 degree model based on sklearn.linear_model with polynomial transformation
    """
    NeuralNetwork = 'neural_network'
    LinearRegression = 'linear_regression'
    PolynomialRegression = 'polynomial_regression'


class FittingStatuses(Enum):
    """
    NotFit - model is not fit
    PreStarted - model fit is pre-started (fitting process is started but background job is not started yet)
    Started - model fit is started
    Fit - model is fit
    Error - error while model fitting
    """
    NotFit = 'not_fit'
    PreStarted = 'pre_started'
    Started = 'started'
    Fit = 'fit'
    Error = 'error'
