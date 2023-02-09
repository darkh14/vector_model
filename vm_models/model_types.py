""" Module for defining enums and other types using in models
    Classes:
        DataTransformerTypes - types of data transformers
"""

from enum import Enum

__all__ = ['DataTransformersTypes']


class DataTransformersTypes(Enum):
    READER = 'reader'
    CHECKER = 'checker'
    ROW_COLUMN_TRANSFORMER = 'row_column_transformer'
    CATEGORICAL_ENCODER = 'categorical_encoder'
    NAN_PROCESSOR = 'nan_processor'
    SCALER = 'scaler'
    SHUFFLER = 'shuffler'
    NONE = 'none'
    
