""" Module to define data types for api """

from datetime import datetime

from typing import Optional, Any
from pydantic import BaseModel
from .. import model_types


SERVICE_NAME = 'vector_model'


class Model(BaseModel):
    """
    Model info request while its initializing
    """
    id: str
    name: str
    type: model_types.ModelTypes
    filter: Optional[dict[str, Any]]


class ModelInfo(Model):
    """
    Model info response (model info response)
    """
    initialized: bool
    fitting_status: model_types.FittingStatuses
    fitting_date: Optional[datetime] = None
    fitting_start_date: Optional[datetime] = None
    fitting_error_text: str = ''
    fitting_error_date: Optional[datetime] = None
    fitting_job_id: str = ''
    fitting_job_pid: int = 0

    metrics: dict[str, float]


class FittingParameters(BaseModel):
    """
    Fitting parameters of model
    """
    pass



