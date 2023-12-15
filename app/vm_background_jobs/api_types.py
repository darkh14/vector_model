""" Module to define data types for api """

from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Any
from .job_types import JobStatuses


class JobInfo(BaseModel):
    """
    Info of background job (response)
    """
    id: str
    job_name: str
    status: JobStatuses
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    pid: int
    error: str
    output: str


class FilterBody(BaseModel):
    """
    Data filter to choose data from DB
    """
    data_filter: Optional[dict[str, Any]] = None
