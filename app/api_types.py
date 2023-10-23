""" Module to define data types for api """
import enum
import datetime

from pydantic import BaseModel
from typing import Type, Any


class RequestStatuses(enum.Enum):
    """
    Request statuses - ok or error
    """
    ok = 'OK'
    error = 'error'


class BaseResult(BaseModel):
    """
    Base response
    """
    pass


class Response(BaseModel):
    """
    Response with status, error text and result of request
    """
    status: RequestStatuses
    error_text: str
    result: BaseResult


class PingResponse(BaseResult):
    """
    Response of ping (with current date and description)
    """
    description: str
    current_date: datetime.datetime


class RequestValidationErrorDetail(BaseModel):
    """
    Response of request validation error detail (422 http code)
    """
    loc: list[str | int]
    msg: str
    type: str


class ExecutionModes(enum.Enum):
    """
    Execution modes:
    DIRECTLY - request will be executed directly in current process
    BACKGROUND - request will be executed in other process (in background)
    """
    DIRECTLY = 'directly'
    BACKGROUND = 'background'


class BackgroundJobResponse(BaseModel):
    """
    Response of background job (description and pid)
    """
    description: str
    mode: ExecutionModes
    pid: int
