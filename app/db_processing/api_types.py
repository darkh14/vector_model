""" Module to define data types for api """

from api_types import BaseResult

from pydantic import BaseModel
from typing import Any, Optional


class InputDB(BaseResult):
    path: str


class OutputDB(InputDB):
    name: str


class Collection(BaseModel):
    data: list = [dict[str, Any]]


class DataFilterBody(BaseResult):
    """
    Data filter
    to choose data in db
    """
    data_filter: Optional[dict[str, Any]]

