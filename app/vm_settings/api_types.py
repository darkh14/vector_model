""" Module to define data types for api """

from pydantic import BaseModel


class VarToSet(BaseModel):
    """
    Request of var setting
    """
    name: str
    value: str | int | float | bool
