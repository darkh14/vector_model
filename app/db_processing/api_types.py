""" Module to define data types for api """

from api_types import BaseResult


class InputDB(BaseResult):
    path: str


class OutputDB(InputDB):
    name: str
