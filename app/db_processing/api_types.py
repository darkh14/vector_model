""" Module to define data types for api """

from api_types import BaseResult


class DBCopyTo(BaseResult):
    """
    DB connection string of DB receiver
    """
    db_copy_to: str
