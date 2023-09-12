""" Module to define data types for api """


import datetime
import enum
from typing import Optional, Any
from pydantic import BaseModel
from api_types import BaseResult
from data_processing.loading_types import LoadingStatuses, LoadingTypes


SERVICE_NAME = 'vector_model'


class Periodicity(enum.Enum):
    """
    Defines scenario periodicity type
    to calculate difference between periods
    """
    DAY = 'day'
    WEEK = 'week'
    DECADE = 'decade'
    MONTH = 'month'
    QUARTER = 'quarter'
    HALF_YEAR = 'half_year'
    YEAR = 'year'


class Package(BaseModel):
    """
    Defines input type
    of data package while initializing loading (without data)
    """
    id: str
    number: int
    checksum: int


class PackageInfo(Package):
    """
    Defines output type
    of data package (part of loading info response)
    """
    start_date: Optional[datetime.datetime]
    end_date: Optional[datetime.datetime]
    error: str
    status: LoadingStatuses
    type: str


class DataRow(BaseModel):
    """
    Defines data row type
    """
    indicator: str
    sum: float


class LoadingDataRow(DataRow):
    """
    Defines loading data row type
    """
    remove: bool


class Inputs(BaseModel):
    """
    Defines data type in loading or while predicting
    """
    inputs: list[DataRow]


class PackageWithData(BaseModel):
    """
    Package of data for loading
    """
    loading_id: str
    id: str
    data: list[LoadingDataRow]

    @classmethod
    def get_data_fields(cls) -> list[str]:
        return ['data']


class Loading(BaseResult):
    """
    Lading type to load data
    Methods:
        to_dict - returns dict of data of this type
    """
    id: str
    type: LoadingTypes
    number_of_packages: int
    packages: list[Package]

    def to_dict(self) -> dict[str, Any]:
        """
        Returns dict of data of this type
        """
        result = dict(self)

        packages = list()
        for package in self.packages:
            packages.append(dict(package))

        result['packages'] = packages

        return result


class LoadingInfo(Loading):
    """
    Loading info type for returning info of loading
    """
    status: LoadingStatuses
    create_date: Optional[datetime.datetime]
    start_date: Optional[datetime.datetime]
    end_date: Optional[datetime.datetime]
    error: str
    packages: list[PackageInfo]


class DataFilterBody(BaseResult):
    """
    Data filter
    to choose data in db
    """
    data_filter: Optional[dict[str, Any]]


class PackageStatusBody(BaseResult):
    """
    Package type to set package status
    """
    loading_id: str
    id: str
    status: LoadingStatuses


class LoadingStatusBody(BaseResult):
    """
    Loading type to set loading status
    """
    id: str
    status: LoadingStatuses


class PredictedDataRow(BaseModel):
    """
    Output predicted data row
    """
    period: datetime.datetime


class PredictedOutputs(BaseModel):
    """
    Output predicted data
    """
    outputs: list[dict[str, Any]]
