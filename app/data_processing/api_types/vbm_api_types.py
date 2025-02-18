""" Module to define data types for api """

import datetime
import enum
from typing import Optional, Any
from pydantic import BaseModel, Extra

from . import base_api_types


SERVICE_NAME = 'vbm'


class DataRow(base_api_types.DataRow):
    """
    Loading, input or output data row
    """
    organisation: str
    scenario: str
    period: datetime.datetime
    indicator: str
    analytic_key: str
    sum: float
    qty: float


class LoadingDataRow(DataRow):
    """
    Loading data row
    """
    remove: bool


class Inputs(base_api_types.Inputs):
    """
    Data in loading or while predicting
    """
    inputs: list[DataRow]


class Organisation(BaseModel):
    """
    Organisation description data
    """
    id: str
    name: str
    description: Optional[Any] = None


class Scenario(BaseModel):
    """
    Scenario description data
    """
    id: str
    name: str
    periodicity: base_api_types.Periodicity
    description: Optional[Any] = None


class Indicator(BaseModel):
    """
    Indicator description data
    """
    id: str
    name: str
    description: Optional[Any] = None
    color: Optional[str] = None


class Analytics(BaseModel):
    """
    Analytic description data
    """
    id: str
    name: str
    description: Optional[Any] = None


class AnalyticKey(BaseModel):
    """
    Analytic key description data
    """
    key: str
    kind: str
    value_id: str


class PackageWithData(base_api_types.PackageWithData):
    """
    Package of data for loading
    """
    loading_id: str
    id: str
    data: list[LoadingDataRow]
    organisations: Optional[list[Organisation]] = None
    scenarios: Optional[list[Scenario]] = None
    indicators: Optional[list[Indicator]] = None
    analytic_keys: Optional[list[AnalyticKey]] = None
    analytics: Optional[list[Analytics]] = None

    @classmethod
    def get_data_fields(cls) -> list[str]:
        """
        Returns list of description catalog
        """
        result = super().get_data_fields()

        result += ['organisations', 'scenarios', 'indicators', 'analytic_keys', 'analytics']

        return result


class ColumnDescription(BaseModel):
    """
    Description of model columns data
    """
    name: str
    indicator: str
    analytic_key: str
    period_shift: int
    period_number: int
    period_accumulation: bool
    value_name: str


class PredictedDataRow(base_api_types.PredictedDataRow):
    """
    Predicted output data row
    Allows to add data columns
    Extra columns are model feature columns (type - float)
    """

    period: datetime.datetime
    organisation: str
    scenario: str

    class Config:
        extra = Extra.allow


class PredictedDescription(BaseModel):
    """
    Description of predicted data
    """
    columns_description: Optional[list[ColumnDescription]] = None
    organisations: Optional[list[Organisation]] = None
    scenarios: Optional[list[Scenario]] = None
    indicators: Optional[list[Indicator]] = None
    analytic_keys: Optional[list[AnalyticKey]] = None
    analytics: Optional[list[Analytics]] = None


class PredictedOutputs(base_api_types.PredictedOutputs):
    """
    Response  of predicting, data and its description
    """
    outputs: list[PredictedDataRow]
    description: PredictedDescription


class SensitivityAnalysisInputData(BaseModel):
    """
    Sensitivity analysis data request
    """
    inputs: list[DataRow]
    input_indicators: list[str]
    output_indicator: str
    deviations: list[float]
    value_name: str
    get_graph: bool = False
    auto_selection_number: int = 0


class SADataRow(BaseModel):
    """
    Sensitivity analysis data row
    """
    scenario: str
    organisation: str
    period: Optional[datetime.datetime] = None
    indicator: str
    deviation: float
    y: float
    y_0: float
    delta: float
    relative_delta: float


class SADescription(BaseModel):
    """
    Sensitivity analysis data description
    """
    organisations: Optional[list[Organisation]] = None
    scenarios: Optional[list[Scenario]] = None
    indicators: Optional[list[Indicator]] = None


class SensitivityAnalysisOutputData(BaseModel):
    """
    Sensitivity analysis data response
    """
    outputs: list[SADataRow]
    description: SADescription
    graph_data: str = ''


class FeatureImportancesOutputRow(BaseModel):
    """
    Feature importances response data row
    """
    feature: str
    value_name: Optional[str] = None
    error_delta: float
    influence_factor: float


class FeatureImportancesOutputData(BaseModel):
    """
    Feature importances data response
    Data rows and its description
    """
    outputs: list[FeatureImportancesOutputRow]
    description: PredictedDescription


class ScenarioTypes(enum.Enum):
    """
    Scenario types in factor analysis
    """
    BASE = 'base'
    CALCULATED = 'calculated'


class FactorAnalysisDataRow(DataRow):
    """
    Factor analysis input data row
    Inherited by DataRow
    """
    scenario_type: ScenarioTypes
    is_main_period: bool = True


class FactorAnalysisInputData(BaseModel):
    """
    Factor analysis input data
    """
    data: list[FactorAnalysisDataRow]
    output_value_based: float
    output_value_calculated: float
    output_value_name: str
    input_indicators: list[str]
    output_indicator: str


class FactorAnalysisOutputDataRow(BaseModel):
    """
    Factor analysis output data row (response)
    """
    indicator: str
    value: float


class FactorAnalysisOutputData(BaseModel):
    """
    Factor analysis output data (response)
    """
    data: list[FactorAnalysisOutputDataRow]
    description: PredictedDescription
    graph_data: str = ''
