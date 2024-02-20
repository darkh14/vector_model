""" Module to define data types for api """

from typing import Optional, Any
from . import base_api_types
from .. import model_types
from pydantic import BaseModel, Extra
from datetime import datetime


SERVICE_NAME = 'vbm'


class IndicatorRow(BaseModel):
    """
    Row of model indicator
    """
    id: str
    use_analytics: bool
    period_shift: Optional[int] = 0
    period_number: Optional[int] = 0
    period_accumulation: Optional[bool] = False
    use_sum: bool
    use_qty: bool
    consider_analytic_bounds: Optional[bool] = False
    analytics_bound: Optional[list[str]] = None


class Model(base_api_types.Model):
    """
    Model info (request while initializing model)
    """
    id: str
    name: str
    type: model_types.ModelTypes
    filter: Optional[dict[str, Any]]
    x_indicators: list[IndicatorRow]
    y_indicators: list[IndicatorRow]

    categorical_features: Optional[list[str]] = None

    class Config:
        extra = Extra.allow

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        result = super().model_dump(*args, **kwargs)

        if 'filter' in result:
            result['filter'] = self._dump_filter_value(result['filter'])

        return result

    def _dump_filter_value(self, filter_value: Optional[str | list[Any] | dict[str, Any]],
                           period_key: bool = False) -> Optional[str | datetime | list[Any] | dict[str, Any]]:

        if isinstance(filter_value, list):
            result_value = []
            for filter_el in filter_value:
                result_value.append(self._dump_filter_value(filter_el, period_key))
        elif isinstance(filter_value, dict):
            result_value = dict()
            for filter_key, filter_el in filter_value.items():
                period_key = period_key or filter_key == 'period'
                result_value[filter_key] = self._dump_filter_value(filter_el, period_key)
        elif period_key:
            result_value = datetime.strptime(filter_value, '%Y-%m-%dT%H:%M:%S')
        else:
            result_value = filter_value

        return result_value


class ModelInfo(base_api_types.ModelInfo):
    """
    Model info response (model info response)
    """
    fitting_status: model_types.FittingStatuses
    fitting_date: Optional[datetime] = None
    fitting_start_date: Optional[datetime] = None
    fitting_error_text: str = ''
    fitting_error_date: Optional[datetime] = None
    fitting_job_id: str = ''
    fitting_job_pid: int = 0

    x_indicators: list[IndicatorRow]
    y_indicators: list[IndicatorRow]

    categorical_features: Optional[list[str]] = None

    x_columns: list[str] = []
    y_columns: list[str] = []

    categorical_columns: list[str] = []

    fi_status: model_types.FittingStatuses
    fi_calculation_date: Optional[datetime] = None
    fi_calculation_start_date: Optional[datetime] = None
    fi_calculation_error_text: str = ''
    fi_calculation_error_date: Optional[datetime] = None
    fi_calculation_job_id: str = ''
    fi_calculation_job_pid: int = 0


class FittingParameters(base_api_types.FittingParameters):
    """
    Fitting parameters of model
    """
    epochs: int = 0
    filter: Optional[dict[str, Any]] = None
    need_to_x_scaling: bool = True
    need_to_y_scaling: bool = True
