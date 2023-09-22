""" VBM (Vector budget model)
    Module contains VBM classes for saving and getting fitting parameters of model
    Classes:
        VbmModelParameters - for model parameters
        VbmFittingParameters - for fitting parameters
"""

from typing import Any, Optional, ClassVar
from dataclasses import dataclass
from datetime import datetime
import os

from vm_logging.exceptions import ModelException
from .base_parameters import ModelParameters, FittingParameters
from .. import model_types

__all__ = ['VbmModelParameters', 'VbmFittingParameters']


@dataclass
class VbmModelParameters(ModelParameters):
    """ Dataclass for storing, saving and getting model parameters. Inherited by ModelParameters class
        Methods:
            set_all - to set all input parameters in object
            get_all - to get all parameters
            _check_new_parameters - checks parameters
        Properties:
            x_indicators
            y_indicators
    """
    service_name: ClassVar[str] = 'vbm'

    x_indicators: Optional[list[dict[str, Any]]] = None
    y_indicators: Optional[list[dict[str, Any]]] = None

    categorical_features: Optional[list[str]] = None

    def __post_init__(self) -> None:
        """
        Defines x, y indicators as empty lists
        """
        super().__post_init__()
        self.x_indicators = []
        self.y_indicators = []

        self.categorical_features = []

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        """
        For setting all parameters, defined in "parameters" parameter
        :param parameters: input parameters to set
        :param without_processing: no need to convert parameters if True
        """
        super().set_all(parameters)

        self.x_indicators = parameters['x_indicators']
        self.y_indicators = parameters['y_indicators']

        self.categorical_features = parameters.get('categorical_features')

    def get_all(self) -> dict[str, Any]:
        """
        For getting values of all parameters
        :return: dict of values of all parameters
        """
        result = super().get_all()

        result['x_indicators'] = self.x_indicators
        result['y_indicators'] = self.y_indicators

        result['categorical_features'] = self.categorical_features

        return result


@dataclass
class VbmFittingParameters(FittingParameters):
    """ Dataclass for storing, saving and getting fitting parameters
        Methods:
            set_all - to set all input parameters in object
            get_all - to get all parameters
            set_start_fitting - to set statuses and other parameters before starting fitting
            set_drop_fitting - to set statuses and other parameters when dropping fitting
            set_error_fitting - to set statuses and other parameters when error is while fitting
            set_start_fi_calculation - to set statuses and other parameters before starting fi calculation
            set_end_fi_calculation - to set statuses and other parameters after finishing fi calculation
            set_error_fi_calculation - to set statuses and other parameters when error is while fi calculating
            set_drop_fi_calculation - to set statuses and other parameters when dropping fi calculation
     """
    service_name: ClassVar[str] = 'vbm'

    fi_status: model_types.FittingStatuses = model_types.FittingStatuses.NotFit

    fi_calculation_date: Optional[datetime] = None
    fi_calculation_start_date: Optional[datetime] = None
    fi_calculation_error_date: Optional[datetime] = None

    fi_calculation_error_text: str = ''

    fi_calculation_job_id: str = ''
    fi_calculation_job_pid: int = 0

    def __post_init__(self) -> None:
        """
        Converts _x_analytics, _y_analytics, _x_analytic_keys, _y_analytic_keys to empty lists to
        empty dict
        """
        super().__post_init__()

    def set_all(self, parameters: dict[str, Any]) -> None:
        """
        For setting all parameters, defined in "parameters" parameter
        :param parameters: input parameters to set
        """
        super().set_all(parameters)

        self.fi_status = parameters.get('fi_status') or model_types.FittingStatuses.NotFit

        self.fi_calculation_date = parameters.get('fi_calculation_date')
        self.fi_calculation_start_date = parameters.get('fi_calculation_start_date')
        self.fi_calculation_error_date = parameters.get('fi_calculation_error_date')

        self.fi_calculation_error_text = parameters.get('fi_calculation_error_text', '')

        self.fi_calculation_job_id = parameters.get('fi_calculation_job_id', '')
        self.fi_calculation_job_pid = parameters.get('fi_calculation_job_pid', 0)

    def get_all(self) -> dict[str, Any]:
        """
        For getting values of all parameters
        :return: dict of values of all parameters
        """
        result = super().get_all()

        fi_parameters = {
            'fi_status': self.fi_status,

            'fi_calculation_date': self.fi_calculation_date,
            'fi_calculation_start_date': self.fi_calculation_start_date,
            'fi_calculation_error_date': self.fi_calculation_error_date,

            'fi_calculation_error_text': self.fi_calculation_error_text,
            'fi_calculation_job_id': self.fi_calculation_job_id,
            'fi_calculation_job_pid': self.fi_calculation_job_pid
        }

        result.update(fi_parameters)

        return result

    def set_start_fitting(self, job_id: str = '') -> None:
        """
        For setting statuses and other parameters before starting fitting
        :param job_id: id of job if fitting in background
        """
        super().set_start_fitting(job_id)

        if self.fi_status in (model_types.FittingStatuses.Fit,
                              model_types.FittingStatuses.Started,
                              model_types.FittingStatuses.PreStarted,
                              model_types.FittingStatuses.Error):
            self.set_drop_fi_calculation()

    def set_drop_fitting(self, model_id='') -> None:
        """
        For setting statuses and other parameters when dropping fitting
        """
        super().set_drop_fitting()

        if self.fi_status in (model_types.FittingStatuses.Fit,
                              model_types.FittingStatuses.Started,
                              model_types.FittingStatuses.PreStarted,
                              model_types.FittingStatuses.Error):
            self.set_drop_fi_calculation()

    def set_error_fitting(self, error_text: str = '') -> None:
        """
        For setting statuses and other parameters when error is while fitting
        :param error_text: text of fitting error
        """
        super().set_error_fitting(error_text)

    def set_start_fi_calculation(self, job_id: str) -> None:
        """
        For setting statuses and other parameters before starting fi calculation
        :param job_id: id of job if fi calculation is in background
        """
        self.fi_status = model_types.FittingStatuses.Started

        self.fi_calculation_date = None
        self.fi_calculation_start_date = datetime.utcnow()
        self.fi_calculation_error_date = None

        self.fi_calculation_error_text = ''

        self.fi_calculation_job_pid = os.getpid()

        if job_id:
            self.fi_calculation_job_id = job_id

    def set_pre_start_fi_calculation(self, job_id: str) -> None:
        """
        For setting statuses and other parameters before starting fi calculation
        :param job_id: id of background job if fi calculation is in background
        """
        self.fi_status = model_types.FittingStatuses.PreStarted

        self.fi_calculation_date = None
        self.fi_calculation_start_date = datetime.utcnow()
        self.fi_calculation_error_date = None

        self.fi_calculation_error_text = ''

        self.fi_calculation_job_pid = os.getpid()

        if job_id:
            self.fi_calculation_job_id = job_id

    def set_end_fi_calculation(self) -> None:
        """
        For setting statuses and other parameters after finishing fi calculation
        """
        if self.fi_status != model_types.FittingStatuses.Started:
            raise ModelException('Can not finish fi calculation. Fi calculation is not started. ' +
                                 'Start fi calculation before')

        self.fi_status = model_types.FittingStatuses.Fit

        self.fi_calculation_date = datetime.utcnow()
        self.fi_calculation_error_date = None

    def set_error_fi_calculation(self, error_text) -> None:
        """
        For setting statuses and other parameters when error is while fi calculating
        :param error_text: text of fitting error
        """
        self.fi_status = model_types.FittingStatuses.Error

        self.fi_calculation_date = None
        self.fi_calculation_error_date = datetime.utcnow()

        self.fi_calculation_error_text = error_text

    def set_drop_fi_calculation(self) -> None:
        """
        For setting statuses and other parameters when dropping fi calculation
        """
        if self.fi_status not in (model_types.FittingStatuses.Fit,
                                  model_types.FittingStatuses.Started,
                                  model_types.FittingStatuses.PreStarted,
                                  model_types.FittingStatuses.Error):
            raise ModelException('Can not drop fi calculation. FI is not calculated')

        self.fi_status = model_types.FittingStatuses.NotFit

        self.fi_calculation_date = None
        self.fi_calculation_start_date = None
        self.fi_calculation_error_date = None

        self.fi_calculation_error_text = ''

        self.fi_calculation_job_id = ''
        self.fi_calculation_job_pid = 0
