""" Module contains VBM class for saving and getting fitting parameters of model """

from typing import Any, Optional
from dataclasses import dataclass, fields
from copy import deepcopy
from datetime import datetime

from .base_parameters import ModelParameters, FittingParameters
from id_generator import IdGenerator

__all__ = ['VbmModelParameters', 'VbmFittingParameters']

@dataclass
class VbmModelParameters(ModelParameters):

    service_name: str = 'vbm'
    _x_indicators: list[dict[str, Any]] = None
    _y_indicators: list[dict[str, Any]] = None

    rsme: int = 0
    mspe: int = 0

    def __init__(self):
        super().__init__()

    def __post_init__(self):
        super().__post_init__()
        self._x_indicators = []
        self._y_indicators = []

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:

        super().set_all(parameters)

        self._check_new_parameters(parameters)

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields =  [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_parameters = [el[1:] if el.startswith('_') else el for el in self_fields]

        for count, par_name in enumerate(self_parameters):
            name = self_fields[count] if without_processing else par_name
            if par_name in parameters:
                setattr(self, name, parameters[par_name])

    def get_all(self) -> dict[str, Any]:
        result = super().get_all()

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields = [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_fields = [el[1:] if el.startswith('_') else el for el in self_fields]

        parameters_to_add = {name: getattr(self, name) for name in self_fields}

        result.update(parameters_to_add)

        return result

    def _check_new_parameters(self, parameters: dict[str, Any], checking_names: Optional[list] = None) -> None:

        super()._check_new_parameters(parameters, checking_names)
        if not checking_names:
            super()._check_new_parameters(parameters, ['x_indicators', 'y_indicators'])

    @property
    def x_indicators(self):
        return self._x_indicators

    @property
    def y_indicators(self):
        return self._y_indicators

    @x_indicators.setter
    def x_indicators(self, value):
        x_indicators = deepcopy(value)

        for el in x_indicators:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._x_indicators = x_indicators

    @y_indicators.setter
    def y_indicators(self, value):
        y_indicators = deepcopy(value)

        for el in y_indicators:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._y_indicators = y_indicators


@dataclass
class VbmFittingParameters(FittingParameters):
    service_name = 'vbm'

    _x_analytics: Optional[list[dict[str, Any]]] = None
    _y_analytics: Optional[list[dict[str, Any]]] = None

    _x_analytic_keys: Optional[list[dict[str, Any]]] = None
    _y_analytic_keys: Optional[list[dict[str, Any]]] = None

    def __post_init__(self):

        super().__post_init__()

        self._x_analytics = []
        self._y_analytics = []

        self._x_analytic_keys = []
        self._y_analytic_keys = []

    def set_all(self, parameters: dict[str, Any], without_processing: bool = False) -> None:
        super().set_all(parameters)

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields =  [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_parameters = [el[1:] if el.startswith('_') else el for el in self_fields]

        for count, par_name in enumerate(self_parameters):
            name = self_fields[count] if without_processing else par_name
            if par_name in parameters:
                setattr(self, name, parameters[par_name])

    def get_all(self) -> dict[str, Any]:
        result = super().get_all()

        super_fields = [el.name for el in fields(super()) if el.name not in ['service_name']]
        self_fields = [el.name for el in fields(self) if el.name not in super_fields + ['service_name']]
        self_fields = [el[1:] if el.startswith('_') else el for el in self_fields]

        parameters_to_add = {name: getattr(self, name) for name in self_fields}

        result.update(parameters_to_add)

        return result

    def set_drop_fitting(self):
        super().set_drop_fitting()

        self._x_analytics = []
        self._y_analytics = []

        self._x_analytic_keys = []
        self._y_analytic_keys = []


    def set_error_fitting(self, first_fitting:bool = False):
        super().set_error_fitting(first_fitting)

        if first_fitting:
            self._x_analytics = []
            self._y_analytics = []

            self._x_analytic_keys = []
            self._y_analytic_keys = []

    @property
    def x_analytics(self):
        return self._x_analytics

    @property
    def y_analytics(self):
        return self._y_analytics

    @x_analytics.setter
    def x_analytics(self, value):
        x_analytics = deepcopy(value)

        for el in x_analytics:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._x_analytics = x_analytics

    @y_analytics.setter
    def y_analytics(self, value):
        y_analytics = deepcopy(value)

        for el in y_analytics:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_dict_id_type(el)

        self._y_analytics = y_analytics

    @property
    def x_analytic_keys(self):
        return self._x_analytic_keys

    @property
    def y_analytic_keys(self):
        return self._y_analytic_keys

    @x_analytic_keys.setter
    def x_analytic_keys(self, value):
        x_analytic_keys = deepcopy(value)

        for el in x_analytic_keys:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_list_of_dict_short_id(el['analytics'])

        self._x_analytic_keys = x_analytic_keys

    @y_analytic_keys.setter
    def y_analytic_keys(self, value):
        y_analytic_keys = deepcopy(value)

        for el in y_analytic_keys:
            if 'short_id' not in el:
                el['short_id'] = IdGenerator.get_short_id_from_list_of_dict_short_id(el['analytics'])

        self._y_analytic_keys = y_analytic_keys