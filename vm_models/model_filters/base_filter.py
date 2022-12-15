
from typing import Any

__all__ = ['FittingFilter']

class FittingFilter:
    service_name: str = ''
    def __init__(self, filter_value: bytes| dict[str, Any], for_model: bool = False) -> None:

        self._value: bytes | dict[str, Any] = filter_value
        self._for_model: bool = for_model

    def get_value_as_model_parameter(self):
        return self._value

    def get_value_as_db_filter(self):
        return self._value

    def get_value_as_json_serializable(self):
        return self._value