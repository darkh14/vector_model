
from typing import Any

__all__ = ['FittingFilter']

class FittingFilter:
    service_name: str = ''
    def __init__(self, filter_value: dict[str, Any], for_model: bool = False) -> None:

        self._value: dict[str, Any] = filter_value
        self._for_model: bool = for_model

    def get_value(self):
        return self._value

    def get_value_for_db(self):
        return self._value