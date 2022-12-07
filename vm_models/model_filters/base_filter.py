
from typing import Any

class FittingFilter:
    service_name: str = ''
    def __init__(self, filter_value: dict[str, Any], from_model: bool = False):
        self._value: dict[str, Any] = filter_value