""" Module contains class for saving and getting fitting parameters of model (filter etc.)"""

from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class FittingParameters:
    model_filter: Optional[dict[str, Any]] = None

    def __post_init__(self):
        pass


