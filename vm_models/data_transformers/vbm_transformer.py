
from typing import Any, Optional
import pandas as pd

from .base_transformer import RowColumnTransformer

class VbmRowColumnTransformer(RowColumnTransformer):

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:

        return x
