import numpy as np
from typing import Union, NamedTuple, Tuple, Dict, Any


class EvalPrediction(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    data_info: Dict[str, Any]
