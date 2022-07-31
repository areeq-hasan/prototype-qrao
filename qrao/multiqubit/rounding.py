from typing import List

import numpy as np


def semideterministic_round(variable_values: List[float]) -> List[int]:
    return [
        (0 if (value > 0) else 1) if not np.isclose(0, value) else np.random.randint(2)
        for value in variable_values
    ]
