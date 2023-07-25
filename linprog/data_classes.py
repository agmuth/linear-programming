from dataclasses import dataclass

import numpy as np


@dataclass
class LinProgResult:
    x: np.array
    basis: np.array
    cost: float
    iters: int
    optimum: bool
