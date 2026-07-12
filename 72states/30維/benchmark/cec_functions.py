import numpy as np
from opfunu.cec_based.cec2005 import F102005, F162005, F192005
# 5. SRR (Shifted Rotated Rastrigin - CEC F10 Definition)
class SRRFunction:
    """CEC 2005 - F10: Shifted Rotated Rastrigin's Function (SRR)"""
    def __init__(self, dim: int):
        self.dim: int = dim
        self.lb: np.ndarray = -5.0 * np.ones(dim)
        self.ub: np.ndarray = 5.0 * np.ones(dim)
        self._raw_func = F102005(ndim=dim)
        
    def __call__(self, x: np.ndarray) -> float:
        return float(self._raw_func.evaluate(x))

# 6. RHC1 (Rotated Hybrid Composition 1 - CEC F15 Definition Inspired)
class RHC1Function:
    """CEC 2005 - F16: Rotated Hybrid Composition Function (RHC1)"""
    def __init__(self, dim: int):
        self.dim: int = dim
        self.lb: np.ndarray = -5.0 * np.ones(dim)
        self.ub: np.ndarray = 5.0 * np.ones(dim)
        self._raw_func = F162005(ndim=dim)
        
    def __call__(self, x: np.ndarray) -> float:
        return float(self._raw_func.evaluate(x))

# 7. RHC2 (Rotated Hybrid Composition 2 - CEC F16 Definition Inspired)
class RHC2Function:
    """CEC 2005 - F19: Rotated Hybrid Composition Function with narrow basin global optimum (RHC2)"""
    def __init__(self, dim: int):
        self.dim: int = dim
        self.lb: np.ndarray = -5.0 * np.ones(dim)
        self.ub: np.ndarray = 5.0 * np.ones(dim)
        self._raw_func = F192005(ndim=dim)
        
    def __call__(self, x: np.ndarray) -> float:
        return float(self._raw_func.evaluate(x))
