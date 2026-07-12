# benchmark_fn.py
import numpy as np
from opfunu.cec_based.cec2005 import F102005, F162005, F192005

class EllipsoidFunction:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        self.coeffs = np.arange(1, dim + 1)
    def __call__(self, x):
        return np.sum(self.coeffs * (x ** 2))

# 2. Rosenbrock 
class RosenbrockFunction:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -2.048 * np.ones(dim)
        self.ub = 2.048 * np.ones(dim)

    def __call__(self, x):

        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)

# 3. Ackley 
class AckleyFunction:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -32.768 * np.ones(dim)
        self.ub = 32.768 * np.ones(dim)

    def __call__(self, x):

        d = self.dim
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return term1 + term2 + 20.0 + np.e

# 4. Griewank 
class GriewankFunction:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -600.0 * np.ones(dim)
        self.ub = 600.0 * np.ones(dim)

    def __call__(self, x):

        sum_sq = np.sum(x**2) / 4000.0
        cos_prod = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))))
        return sum_sq - cos_prod + 1.0

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

def get_benchmark_function(name, dim):
    name_lower = name.lower()
    if name_lower == 'ellipsoid':
        return EllipsoidFunction(dim)
    elif name_lower == 'rosenbrock':
        return RosenbrockFunction(dim)
    elif name_lower == 'ackley':
        return AckleyFunction(dim)
    elif name_lower == 'griewank':
        return GriewankFunction(dim)
    elif name_lower == 'srr':
        return SRRFunction(dim)
    elif name_lower == 'rhc1':
        return RHC1Function(dim)
    elif name_lower == 'rhc2':
        return RHC2Function(dim)
    else:
        raise ValueError(f"Unknown function: {name}")
