
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


def _generate_cec_100d_data(seed):
    dim = 100
    rng = np.random.default_rng(seed)
    o = rng.uniform(-5, 5, dim)                
    A = rng.standard_normal((dim, dim))
    M, _ = np.linalg.qr(A)                   
    return o, M

# 5. SRR (Shifted Rotated Rastrigin - CEC F10 Definition)

class SRRFunction:
    """CEC 2005 - F10: Shifted Rotated Rastrigin's Function (SRR) - 100D ONLY"""
    def __init__(self, dim: int = 100):
        self.dim: int = 100 
        self.lb: np.ndarray = -5.0 * np.ones(self.dim)
        self.ub: np.ndarray = 5.0 * np.ones(self.dim)
        self.f_bias = -330.0
        

        self.o, self.M = _generate_cec_100d_data(seed=200510)
        
    def __call__(self, x: np.ndarray) -> float:

        z = np.dot(x - self.o, self.M)

        f = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10) + self.f_bias
        return float(f)


# 6. RHC1 (Rotated Hybrid Composition 1 - CEC F16 Definition Inspired)
class RHC1Function:
    """CEC 2005 - F16: Rotated Hybrid Composition Function (RHC1) - 100D ONLY"""
    def __init__(self, dim: int = 100):
        self.dim: int = 100  
        self.lb: np.ndarray = -5.0 * np.ones(self.dim)
        self.ub: np.ndarray = 5.0 * np.ones(self.dim)
        self.f_bias = 120.0 
        self.num_funcs = 10  
        
  
        self.o_list = []
        self.M_list = []
        for i in range(self.num_funcs):
            o, M = _generate_cec_100d_data(seed=200516 + i)
            self.o_list.append(o)
            self.M_list.append(M)
            
        self.sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.lambda_i = np.array([1.0, 1.0, 10.0, 10.0, 5/60, 5/60, 5/32, 5/32, 5/100, 5/100])
        self.f_max = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        
    def __call__(self, x: np.ndarray) -> float:
        w = np.zeros(self.num_funcs)
        fit = np.zeros(self.num_funcs)
        
        for i in range(self.num_funcs):
            w[i] = np.exp(-np.sum((x - self.o_list[i])**2) / (2 * self.dim * (self.sigma[i]**2)))
            z = np.dot(x - self.o_list[i], self.M_list[i]) / self.lambda_i[i]
            fit[i] = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
            
        w_max = np.max(w)
        if w_max == 0: w = np.ones(self.num_funcs)
        w = np.where((w != w_max) & (w < 1e-15), 0, w)
        sum_w = np.sum(w)
        w = w / sum_w
        
        res = np.sum(w * (fit / self.f_max)) + self.f_bias
        return float(res)

class RHC2Function:
    """CEC 2005 - F19: Rotated Hybrid Composition Function with narrow basin (RHC2) - 100D ONLY"""
    def __init__(self, dim: int = 100):
        self.dim: int = 100 
        self.lb: np.ndarray = -5.0 * np.ones(self.dim)
        self.ub: np.ndarray = 5.0 * np.ones(self.dim)
        self.f_bias = 10.0  
        self.num_funcs = 10
        
        self.o_list = []
        self.M_list = []
        for i in range(self.num_funcs):
            o, M = _generate_cec_100d_data(seed=200519 + i)
            self.o_list.append(o)
            self.M_list.append(M)
            
        self.sigma = np.array([0.1, 2.0, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0])
        self.lambda_i = np.array([0.1, 1.0, 10.0, 10.0, 5/60, 5/60, 5/32, 5/32, 5/100, 5/100])
        self.f_max = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        
    def __call__(self, x: np.ndarray) -> float:
        w = np.zeros(self.num_funcs)
        fit = np.zeros(self.num_funcs)
        
        for i in range(self.num_funcs):
            w[i] = np.exp(-np.sum((x - self.o_list[i])**2) / (2 * self.dim * (self.sigma[i]**2)))
            z = np.dot(x - self.o_list[i], self.M_list[i]) / self.lambda_i[i]
            fit[i] = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
            
        w_max = np.max(w)
        if w_max == 0: w = np.ones(self.num_funcs)
        w = np.where((w != w_max) & (w < 1e-15), 0, w)
        sum_w = np.sum(w)
        w = w / sum_w
        
        res = np.sum(w * (fit / self.f_max)) + self.f_bias
        return float(res)

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
