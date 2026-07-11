import numpy as np
# 1. Ellipsoid (High Conditioned Elliptic Function - CEC F3 Definition)
class EllipsoidFunction:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -5.12 * np.ones(dim)
        self.ub = 5.12 * np.ones(dim)
        # 使用標準線性係數 (1, 2, ..., d) 
        self.coeffs = np.arange(1, dim + 1)
    def __call__(self, x):
        return np.sum(self.coeffs * (x ** 2))

# 2. Rosenbrock (Shifted Rotated Rosenbrock - CEC F6 Definition)
class RosenbrockFunction:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -2.048 * np.ones(dim)
        self.ub = 2.048 * np.ones(dim)

    def __call__(self, x):

        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)

# 3. Ackley (Shifted Rotated Ackley - CEC F8 Definition)
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

# 4. Griewank (Shifted Rotated Griewank - CEC F7 Definition)
class GriewankFunction:
    def __init__(self, dim):
        self.dim = dim
        self.lb = -600.0 * np.ones(dim)
        self.ub = 600.0 * np.ones(dim)

    def __call__(self, x):

        sum_sq = np.sum(x**2) / 4000.0
        cos_prod = np.prod(np.cos(x / np.sqrt(np.arange(1, self.dim + 1))))
        return sum_sq - cos_prod + 1.0
    