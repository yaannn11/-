from .basic_functions import *
from .cec_functions import *
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
