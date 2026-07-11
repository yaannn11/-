import numpy as np

# =====================================================================
# 輔助函式：固定種子生成 100 維所需的平移向量與正交旋轉矩陣
# =====================================================================
def _generate_cec_100d_data(seed):
    dim = 100
    rng = np.random.default_rng(seed)
    o = rng.uniform(-5, 5, dim)                 # 平移向量
    A = rng.standard_normal((dim, dim))
    M, _ = np.linalg.qr(A)                      # QR分解確保 M 是完美的正交旋轉矩陣
    return o, M

# =====================================================================
# 5. SRR (Shifted Rotated Rastrigin - CEC F10 Definition)
# =====================================================================
class SRRFunction:
    """CEC 2005 - F10: Shifted Rotated Rastrigin's Function (SRR) - 100D ONLY"""
    def __init__(self, dim: int = 100):
        self.dim: int = 100  # 強制固定 100 維
        self.lb: np.ndarray = -5.0 * np.ones(self.dim)
        self.ub: np.ndarray = 5.0 * np.ones(self.dim)
        self.f_bias = -330.0
        
        # 初始化 100 維的平移與旋轉矩陣
        self.o, self.M = _generate_cec_100d_data(seed=200510)
        
    def __call__(self, x: np.ndarray) -> float:
        # 1. 平移與旋轉變換
        z = np.dot(x - self.o, self.M)
        # 2. Rastrigin 標準公式計算
        f = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10) + self.f_bias
        return float(f)

# =====================================================================
# 6. RHC1 (Rotated Hybrid Composition 1 - CEC F16 Definition Inspired)
# =====================================================================
class RHC1Function:
    """CEC 2005 - F16: Rotated Hybrid Composition Function (RHC1) - 100D ONLY"""
    def __init__(self, dim: int = 100):
        self.dim: int = 100  # 強制固定 100 維
        self.lb: np.ndarray = -5.0 * np.ones(self.dim)
        self.ub: np.ndarray = 5.0 * np.ones(self.dim)
        self.f_bias = 120.0  # F16 官方理論最低點
        self.num_funcs = 10  # 混合 10 個子函數
        
        # 分別為 10 個基礎函數生成各自的 100 維旋轉/平移矩陣
        self.o_list = []
        self.M_list = []
        for i in range(self.num_funcs):
            o, M = _generate_cec_100d_data(seed=200516 + i)
            self.o_list.append(o)
            self.M_list.append(M)
            
        # 官方 F16 的混合權重控制參數
        self.sigma = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        self.lambda_i = np.array([1.0, 1.0, 10.0, 10.0, 5/60, 5/60, 5/32, 5/32, 5/100, 5/100])
        self.f_max = np.array([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        
    def __call__(self, x: np.ndarray) -> float:
        w = np.zeros(self.num_funcs)
        fit = np.zeros(self.num_funcs)
        
        for i in range(self.num_funcs):
            # 計算各子函數的距離權重
            w[i] = np.exp(-np.sum((x - self.o_list[i])**2) / (2 * self.dim * (self.sigma[i]**2)))
            # 子函數計算
            z = np.dot(x - self.o_list[i], self.M_list[i]) / self.lambda_i[i]
            fit[i] = np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
            
        w_max = np.max(w)
        if w_max == 0: w = np.ones(self.num_funcs)
        w = np.where((w != w_max) & (w < 1e-15), 0, w)
        sum_w = np.sum(w)
        w = w / sum_w
        
        res = np.sum(w * (fit / self.f_max)) + self.f_bias
        return float(res)

# =====================================================================
# 7. RHC2 (Rotated Hybrid Composition 2 - CEC F16 Definition Inspired)
# =====================================================================
class RHC2Function:
    """CEC 2005 - F19: Rotated Hybrid Composition Function with narrow basin (RHC2) - 100D ONLY"""
    def __init__(self, dim: int = 100):
        self.dim: int = 100  # 強制固定 100 維
        self.lb: np.ndarray = -5.0 * np.ones(self.dim)
        self.ub: np.ndarray = 5.0 * np.ones(self.dim)
        self.f_bias = 10.0   # F19 官方理論最低點
        self.num_funcs = 10
        
        self.o_list = []
        self.M_list = []
        for i in range(self.num_funcs):
            o, M = _generate_cec_100d_data(seed=200519 + i)
            self.o_list.append(o)
            self.M_list.append(M)
            
        # F19 的特色：擁有較窄的全局最優解谷底（Narrow basin）
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