class RBF:
    """
    Cubic Radial Basis Function (RBF) surrogate model with X & Y normalization.
    Normalization ensures numerical stability and scale invariance across different functions.
    """
    def __init__(self):
        self.centers_norm = None
        self.weights = None
        self.lb_X = None
        self.range_X = None
        self.mean_y = None
        self.std_y = None
        self.cond_phi = None
        self.cond_phi_reg = None
        self.rmse = None
        
    def fit(self, X, y, use_log=True):
        N, d = X.shape
        self.lb_X = np.min(X, axis=0)
        self.ub_X = np.max(X, axis=0)
        self.range_X = self.ub_X - self.lb_X
        # Handle zero-range dimensions
        self.range_X[self.range_X < 1e-8] = 1.0
        
        X_norm = (X - self.lb_X) / self.range_X
        self.centers_norm = np.copy(X_norm)
        
        # -----------------------------------------------------------------
        # 萬能防禦 1：在 fit 結束前，記錄本次真實 Y（未經過 Sign-Log 轉換前）的真實尺度
        # -----------------------------------------------------------------
        self.y_raw_min = np.min(y)
        self.y_raw_max = np.max(y)
        self.y_raw_range = self.y_raw_max - self.y_raw_min
        
        # 避免所有點的值都一樣導致 range 為 0
        if self.y_raw_range < 1e-8:
            self.y_raw_range = 1.0

        # 儲存目前的模式狀態，供 predict 使用
        self.use_log = use_log

        if self.use_log:
            # =====================================================================
            # 階段 A：對數空間 (防禦外推爆炸)
            # =====================================================================
            y_log = np.sign(y) * np.log10(np.abs(y) + 1.0)
            self.mean_y = np.mean(y_log)
            self.std_y = np.std(y_log)
            if self.std_y < 1e-8:
                self.std_y = 1.0
            y_norm = (y_log - self.mean_y) / self.std_y
        else:
            # =====================================================================
            # 階段 B：原始空間 (釋放狼性梯度與強烈 Reward)
            # =====================================================================
            self.mean_y = np.mean(y)
            self.std_y = np.std(y)
            if self.std_y < 1e-8:
                self.std_y = 1.0
            y_norm = (y - self.mean_y) / self.std_y
        
        if N < 2:
            self.weights = np.copy(y_norm)
            self.cond_phi = 1.0
            self.cond_phi_reg = 1.0
            self.rmse = 0.0
            return
            
        dists = cdist(X_norm, X_norm, 'euclidean')
        Phi = dists ** 3
        
        try:
            self.cond_phi = np.linalg.cond(Phi)
        except Exception:
            self.cond_phi = np.nan
            
        try:
            self.cond_phi_reg = np.linalg.cond(Phi + 1e-6 * np.eye(N))
        except Exception:
            self.cond_phi_reg = np.nan
            
        # Add regularization (1e-6) for numerical stability during pinv
        self.weights = np.linalg.pinv(Phi + 1e-6 * np.eye(N)) @ y_norm
        
        # Calculate training RMSE
        pred_norm = Phi @ self.weights
        pred_log = pred_norm * self.std_y + self.mean_y
        
        # =====================================================================
        # 安全裁切：限制在 [-20, 20]，防止還原時 10** 發生 overflow
        # =====================================================================
        pred_log_clipped = np.clip(pred_log, -20, 20)
        # 反向還原 Sign-Log
        pred = np.sign(pred_log_clipped) * (10**np.abs(pred_log_clipped) - 1.0)
        self.rmse = np.sqrt(np.mean((pred - y)**2))
        
    def predict(self, X):
        if X.ndim == 1:
            X_input = X.reshape(1, -1)
        else:
            X_input = X
            
        # Normalize input
        X_norm = (X_input - self.lb_X) / self.range_X
        
        dists = cdist(X_norm, self.centers_norm, 'euclidean')
        Phi = dists ** 3
        pred_norm = Phi @ self.weights
        
        # =====================================================================
        # ✨ 修正後的反標準化與還原邏輯
        # =====================================================================
        if self.use_log:
            # 如果訓練時用對數空間，才需要進行 Sign-Log 安全裁切與還原
            pred_log = pred_norm * self.std_y + self.mean_y
            pred_log_clipped = np.clip(pred_log, -20, 20)
            pred = np.sign(pred_log_clipped) * (10**np.abs(pred_log_clipped) - 1.0)
        else:
            # 如果訓練時是原始空間，直接乘回標準差並加上平均值即可！
            pred = pred_norm * self.std_y + self.mean_y
        # =====================================================================
        
        # =====================================================================
        # 💡 【就在這裡加入：動態下限防禦】
        # 允許預測出負數，但最誇張只能比當前真實資料的最小值再低 0.5 倍的範圍。
        # =====================================================================
        
        lower_bound = self.y_raw_min - 0.5 * self.y_raw_range
        pred = np.maximum(pred, lower_bound)
        
        # =====================================================================
        
        
        if X.ndim == 1:
            return pred[0]
        return pred
import numpy as np
from scipy.optimize import differential_evolution
