import numpy as np
from scipy.spatial.distance import cdist

class RBF:
    """
    自適應徑向基底函數 (RBF) 代理模型 (Cubic 與 Gaussian 自動選取)
    """
    def __init__(self, kernel_mode='adaptive'):
        self.kernel_mode = kernel_mode
        self.selected_kernel = None
        self.centers_norm = None
        self.weights = None
        self.lb_X = None
        self.range_X = None
        self.mean_y = None
        self.std_y = None
        self.beta = None
        self.rmse = None
        
    def fit(self, X, y, use_log=False):
        """
        擬合 RBF 模型，包含輸入與輸出的正規化，並自適應選擇核心。
        """
        N, d = X.shape
        # 1. X 正規化到 [0, 1] 區間
        self.lb_X = np.min(X, axis=0)
        self.ub_X = np.max(X, axis=0)
        self.range_X = self.ub_X - self.lb_X
        self.range_X[self.range_X < 1e-8] = 1.0  # 防止除以零
        X_norm = (X - self.lb_X) / self.range_X
        self.centers_norm = np.copy(X_norm)
        
        # 2. y 記錄與正規化
        self.y_raw_min = np.min(y)
        self.y_raw_max = np.max(y)
        self.y_raw_range = max(1e-8, self.y_raw_max - self.y_raw_min)
        
        self.use_log = use_log
        if self.use_log:
            # 對數變換 (防止外推爆炸)
            y_log = np.sign(y) * np.log10(np.abs(y) + 1.0)
            self.mean_y = np.mean(y_log)
            self.std_y = max(1e-8, np.std(y_log))
            y_norm = (y_log - self.mean_y) / self.std_y
        else:
            self.mean_y = np.mean(y)
            self.std_y = max(1e-8, np.std(y))
            y_norm = (y - self.mean_y) / self.std_y
            
        if N < 2:
            self.weights = np.copy(y_norm)
            self.selected_kernel = 'cubic'
            self.rmse = 0.0
            return
            
        # 3. 計算兩點間的距離矩陣
        dists = cdist(X_norm, X_norm, 'euclidean')
        
        # 4. 自適應核心選取機制 (Rippa LOOCV 誤差預估)
        if self.kernel_mode == 'cubic':
            self.selected_kernel = 'cubic'
        elif self.kernel_mode == 'gaussian':
            self.selected_kernel = 'gaussian'
        else:  # 'adaptive'
            # 4.1 計算 Cubic 核心的 LOOCV 誤差
            Phi_c = dists ** 3
            try:
                InvPhi_c = np.linalg.pinv(Phi_c + 1e-6 * np.eye(N))
                w_c = InvPhi_c @ y_norm
                diag_c = np.diag(InvPhi_c)
                diag_c_safe = np.where(np.abs(diag_c) > 1e-12, diag_c, 1e-12)
                loocv_err_cubic = np.mean((w_c / diag_c_safe) ** 2)
            except Exception:
                loocv_err_cubic = float('inf')
                
            # 4.2 計算 Gaussian 核心的 LOOCV 誤差
            D_max = max(1e-8, np.max(dists))
            beta_val = D_max * (d * N) ** (-1.0 / d)
            Phi_g = np.exp(-(dists ** 2) / beta_val)
            try:
                InvPhi_g = np.linalg.pinv(Phi_g + 1e-6 * np.eye(N))
                w_g = InvPhi_g @ y_norm
                diag_g = np.diag(InvPhi_g)
                diag_g_safe = np.where(np.abs(diag_g) > 1e-12, diag_g, 1e-12)
                loocv_err_gauss = np.mean((w_g / diag_g_safe) ** 2)
            except Exception:
                loocv_err_gauss = float('inf')
                
            # 4.3 選擇誤差較小的核心
            if loocv_err_cubic <= loocv_err_gauss:
                self.selected_kernel = 'cubic'
            else:
                self.selected_kernel = 'gaussian'
                
        # 5. 使用選定的核心進行最終 RBF 擬合
        if self.selected_kernel == 'cubic':
            Phi = dists ** 3
        else:
            D_max = max(1e-8, np.max(dists))
            self.beta = D_max * (d * N) ** (-1.0 / d)
            Phi = np.exp(-(dists ** 2) / self.beta)
            
        self.weights = np.linalg.pinv(Phi + 1e-6 * np.eye(N)) @ y_norm
        
        # 6. 計算訓練誤差 RMSE
        pred_norm = Phi @ self.weights
        if self.use_log:
            pred_log = pred_norm * self.std_y + self.mean_y
            pred_log_clipped = np.clip(pred_log, -20, 20)
            pred = np.sign(pred_log_clipped) * (10**np.abs(pred_log_clipped) - 1.0)
        else:
            pred = pred_norm * self.std_y + self.mean_y
        self.rmse = np.sqrt(np.mean((pred - y)**2))
        
    def predict(self, X):
        """
        預測新輸入點的適應度值。
        """
        if X.ndim == 1:
            X_input = X.reshape(1, -1)
        else:
            X_input = X
            
        X_norm = (X_input - self.lb_X) / self.range_X
        dists = cdist(X_norm, self.centers_norm, 'euclidean')
        
        if self.selected_kernel == 'cubic':
            Phi = dists ** 3
        else:
            Phi = np.exp(-(dists ** 2) / self.beta)
            
        pred_norm = Phi @ self.weights
        
        if self.use_log:
            pred_log = pred_norm * self.std_y + self.mean_y
            pred_log_clipped = np.clip(pred_log, -20, 20)
            pred = np.sign(pred_log_clipped) * (10**np.abs(pred_log_clipped) - 1.0)
        else:
            pred = pred_norm * self.std_y + self.mean_y
            
        # 安全邊界防禦：避免預測值異常偏低
        lower_bound = self.y_raw_min - 0.5 * self.y_raw_range
        return np.maximum(pred, lower_bound)


# =========================================================================
# 測試展示：使用 50維 測試數據驗證自適應擬合
# =========================================================================
if __name__ == "__main__":
    np.random.seed(42)
    dim = 50
    n_samples = 150
    
    # 1. 建立測試函數 (例如 50維 Rosenbrock 函數)
    def rosenbrock(x):
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2)
        
    # 2. 隨機採樣訓練集與測試集
    X_train = np.random.uniform(-2.048, 2.048, (n_samples, dim))
    y_train = np.array([rosenbrock(x) for x in X_train])
    
    X_test = np.random.uniform(-2.048, 2.048, (20, dim))
    y_test = np.array([rosenbrock(x) for x in X_test])
    
    # 3. 測試 Cubic 擬合
    rbf_cubic = RBF(kernel_mode='cubic')
    rbf_cubic.fit(X_train, y_train)
    y_pred_cubic = rbf_cubic.predict(X_test)
    test_rmse_cubic = np.sqrt(np.mean((y_pred_cubic - y_test)**2))
    
    # 4. 測試 Gaussian 擬合
    rbf_gauss = RBF(kernel_mode='gaussian')
    rbf_gauss.fit(X_train, y_train)
    y_pred_gauss = rbf_gauss.predict(X_test)
    test_rmse_gauss = np.sqrt(np.mean((y_pred_gauss - y_test)**2))
    
    # 5. 測試 Adaptive 擬合
    rbf_adaptive = RBF(kernel_mode='adaptive')
    rbf_adaptive.fit(X_train, y_train)
    y_pred_adaptive = rbf_adaptive.predict(X_test)
    test_rmse_adaptive = np.sqrt(np.mean((y_pred_adaptive - y_test)**2))
    
    print("=========================================================")
    print("50維自適應 RBF 擬合測試結果")
    print("=========================================================")
    print(f"Cubic 模式選用核心: {rbf_cubic.selected_kernel} | 測試集 RMSE: {test_rmse_cubic:.2e}")
    print(f"Gaussian 模式選用核心: {rbf_gauss.selected_kernel} | 測試集 RMSE: {test_rmse_gauss:.2e}")
    print(f"Adaptive 模式選用核心: {rbf_adaptive.selected_kernel} | 測試集 RMSE: {test_rmse_adaptive:.2e}")
    print("=========================================================")
