import numpy as np
from scipy.spatial.distance import cdist

class RBF:
    """
    Adaptive Radial Basis Function (RBF) surrogate model with X & Y normalization.
    Automatically chooses between Cubic and Gaussian kernels at fit time using
    Rippa's closed-form Leave-One-Out Cross-Validation (LOOCV) formula.
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
        self.kernel_type = None # 'cubic' or 'gaussian'
        self.beta = None # shape parameter for Gaussian RBF
        self.y_raw_min = None
        self.y_raw_max = None
        self.y_raw_range = None
        self.use_log = True

    def fit(self, X, y, use_log=True):
        """
        Fit the RBF model using input data X and target values y.
        Automatically performs normalizations and selects the optimal kernel
        (Cubic vs Gaussian) based on Rippa's closed-form LOOCV RMSE.
        """
        N, d = X.shape
        self.lb_X = np.min(X, axis=0)
        self.ub_X = np.max(X, axis=0)
        self.range_X = self.ub_X - self.lb_X
        # Handle zero-range dimensions
        self.range_X[self.range_X < 1e-8] = 1.0
        
        X_norm = (X - self.lb_X) / self.range_X
        self.centers_norm = np.copy(X_norm)
        
        # Record raw target scale for boundary defense
        self.y_raw_min = np.min(y)
        self.y_raw_max = np.max(y)
        self.y_raw_range = self.y_raw_max - self.y_raw_min
        if self.y_raw_range < 1e-8:
            self.y_raw_range = 1.0

        self.use_log = use_log

        # Target space normalization
        if self.use_log:
            # Log transform to prevent extrapolation divergence
            y_log = np.sign(y) * np.log10(np.abs(y) + 1.0)
            self.mean_y = np.mean(y_log)
            self.std_y = np.std(y_log)
            if self.std_y < 1e-8:
                self.std_y = 1.0
            y_norm = (y_log - self.mean_y) / self.std_y
        else:
            self.mean_y = np.mean(y)
            self.std_y = np.std(y)
            if self.std_y < 1e-8:
                self.std_y = 1.0
            y_norm = (y - self.mean_y) / self.std_y
        
        if N < 2:
            self.kernel_type = 'cubic'
            self.weights = np.copy(y_norm)
            self.cond_phi = 1.0
            self.cond_phi_reg = 1.0
            self.rmse = 0.0
            return
    
        dists = cdist(X_norm, X_norm, 'euclidean')
        
        # Model Evaluation A: Cubic RBF
        Phi_cubic = dists ** 3
        Inv_cubic = np.linalg.pinv(Phi_cubic + 1e-6 * np.eye(N))
        w_cubic = Inv_cubic @ y_norm
        diag_Inv_cubic = np.diag(Inv_cubic)
        diag_Inv_cubic_safe = np.copy(diag_Inv_cubic)
        diag_Inv_cubic_safe[np.abs(diag_Inv_cubic_safe) < 1e-12] = 1.0
        
        # Rippa's closed-form LOOCV error: e_i = w_i / Inv_ii
        loocv_err_cubic = w_cubic / diag_Inv_cubic_safe
        loocv_rmse_cubic = np.sqrt(np.mean(loocv_err_cubic ** 2))
        
      
        # Model Evaluation B: Gaussian RBF
        Dmax = np.max(dists)
        if Dmax < 1e-8:
            Dmax = 1.0
        beta = Dmax * (d * N) ** (-1.0 / d)
        
        Phi_gauss = np.exp(-(cdist(X_norm, X_norm, 'euclidean') ** 2) / beta)
        Inv_gauss = np.linalg.pinv(Phi_gauss + 1e-6 * np.eye(N))
        w_gauss = Inv_gauss @ y_norm
        diag_Inv_gauss = np.diag(Inv_gauss)
        diag_Inv_gauss_safe = np.copy(diag_Inv_gauss)
        diag_Inv_gauss_safe[np.abs(diag_Inv_gauss_safe) < 1e-12] = 1.0
        
        # Rippa's closed-form LOOCV error: e_i = w_i / Inv_ii
        loocv_err_gauss = w_gauss / diag_Inv_gauss_safe
        loocv_rmse_gauss = np.sqrt(np.mean(loocv_err_gauss ** 2))
        
        # Adaptive Selection: Choose kernel with smaller LOOCV RMSE
        if loocv_rmse_gauss < loocv_rmse_cubic:
            self.kernel_type = 'gaussian'
            self.weights = w_gauss
            self.beta = beta
            Phi_chosen = Phi_gauss
        else:
            self.kernel_type = 'cubic'
            self.weights = w_cubic
            self.beta = None
            Phi_chosen = Phi_cubic
            
        try:
            self.cond_phi = np.linalg.cond(Phi_chosen)
        except Exception:
            self.cond_phi = np.nan
            
        try:
            self.cond_phi_reg = np.linalg.cond(Phi_chosen + 1e-6 * np.eye(N))
        except Exception:
            self.cond_phi_reg = np.nan
            
        # Calculate training RMSE
        pred_norm = Phi_chosen @ self.weights
        pred_log = pred_norm * self.std_y + self.mean_y
        pred_log_clipped = np.clip(pred_log, -20, 20)
        
        if self.use_log:
            pred = np.sign(pred_log_clipped) * (10**np.abs(pred_log_clipped) - 1.0)
        else:
            pred = pred_norm * self.std_y + self.mean_y
        self.rmse = np.sqrt(np.mean((pred - y)**2))
        
    def predict(self, X):
        Predict values for query points X.
        if X.ndim == 1:
            X_input = X.reshape(1, -1)
        else:
            X_input = X
            
        # Normalize input
        X_norm = (X_input - self.lb_X) / self.range_X
        
        dists = cdist(X_norm, self.centers_norm, 'euclidean')
        if self.kernel_type == 'cubic':
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
        
        # Lower bound dynamic defense
        lower_bound = self.y_raw_min - 0.5 * self.y_raw_range
        pred = np.maximum(pred, lower_bound)
        
        if X.ndim == 1:
            return pred[0]
        return pred

if __name__ == "__main__":
    # Standard usage example
    print("Initializing RBF Model...")
    model = RBF()
    
    # Generate mock training data (Sphere Function with some noise)
    np.random.seed(42)
    X_train = np.random.uniform(-5.12, 5.12, (50, 5))
    y_train = np.sum(X_train**2, axis=1) + np.random.normal(0, 0.1, 50)
    
    # Fit the adaptive RBF
    model.fit(X_train, y_train, use_log=True)
    
    print(f"Fit completed successfully.")
    print(f"Selected Kernel: {model.kernel_type.upper()}")
    if model.beta:
        print(f"Gaussian Beta parameter: {model.beta:.6f}")
    print(f"Training RMSE: {model.rmse:.6e}")
    
    # Predict on new query points
    X_test = np.random.uniform(-5.12, 5.12, (5, 5))
    predictions = model.predict(X_test)
    print("Predictions on 5 test points:")
    for idx, (pt, val) in enumerate(zip(X_test, predictions)):
        print(f"  Point {idx+1}: {pt} -> Pred: {val:.4f}")
