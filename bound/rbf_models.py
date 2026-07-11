# rbf_models.py
import numpy as np
from scipy.spatial.distance import cdist

class RBF:
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
        self.beta = None
        self.kernel = 'cubic'
    def fit(self, X, y, use_log=True, kernel='cubic'):
        N, d = X.shape
        self.lb_X = np.min(X, axis=0)
        self.ub_X = np.max(X, axis=0)
        self.range_X = self.ub_X - self.lb_X
        self.range_X[self.range_X < 1e-8] = 1.0
        X_norm = (X - self.lb_X) / self.range_X
        self.centers_norm = np.copy(X_norm)
        self.kernel=kernel
        
        self.y_raw_min = np.min(y)
        self.y_raw_max = np.max(y)
        self.y_raw_range = self.y_raw_max - self.y_raw_min
        
        if self.y_raw_range < 1e-8:
            self.y_raw_range = 1.0

        self.use_log = use_log

        if self.use_log:
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
            self.weights = np.copy(y_norm)
            self.cond_phi = 1.0
            self.cond_phi_reg = 1.0
            self.rmse = 0.0
            return
            
        dists = cdist(X_norm, X_norm, 'euclidean')
        
        if self.kernel == 'gaussian':
            D_max = np.max(dists)
            if D_max < 1e-8:
                D_max = 1.0
            self.beta = D_max * (N ** (-1.0 / d))
            Phi = np.exp(-(dists ** 2) / self.beta)
        else:
            Phi = dists ** 3
        
        try:
            self.cond_phi = np.linalg.cond(Phi)
        except Exception:
            self.cond_phi = np.nan
            
        try:
            self.cond_phi_reg = np.linalg.cond(Phi + 1e-6 * np.eye(N))
        except Exception:
            self.cond_phi_reg = np.nan
            
        self.weights = np.linalg.pinv(Phi + 1e-6 * np.eye(N)) @ y_norm
        
        pred_norm = Phi @ self.weights

        if self.use_log:
            pred_log = pred_norm * self.std_y + self.mean_y
            pred_log_clipped = np.clip(pred_log, -20, 20)
            pred = np.sign(pred_log_clipped) * (10**np.abs(pred_log_clipped) - 1.0)
        else:
            pred = pred_norm * self.std_y + self.mean_y    
        self.rmse = np.sqrt(np.mean((pred - y)**2))
    def predict(self, X):
        if X.ndim == 1:
            X_input = X.reshape(1, -1)
        else:
            X_input = X
            
        # Normalize input
        X_norm = (X_input - self.lb_X) / self.range_X
        
        dists = cdist(X_norm, self.centers_norm, 'euclidean')
        if self.kernel == 'gaussian':
            Phi = np.exp(-(dists ** 2) / self.beta)
        else:
            Phi = dists ** 3
        pred_norm = Phi @ self.weights
        if self.use_log:
            pred_log = pred_norm * self.std_y + self.mean_y
            pred_log_clipped = np.clip(pred_log, -20, 20)
            pred = np.sign(pred_log_clipped) * (10**np.abs(pred_log_clipped) - 1.0)
        else:
            pred = pred_norm * self.std_y + self.mean_y
        if X.ndim == 1:
            return pred[0]
        return pred


class JADE:
    """
    Vectorized Adaptive Differential Evolution for surrogate model minimization.
    """
    def __init__(self, dim, lb, ub, max_evals=1000, pop_size=30):
        self.dim = dim
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.max_evals = max_evals
        self.pop_size = pop_size
        self.p = 0.15
        self.c = 0.1
        
    def minimize(self, rbf_model):
        pop = self.lb + np.random.rand(self.pop_size, self.dim) * (self.ub - self.lb)
        fitness = rbf_model.predict(pop)
        
        mu_F = 0.5
        mu_Cr = 0.5
        evals = self.pop_size
        
        while evals < self.max_evals:
            success_F = []
            success_Cr = []
            
            Cr_vals = np.random.normal(mu_Cr, 0.1, self.pop_size)
            Cr_vals = np.clip(Cr_vals, 0, 1)
            
            F_vals = np.random.standard_cauchy(self.pop_size) * 0.1 + mu_F
            F_vals = np.clip(F_vals, 0.01, 1.0)
            
            U = np.copy(pop)
            sorted_idx = np.argsort(fitness)
            pbest_num = max(1, int(self.pop_size * self.p))
            pbest_choices = sorted_idx[:pbest_num]
            
            for i in range(self.pop_size):
                x_best = pop[np.random.choice(pbest_choices)]
                
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                r1 = np.random.choice(candidates)
                candidates.remove(r1)
                r2 = np.random.choice(candidates)
                
                v = pop[i] + F_vals[i] * (x_best - pop[i]) + F_vals[i] * (pop[r1] - pop[r2])
                v = np.clip(v, self.lb, self.ub)
                
                j_rand = np.random.randint(self.dim)
                u = np.copy(pop[i])
                cross_mask = np.random.rand(self.dim) < Cr_vals[i]
                cross_mask[j_rand] = True
                u[cross_mask] = v[cross_mask]
                U[i] = u
                
            fit_U = rbf_model.predict(U)
            evals += self.pop_size
            
            for i in range(self.pop_size):
                if fit_U[i] < fitness[i]:
                    pop[i] = U[i]
                    fitness[i] = fit_U[i]
                    success_F.append(F_vals[i])
                    success_Cr.append(Cr_vals[i])
                    
            if len(success_F) > 0:
                mu_Cr = (1 - self.c) * mu_Cr + self.c * np.mean(success_Cr)
                num = np.sum(np.array(success_F) ** 2)
                den = np.sum(success_F)
                mu_F = (1 - self.c) * mu_F + self.c * (num / den if den > 0 else 0.5)
                
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]