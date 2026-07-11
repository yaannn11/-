import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import multiprocessing
import time
import os

# =========================================================================
# 1. Core ESA Optimization Classes (Cubic RBF with X & Y Normalization)
# =========================================================================

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

class QLearningAgent:
    def __init__(self, num_actions=4, num_states=8, alpha=0.1, gamma=0.9, T=1.0):
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.T = T
        self.q_table = np.full((num_states, num_actions), 0.25)
        
    def select_action(self, state):
        q_values = self.q_table[state]
        exp_q = np.exp((q_values - np.max(q_values)) / self.T)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.num_actions, p=probs)
        
    def update(self, state, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])

class EvolutionarySamplingAgent:
    """
    Evolutionary Sampling Agent (ESA) Framework.
    """
    def __init__(self, obj_func, dim, lb, ub, max_nfe=1000):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        self.max_nfe = max_nfe
        self.X_db = []
        self.y_db = []
        self.nfe = 0
        self.best_x = None
        self.best_y = float('inf')
        self.agent = QLearningAgent(num_actions=4, num_states=8)
        self.current_state = np.random.randint(8)
        self.history_nfe = []
        self.history_fitness = []
        self.rbf_records = []
        self.eval_records = []
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        #=====================================================================
        # 💡 新增：動態計算決策空間的最小距離閾值（例如：全局對角線長度的 0.1%）
        # =====================================================================
        domain_diagonal = np.linalg.norm(self.ub - self.lb)
        self.min_dist_threshold = 1e-3 * domain_diagonal
        
    def record_rbf(self, rbf, action, num_points):
        self.rbf_records.append({
            'nfe': self.nfe,
            'action': action,
            'num_points': num_points,
            'rmse': rbf.rmse,
            'cond_phi': rbf.cond_phi,
            'cond_phi_reg': rbf.cond_phi_reg
        })
        
    def record_evaluation(self, action, state, predicted_y, actual_y, x_c, improved):
        if len(self.X_db) > 1:
            dists = np.linalg.norm(np.array(self.X_db[:-1]) - x_c, axis=1)
            min_dist = np.min(dists)
        else:
            min_dist = 0.0
            
        abs_err = abs(actual_y - predicted_y)
        rel_err = abs_err / max(1.0, abs(actual_y))
        
        self.eval_records.append({
            'nfe': self.nfe,
            'action': action,
            'state': state,
            'reward': 1.0 if improved else 0.0,
            'predicted_y': predicted_y,
            'actual_y': actual_y,
            'error': abs_err,
            'relative_error': rel_err,
            'min_dist_db': min_dist
        })
    def _evaluate(self, x):
        x = np.clip(x, self.lb, self.ub)
        y = self.obj_func(x)
        self.nfe += 1
        self.X_db.append(x)
        self.y_db.append(y)
        
        improved = False
        if y < self.best_y:
            self.best_y = y
            self.best_x = np.copy(x)
            improved = True
            
        self.history_nfe.append(self.nfe)
        self.history_fitness.append(self.best_y)
        return improved, y
        
    def initialize_db(self):
        # LHS Initialization (100 samples)
        n_init = 100
        lhs_samples = np.zeros((n_init, self.dim))
        for j in range(self.dim):
            intervals = np.linspace(0, 1, n_init + 1)
            temp = np.random.uniform(intervals[:-1], intervals[1:], n_init)
            np.random.shuffle(temp)
            lhs_samples[:, j] = temp
        X_init = self.lb + lhs_samples * (self.ub - self.lb)
        for x in X_init:
            self._evaluate(x)
            
    def get_best_data(self, k):
        X = np.array(self.X_db)
        y = np.array(self.y_db)
        idx = np.argsort(y)
        X_sorted = X[idx]
        y_sorted = y[idx]
        
        # =====================================================================
        # 💡 核心修改：距離篩選過濾器 (Distance Filter)
        # =====================================================================
        filtered_X = []
        filtered_y = []
        
        for i in range(len(X_sorted)):
            pt = X_sorted[i]
            # 如果是第一個點（當前最優點），無條件保留
            if len(filtered_X) == 0:
                filtered_X.append(pt)
                filtered_y.append(y_sorted[i])
            else:
                # 計算當前點與「已經被保留的點」之間的最小距離
                dists = np.linalg.norm(np.array(filtered_X) - pt, axis=1)
                if np.min(dists) >= self.min_dist_threshold:
                    filtered_X.append(pt)
                    filtered_y.append(y_sorted[i])
                    
            # 如果收集到的優質且不重複的點已經夠了，就提早結束
            if len(filtered_X) >= k:
                break
                
        # 防禦機制：如果過濾完發現點太少（例如小於 5 個點），會導致 RBF 無法正常擬合
        # 此時放寬限制，退回使用原始未過濾的前 k 個點
        if len(filtered_X) < min(k, 5):
            return X_sorted[:k], y_sorted[:k]
            
        return np.array(filtered_X), np.array(filtered_y)
        
    def action_a1(self, state):
        n_pop = 100
        P_X, _ = self.get_best_data(min(n_pop, len(self.X_db)))
        g_best = min(300, len(self.X_db))
        X_train, y_train = self.get_best_data(g_best)
        rbf = RBF()
        if self.nfe <400:
            rbf.fit(X_train, y_train, use_log=True)
        else:
            rbf.fit(X_train, y_train, use_log=False)
        self.record_rbf(rbf, 0, len(X_train))
        
        n_actual = len(P_X)
        trial_vectors = []
        F = 0.5
        Cr = 0.9
        
        for i in range(n_actual):
            candidates = list(range(n_actual))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            v = P_X[r1] + F * (P_X[r2] - P_X[r3])
            v = np.clip(v, self.lb, self.ub)
            
            u = np.copy(P_X[i])
            j_rand = np.random.randint(self.dim)
            cross_mask = np.random.rand(self.dim) < Cr
            cross_mask[j_rand] = True
            u[cross_mask] = v[cross_mask]
            trial_vectors.append(u)
            
        trial_vectors = np.array(trial_vectors)
        pred_y = rbf.predict(trial_vectors)
        best_trial_idx = np.argmin(pred_y)
        x_c = trial_vectors[best_trial_idx]
        pred_yc = pred_y[best_trial_idx]
        
        success, y_c = self._evaluate(x_c)
        self.record_evaluation(0, state, pred_yc, y_c, x_c, success)
        return success, x_c, y_c

    def action_a2(self, state):
        l_best = min(25 + self.dim, 60, len(self.X_db))
        X_train, y_train = self.get_best_data(l_best)
        rbf = RBF()
        if self.nfe < 400:
            rbf.fit(X_train, y_train, use_log=True)
        else:
            rbf.fit(X_train, y_train, use_log=False)
        self.record_rbf(rbf, 1, len(X_train))
        
        lb_local = np.min(X_train, axis=0)
        ub_local = np.max(X_train, axis=0)
        
        # Expand local search boundaries by 5% of global domain to allow extrapolation
        '''
        expansion = 0.05 * (self.ub - self.lb)
        lb_local = np.maximum(self.lb, lb_local - expansion)
        ub_local = np.minimum(self.ub, ub_local + expansion)
        '''


        optimizer = JADE(self.dim, lb_local, ub_local, max_evals=1000, pop_size=30)
        x_c, pred_yc = optimizer.minimize(rbf)
        
        success, y_c = self._evaluate(x_c)
        self.record_evaluation(1, state, pred_yc, y_c, x_c, success)
        return success, x_c, y_c

    def action_a3(self, state):
        m_pop = min(100, len(self.X_db))
        P_X, y_pop = self.get_best_data(m_pop)
        
        rbf = RBF()

        if self.nfe < 400:
            rbf.fit(P_X, y_pop, use_log=True)
        else:
            rbf.fit(P_X, y_pop, use_log=False)

        self.record_rbf(rbf, 2, len(P_X))
        
        phi = np.random.permutation(self.dim)
        x_best = np.copy(P_X[0])
        
        for i in phi:
            P_temp = np.tile(x_best, (len(P_X), 1))
            P_temp[:, i] = P_X[:, i]
            pred_y = rbf.predict(P_temp)
            best_idx = np.argmin(pred_y)
            x_best = P_temp[best_idx]
            
        x_c = x_best
        pred_yc = rbf.predict(x_c.reshape(1, -1))[0]
        
        success, y_c = self._evaluate(x_c)
        self.record_evaluation(2, state, pred_yc, y_c, x_c, success)
        return success, x_c, y_c

    def action_a4(self, state):
        m_pop = min(100, len(self.X_db))
        X_train, y_train = self.get_best_data(m_pop)
        
        lb_local = np.min(X_train, axis=0)
        ub_local = np.max(X_train, axis=0)
        
        x_best = np.copy(self.best_x)
        y_best = self.best_y
        
        best_data_idx = np.argsort(self.y_db)
        x_min_f = self.X_db[best_data_idx[0]]
        x_max_f = self.X_db[best_data_idx[-1]]
        D = np.linalg.norm(x_min_f - x_max_f)
        delta_k = 0.5 * D if D > 0 else 1.0
        
        xi = 2.0
        success_any = False
        
        for k in range(3):
            lb_tr = np.maximum(lb_local, x_best - delta_k)
            ub_tr = np.minimum(ub_local, x_best + delta_k)
            
            for j in range(self.dim):
                if ub_tr[j] - lb_tr[j] < 1e-5:
                    lb_tr[j] = max(self.lb[j], lb_tr[j] - 0.01 * (self.ub[j] - self.lb[j]))
                    ub_tr[j] = min(self.ub[j], ub_tr[j] + 0.01 * (self.ub[j] - self.lb[j]))
            
            X_all = np.array(self.X_db)
            y_all = np.array(self.y_db)
            
            in_tr = np.all((X_all >= lb_tr) & (X_all <= ub_tr), axis=1)
            X_tr = X_all[in_tr]
            y_tr = y_all[in_tr]
            
            if len(X_tr) < 10:
                dists = np.linalg.norm(X_all - x_best, axis=1)
                near_idx = np.argsort(dists)[:20]
                X_tr = X_all[near_idx]
                y_tr = y_all[near_idx]
                
            rbf = RBF()

            if self.nfe < 400:
                rbf.fit(X_tr, y_tr, use_log=True)
            else:
                rbf.fit(X_tr, y_tr, use_log=False)
            
            self.record_rbf(rbf, 3, len(X_tr))
            
            optimizer = JADE(self.dim, lb_tr, ub_tr, max_evals=600, pop_size=20)
            x_c, pred_xc = optimizer.minimize(rbf)
            
            pred_best = rbf.predict(x_best.reshape(1, -1))[0]
            
            y_best_prev = y_best
            success, y_c = self._evaluate(x_c)
            self.record_evaluation(3, state, pred_xc, y_c, x_c, success)
            
            if success:
                success_any = True
                
            real_decrease = y_best_prev - y_c
            pred_decrease = pred_best - pred_xc
            
            if abs(pred_decrease) > 1e-12:
                rho = real_decrease / pred_decrease
            else:
                rho = 1.0 if real_decrease > 0 else 0.0
                
            if rho <= 0.25:
                delta_k = 0.25 * delta_k
            elif rho >= 0.75:
                delta_k = xi * delta_k
            
            if y_c < y_best:
                x_best = np.copy(x_c)
                y_best = y_c
                
            if self.nfe >= self.max_nfe:
                break
                
        return success_any, x_best, y_best

    def run(self):
        self.initialize_db()
        state = self.current_state
        while self.nfe < self.max_nfe:
            action = self.agent.select_action(state)
            self.action_counts[action] += 1
            best_y_before = self.best_y
            
            if action == 0:
                success, x_c, y_c = self.action_a1(state)
            elif action == 1:
                success, x_c, y_c = self.action_a2(state)
            elif action == 2:
                success, x_c, y_c = self.action_a3(state)
            else:
                success, x_c, y_c = self.action_a4(state)
                
            reward = 1.0 if self.best_y < best_y_before else 0.0
            next_state = action * 2 + (1 if reward > 0.5 else 0)
            self.agent.update(state, action, reward, next_state)
            state = next_state
            
        return self.best_x, self.best_y, self.history_nfe, self.history_fitness, self.rbf_records, self.eval_records, self.agent.q_table, self.action_counts

# =========================================================================
# 2. 7 Benchmark Functions Definitions (Shifting, Rotation & Normalized Compositions)
# =========================================================================

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

# =========================================================================
# 3. Experiment Runner & Process Pool Task
# =========================================================================

def run_experiment_task(args):
    func_name, seed, dim, max_nfe = args
    func = get_benchmark_function(func_name, dim)
    np.random.seed(seed)
    
    esa = EvolutionarySamplingAgent(func, dim, func.lb, func.ub, max_nfe=max_nfe)
    best_x, best_y, history_nfe, history_fitness, rbf_records, eval_records, q_table, action_counts = esa.run()
    
    return {
        'func_name': func_name,
        'seed': seed,
        'best_y': best_y,
        'history_nfe': history_nfe,
        'history_fitness': history_fitness,
        'rbf_records': rbf_records,
        'eval_records': eval_records,
        'q_table': q_table.tolist() if hasattr(q_table, 'tolist') else q_table,
        'action_counts': action_counts
    }

if __name__ == "__main__":
    t_start = time.time()
    
    # Configuration
    dim = 30
    max_nfe = 1000
    seeds = [10, 20, 30, 42, 50]
    functions = ['Ellipsoid', 'Rosenbrock', 'Ackley', 'Griewank', 'SRR', 'RHC1', 'RHC2']
    
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(base_dir, 'charts'), exist_ok=True)
    
    print("=====================================================================")
    print("Evolutionary Sampling Agent (ESA) Parallel 7 Run (Normalized RBF)")
    print("=====================================================================")
    print(f"Dimension: {dim}D | NFE Budget: {max_nfe} | Seeds: {seeds}")
    print("Running experiments in parallel...")
    
    task_args = []
    for f in functions:
        for s in seeds:
            task_args.append((f, s, dim, max_nfe))
            
    # Execute in parallel
    pool = multiprocessing.Pool(processes=min(len(task_args), os.cpu_count() or 4))
    results_list = pool.map(run_experiment_task, task_args)
    pool.close()
    pool.join()
    
    results_dict = {f: {} for f in functions}
    for res in results_list:
        results_dict[res['func_name']][res['seed']] = {
            'best_y': res['best_y'],
            'history_nfe': res['history_nfe'],
            'history_fitness': res['history_fitness'],
            'rbf_records': res['rbf_records'],
            'eval_records': res['eval_records'],
            'q_table': res['q_table'],
            'action_counts': res['action_counts']
        }
        
    # Write statistical report
    output_path = os.path.join(base_dir, 'esa_experiments_output.txt')
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write("========================================================\n")
        out_f.write("ESA 7 Functions Run Report (Normalized RBF & Bounds Corrected)\n")
        out_f.write("========================================================\n\n")
        
        summary_stats = {}
        
        for f in functions:
            out_f.write(f"--- Function: {f} ---\n")
            seed_vals = []
            for s in seeds:
                val = results_dict[f][s]['best_y']
                seed_vals.append(val)
                out_f.write(f"  Seed {s}: {val:.6e}\n")
                
            seed_vals = np.array(seed_vals)
            mean_val = np.mean(seed_vals)
            std_val = np.std(seed_vals)
            best_val = np.min(seed_vals)
            worst_val = np.max(seed_vals)
            
            summary_stats[f] = {
                'mean': mean_val,
                'std': std_val,
                'best': best_val,
                'worst': worst_val
            }
            
            out_f.write("  - Summary Stats -\n")
            out_f.write(f"    Mean   : {mean_val:.6e}\n")
            out_f.write(f"    Std Dev: {std_val:.6e}\n")
            out_f.write(f"    Best   : {best_val:.6e}\n")
            out_f.write(f"    Worst  : {worst_val:.6e}\n\n")
            
        # Write final markdown table
        out_f.write("========================================================\n")
        out_f.write("SUMMARY STATISTICS MATRIX\n")
        out_f.write("========================================================\n")
        out_f.write("| Function | Best | Worst | Mean | Std Dev |\n")
        out_f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for f in functions:
            stats = summary_stats[f]
            out_f.write(f"| {f} | {stats['best']:.6e} | {stats['worst']:.6e} | {stats['mean']:.6e} | {stats['std']:.6e} |\n")
        out_f.write("========================================================\n")
        
    print(f"Statistical report saved to {output_path}")
    
    # 4. Generate Convergence Curves
    print("Generating convergence plots...")
    for f in functions:
        plt.figure(figsize=(8, 5))
        for s in seeds:
            hist_nfe = results_dict[f][s]['history_nfe']
            hist_fitness = results_dict[f][s]['history_fitness']
            plt.plot(hist_nfe, hist_fitness, label=f"Seed {s}", alpha=0.8)
            
        plt.yscale('log')
        plt.xlabel('Number of Function Evaluations (NFE)')
        plt.ylabel('Best Fitness (Log Scale)')
        plt.title(f"Convergence Curves: {f} (Normalized RBF)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, f"charts/convergence_{f.lower()}.png"), dpi=200)
        plt.close()
        
    # 5. Generate Bar Chart Performance Comparison
    print("Generating statistical summary bar chart...")
    means = [summary_stats[f]['mean'] for f in functions]
    stds = [summary_stats[f]['std'] for f in functions]
    bests = [summary_stats[f]['best'] for f in functions]
    
    x_indices = np.arange(len(functions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x_indices - width/2, means, width, yerr=stds, label='Mean Fitness', color='#3498db', capsize=5)
    rects2 = ax.bar(x_indices + width/2, bests, width, label='Best Fitness', color='#2ecc71')
    
    ax.set_yscale('log')
    ax.set_ylabel('Fitness Value (Log Scale)', fontsize=12, fontweight='bold')
    ax.set_title('ESA 7 Functions Optimization Performance (Normalized RBF)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(functions, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'charts/performance_comparison.png'), dpi=300)
    plt.close()
    
    # 6. Save Analysis CSV Files
    print("Generating analysis CSV files...")
    
    # 6.1 Save Prediction Errors (Trajectory Diagnostic Log)
    rmse_path = os.path.join(base_dir, 'prediction_errors.csv')
    with open(rmse_path, 'w', encoding='utf-8') as f_rmse:
        f_rmse.write("Function,Seed,NFE,Action,State,Reward,Predicted_Y,Actual_Y,Error,Relative_Error,Min_Dist_DB\n")
        for f in functions:
            for s in seeds:
                records = results_dict[f][s]['eval_records']
                for r in records:
                    f_rmse.write(f"{f},{s},{r['nfe']},{r['action']},{r['state']},{r['reward']:.1f},"
                                 f"{r['predicted_y']:.6e},{r['actual_y']:.6e},{r['error']:.6e},"
                                 f"{r['relative_error']:.6e},{r['min_dist_db']:.6e}\n")
    print(f"Prediction errors file saved to {rmse_path}")
    
    # 6.2 Save Condition Numbers
    cond_path = os.path.join(base_dir, 'condition_numbers.csv')
    with open(cond_path, 'w', encoding='utf-8') as f_cond:
        f_cond.write("Function,Seed,NFE,Action,NumPoints,CondPhi,CondPhiReg\n")
        for f in functions:
            for s in seeds:
                records = results_dict[f][s]['rbf_records']
                for r in records:
                    f_cond.write(f"{f},{s},{r['nfe']},{r['action']},{r['num_points']},{r['cond_phi']:.6e},{r['cond_phi_reg']:.6e}\n")
    print(f"Condition numbers file saved to {cond_path}")
    
    # 6.3 Save Q-tables
    q_path = os.path.join(base_dir, 'q_tables.csv')
    with open(q_path, 'w', encoding='utf-8') as f_q:
        f_q.write("Function,Seed,State,Q_Action_0,Q_Action_1,Q_Action_2,Q_Action_3\n")
        for f in functions:
            for s in seeds:
                q_table = results_dict[f][s]['q_table']
                for state_idx, q_vals in enumerate(q_table):
                    f_q.write(f"{f},{s},{state_idx},{q_vals[0]:.6e},{q_vals[1]:.6e},{q_vals[2]:.6e},{q_vals[3]:.6e}\n")
    print(f"Q-tables file saved to {q_path}")
    
    # 6.4 Save Strategy Usage Counts
    usage_path = os.path.join(base_dir, 'strategy_usage.csv')
    with open(usage_path, 'w', encoding='utf-8') as f_use:
        f_use.write("Function,Seed,Action_0_Count,Action_1_Count,Action_2_Count,Action_3_Count,Action_0_Ratio,Action_1_Ratio,Action_2_Ratio,Action_3_Ratio\n")
        for f in functions:
            for s in seeds:
                counts = results_dict[f][s]['action_counts']
                total = sum(counts.values())
                if total > 0:
                    ratios = {act: counts[act]/total for act in range(4)}
                else:
                    ratios = {act: 0.0 for act in range(4)}
                f_use.write(f"{f},{s},"
                            f"{counts[0]},{counts[1]},{counts[2]},{counts[3]},"
                            f"{ratios[0]:.4f},{ratios[1]:.4f},{ratios[2]:.4f},{ratios[3]:.4f}\n")
    print(f"Strategy usage counts file saved to {usage_path}")
    
    t_elapsed = time.time() - t_start
    print("=====================================================================")
    print(f"All experiments completed successfully in {t_elapsed:.2f} seconds.")
    print("=====================================================================")
