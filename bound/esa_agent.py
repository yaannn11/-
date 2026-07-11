import numpy as np
from rbf_models import RBF, JADE
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
        self.safety_multiplier = 0.5
        self.current_safety_multiplier = self.safety_multiplier
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
    def cal_lower_bound(self):
        y_min = np.min(self.y_db)
        y_db_arr = np.asarray(self.y_db)
        current_std = np.std(y_db_arr) if len(y_db_arr) > 0 else 1.0
        if current_std < 1e-8: current_std = 1.0
        return self.current_safety_multiplier * current_std
    def cal_upper_bound(self):
        y_min = np.min(self.y_db)
        y_db_arr = np.asarray(self.y_db)
        current_std = np.std(y_db_arr) if len(y_db_arr) > 0 else 1.0
        if current_std < 1e-8: current_std = 1.0
        return self.safety_multiplier * current_std
    def cal_error(self, pred_yx):
        return float(np.min(self.y_db)) - float(pred_yx)
    def bound_check(self, current_pred,method):
        if current_pred is None:
            return False, "error"
        if len(self.y_db) == 0:
            return -float('inf')
        y_min = np.min(self.y_db)
        lower_bound=y_min - self.cal_lower_bound()
        upper_bound= y_min + self.cal_upper_bound()
        if current_pred < lower_bound:
            return False, "lower_break"
        if current_pred > upper_bound:
            return False, "upper_break"
        return True, 'normal'
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

        check,reason=self.bound_check(pred_yc,'a1')
        error=self.cal_error(pred_yc)
        if not check:
            return False, x_c, None, True, pred_yc, reason,self.cal_lower_bound(),self.cal_upper_bound(), error
        
        success, y_c = self._evaluate(x_c)
        self.record_evaluation(0, state, pred_yc, y_c, x_c, success)
        return success, x_c, y_c, False, pred_yc, 'normal',self.cal_lower_bound(),self.cal_upper_bound(), error

    def action_a2(self, state):
        l_best = min(25 + self.dim, 60, len(self.X_db))
        X_train, y_train = self.get_best_data(l_best)
        rbf = RBF()
        if self.nfe < 400:
            rbf.fit(X_train, y_train, use_log=False, kernel='gaussian')
        else:
            rbf.fit(X_train, y_train, use_log=False, kernel='gaussian')
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

        check,reason=self.bound_check(pred_yc,'a2')
        error=self.cal_error(pred_yc)
        if not check:
            return False, x_c, None, True, pred_yc, reason,self.cal_lower_bound(),self.cal_upper_bound(), error
        
        success, y_c = self._evaluate(x_c)
        self.record_evaluation(1, state, pred_yc, y_c, x_c, success)
        return success, x_c, y_c, False, pred_yc,'normal',self.cal_lower_bound(),self.cal_upper_bound(), error

    def action_a3(self, state):
        m_pop = min(100, len(self.X_db))
        P_X, y_pop = self.get_best_data(m_pop)
        
        rbf = RBF()

        if self.nfe < 400:
            rbf.fit(P_X, y_pop, use_log=False)
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

        check,reason=self.bound_check(pred_yc,'a3')
        error=self.cal_error(pred_yc)
        pred_yyc=pred_yc
        if not check:
            return False, x_c, None, True, pred_yc, reason,self.cal_lower_bound(),self.cal_upper_bound(), error
        
        success, y_c = self._evaluate(x_c)
        
        self.record_evaluation(2, state, pred_yc, y_c, x_c, success)
        return success, x_c, y_c, False, pred_yc, 'normal',self.cal_lower_bound(),self.cal_upper_bound(), error

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
        eval_count = 0   
        fired_safeguard = False 
        all_pred_xc=[]
        current_error=[]
        lower_bound=[]
        upper_bound=[]
        a4_nfe=[]
        
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
                rbf.fit(X_tr, y_tr, use_log=False, kernel='gaussian')
            else:
                rbf.fit(X_tr, y_tr, use_log=False, kernel='gaussian')
            
            self.record_rbf(rbf, 3, len(X_tr))
            
            optimizer = JADE(self.dim, lb_tr, ub_tr, max_evals=600, pop_size=20)
            x_c, pred_xc = optimizer.minimize(rbf)
            all_pred_xc.append(pred_xc)
            current_error.append(self.cal_error(pred_xc))
            pred_best = rbf.predict(x_best.reshape(1, -1))[0]
            lower_bound.append(self.cal_lower_bound())
            upper_bound.append(self.cal_upper_bound())
            a4_nfe.append(self.nfe)
            check,reason=self.bound_check(pred_xc,'a4')
            if not check:
                if eval_count < 1:
                    fired_safeguard = True
                else:
                    reason='normal'
                delta_k = 0.25 * delta_k  
                break            
            else :
                reason='normal'
            y_best_prev = y_best
            success, y_c = self._evaluate(x_c)
            eval_count += 1
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
        return success_any, x_best, y_best, fired_safeguard, eval_count, all_pred_xc,reason,current_error,lower_bound,upper_bound,a4_nfe

    import numpy as np

    def run(self):
        # 初始化資料庫與固定長度的歷史空間
        self.initialize_db()
        state = self.current_state
        self._init_history_buffers()
        
        # 主優化迴圈
        while self.nfe < self.max_nfe:
            action_approved = False
            retry_count = 0
            max_retries = 5
            start_nfe = self.nfe  # 記住進入這一輪優化時的初始 NFE

            while not action_approved and retry_count < max_retries:
                action = self.agent.select_action(state)
                self.action_counts[action] += 1
                best_y_before = self.best_y
            
                fired_safeguard, reason, nfe_list, errors, lb_list, ub_list = self._execute_action(action, state, start_nfe)

                self._update_history_records(nfe_list, errors, lb_list, ub_list)

                if fired_safeguard:
                    state, retry_count = self._handle_safeguard_failure(action, reason, state, retry_count)
                else:
                    action_approved = True

            if not action_approved:
                self.current_safety_multiplier += 0.2
                continue
            
            state = self._handle_action_success(action, state, best_y_before)
            
        actual_round_nfe = list(range(self.nfe + 1))
        return self._get_return_dict(actual_round_nfe)
    def _init_history_buffers(self):
        #初始化以 NFE 為索引的固定長度空間
        size = self.max_nfe + 1
        self.history_retries = [-1] * size
        self.history_multipliers = [self.safety_multiplier] * size
        self.history_errors = [""] * size
        
        self.history_buffers = [float('nan')] * size
        self.history_upper_buffers = [float('nan')] * size
        self.history_final_buffers = [float('nan')] * size
        self.history_final_upper_buffers = [float('nan')] * size
        
        self.history_first_improvement = [float('nan')] * size
        self.history_final_improvement = [float('nan')] * size


    def _execute_action(self, action, state, start_nfe):
        #執行對應的 Action並將輸出格式標準化
        if action in (0, 1, 2):
            if action == 0:
                res = self.action_a1(state)
            elif action == 1:
                res = self.action_a2(state)
            else:
                res = self.action_a3(state)
                
            success, x_c, y_c, fired_safeguard, current_pred, reason, lower_bound, upper_bound, current_error = res
            
            nfe_list = [start_nfe]
            errors = [current_error] if current_error is not None else [0.0]
            lb_list = [lower_bound]
            ub_list = [upper_bound]
            
        else:
            success, x_c, y_c, fired_safeguard, eval_count, current_pred, reason, current_error, lower_bound, upper_bound, a4_nfe = self.action_a4(state)
            
            nfe_list = a4_nfe
            errors = current_error if current_error is not None else [0.0]
            lb_list = lower_bound if isinstance(lower_bound, list) else [lower_bound] * len(errors)
            ub_list = upper_bound if isinstance(upper_bound, list) else [upper_bound] * len(errors)
            
        return fired_safeguard, reason, nfe_list, errors, lb_list, ub_list


    def _update_history_records(self, nfe_list, errors, lb_list, ub_list):
        #精準紀錄 First 視角與 Final 視角的歷史軌跡
        for idx, nfe_pos in enumerate(nfe_list):
            n = int(min(nfe_pos, self.max_nfe))
            
            if np.isnan(self.history_first_improvement[n]):
                self.history_first_improvement[n] = errors[idx]
                self.history_buffers[n] = lb_list[idx]
                self.history_upper_buffers[n] = ub_list[idx]

            self.history_final_improvement[n] = errors[idx]
            self.history_final_buffers[n] = lb_list[idx]
            self.history_final_upper_buffers[n] = ub_list[idx]
            
            self.history_retries[n] += 1
            self.history_multipliers[n] = self.current_safety_multiplier
            
            error_str = f"{errors[idx]:.4e}"
            if self.history_errors[n] == "":
                self.history_errors[n] = error_str
            else:
                self.history_errors[n] += f"_{error_str}"


    def _handle_safeguard_failure(self, action, reason, state, retry_count):
        if reason == "lower_break":
            penalty_reward = -0.2
        elif reason == "upper_break":
            penalty_reward = -0.4  
        else:
            penalty_reward = -0.2 
            
        next_state = action * 2 + 0  
        self.agent.update(state, action, penalty_reward, next_state)
        return next_state, retry_count + 1


    def _handle_action_success(self, action, state, best_y_before):
        self.current_safety_multiplier = self.safety_multiplier
        reward = 1.0 if self.best_y < best_y_before else 0.0
        next_state = action * 2 + (1 if reward > 0.5 else 0)
        self.agent.update(state, action, reward, next_state)
        return next_state


    def _get_return_dict(self, actual_round_nfe):
        return {
            'best_x': self.best_x,
            'best_y': self.best_y,
            'history_nfe': self.history_nfe,
            'history_fitness': self.history_fitness,
            'rbf_records': self.rbf_records,
            'eval_records': self.eval_records,
            'q_table': self.agent.q_table, 
            'action_counts': self.action_counts,
            'history_retries': self.history_retries,
            'history_multipliers': self.history_multipliers,
            'history_round_nfe': actual_round_nfe,
            'history_errors': self.history_errors,
            'history_buffers': self.history_buffers,
            'history_upper_buffers': self.history_upper_buffers,
            'history_first_improvement': self.history_first_improvement,
            'history_final_improvement': self.history_final_improvement,
            'history_final_buffers': self.history_final_buffers,
            'history_final_upper_buffers': self.history_final_upper_buffers
        }