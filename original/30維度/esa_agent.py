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
      
        # 距離篩選過濾器 (Distance Filter)
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
