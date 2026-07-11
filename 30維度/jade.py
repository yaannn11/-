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
