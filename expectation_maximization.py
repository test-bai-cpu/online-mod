import numpy as np
from scipy.stats import multivariate_normal

import utils


class ExpectationMaximization:
    
    def __init__(
        self,
        grid_data: np.ndarray,
        cluster_centers: np.ndarray,
        cluster_covariance: np.ndarray,
        mixing_factors: np.ndarray,
        wind_num: int = 1,
        convergence_thres: float = 1e-3,
        max_iteration: int = 500
    ) -> None:
        self.grid_data = grid_data
        self.cluster_centers = cluster_centers
        self.cluster_covariance = cluster_covariance
        self.mixing_factors = mixing_factors
        self.wind_num = wind_num
        self.convergence_thres = convergence_thres
        self.max_iteration = max_iteration
        
        self.mean = np.array([])
        self.cov = np.array([])
        self.mix = np.array([])
    
    def run_em_algorithm(self) -> None:
        data = np.array(self.grid_data)
        wind_k = np.arange(-self.wind_num, self.wind_num + 1)
        cluster_nums = len(self.cluster_centers)

        #### computing initial covariance from mean-shift ####
        m = self.cluster_centers.copy()
        c = np.zeros((cluster_nums,2,2), dtype=float)
        p = self.mixing_factors.copy().reshape(-1,1)
        
        for j in range(cluster_nums):
            cluster_cov = self.cluster_covariance[j,:,:]
            f_a = np.diag(cluster_cov)
            f_b = np.floor(np.log10(f_a))
            c[j,:,:] = np.diag(10**(f_b - 1))

        ## EM algorithm ##
        delta = np.inf
        old_log_likelihood = 0
        iter = 0
        
        while np.abs(delta) > self.convergence_thres and iter < self.max_iteration:
            r = np.zeros((cluster_nums, len(wind_k), len(data)), dtype=float)
            
            ## Expectation Step
            for j in range(cluster_nums):
                for k in range(len(wind_k)):
                    wrap_num = wind_k[k]
                    likelihood = np.array([multivariate_normal.pdf(np.array([row[0], row[1] + 2 * np.pi * wrap_num ]), mean=m[j,:], cov=c[j,:,:], allow_singular=True) for row in data]) * p[j,0]
                    r[j,k,:] = likelihood
            r[r < np.finfo(float).eps] = 0
            sum_r = np.tile(np.sum(r, axis=(0, 1)), (cluster_nums, len(wind_k), 1))
            
            r = np.divide(r, sum_r, out=np.zeros_like(r), where=sum_r!=0)
            
            ## Maximization Step
            m = np.zeros((cluster_nums, 2), dtype=float)
            for j in range(cluster_nums):
                t = np.zeros((len(data), 2), dtype=float)
                for k in range(len(wind_k)):
                    wrap_num = wind_k[k]
                    data_copy = data.copy()
                    data_copy[:, 1] += 2 * np.pi * wrap_num
                    t += data_copy * np.tile(r[j,k,:].reshape(-1,1), (1, 2))

                sum_r_j = np.sum(r[j,:,:])
                m[j,:] = np.divide(np.sum(t, axis=0), sum_r_j, where=sum_r_j!=0)
                if sum_r_j == 0:
                    m[j,:] = np.zeros_like(m[j,:])
            
            c = np.zeros((cluster_nums,2,2), dtype=float)
            for j in range(cluster_nums):
                t = np.zeros((len(data), 2, 2), dtype=float)
                for k in range(len(wind_k)):
                    wrap_num = wind_k[k]
                    data_copy = data.copy()
                    data_copy[:, 1] += 2 * np.pi * wrap_num
                    d_mod = data_copy - np.tile(m[j,:], (len(data), 1))
                    t[:,0,0] += d_mod[:,0]**2 * r[j,k,:]
                    t[:,1,1] += d_mod[:,1]**2 * r[j,k,:]
                    t[:,0,1] += d_mod[:,0] * d_mod[:, 1] * r[j,k,:]
                    t[:,1,0] = t[:,0,1]
                    
                sum_r_j = np.sum(r[j,:,:])
                c[j,:,:] = np.divide(np.sum(t, axis=0), sum_r_j, where=sum_r_j!=0)
                if sum_r_j == 0:
                    c[j,:,:] = np.zeros_like(c[j,:,:])
            
            p = np.ones((cluster_nums, 1)) * (1 / cluster_nums)
            for j in range(cluster_nums):
                p[j,0] = np.sum(r[j,:,:]) / len(data)
                
            # check if sum p not zero
            if np.sum(p) == 0:
                p = np.ones((cluster_nums, 1)) * (1 / cluster_nums)
            else:
                p = p / np.sum(p)
            
            for j in range(cluster_nums):
                try:
                    np.linalg.cholesky(c[j,:,:])
                    chol_f = 0
                except np.linalg.LinAlgError:
                    chol_f = 1
                if (chol_f != 0 and c[j,0,0] > 10**(-10) and c[j,1,1] > 10**(-10)) or (np.linalg.cond(c[j,:,:]) > 1/10**(-10)):
                    c[j,:,:] += np.eye(2) * 10**(-10)
                    
            ## discarding clusters with too small covariance
            self.RemNan = 0
            self.Remsmal = 0
            new_cluster_nums = cluster_nums

            rem = np.zeros(cluster_nums, dtype=float)
            rs = 0
            rn = 0
            for j in range(cluster_nums):
                if c[j,0,0] < 10**(-4) or c[j,1,1] < 10**(-4):
                    rem[j] = 1
                    new_cluster_nums -= 1
                    self.Remsmal += 1
                    rs += 1
                if np.linalg.cond(c[j,:,:]) > 1/10**(-10):
                    rem[j] = 1
                    new_cluster_nums -= 1
                    self.RemNan += 1
                    rn += 1

            rem = rem.astype(bool)
            
            if new_cluster_nums < cluster_nums:
                c = c[~rem,:,:]
                m = m[~rem,:]
                p = p[~rem]
                cluster_nums = new_cluster_nums
                old_log_likelihood = 0
                delta = np.inf
                p = p / np.sum(p)
                r = np.zeros((cluster_nums, len(wind_k), len(data)), dtype=float)

            if m.size == 0:
                return
            m[:,1] = utils.wrap_to_2pi_no_round(m[:,1])
            
            
            ##########check delta log likelihood to see if continue##########
            log_likelihood = 0
            ll = np.zeros((len(data), 1), dtype=float)
            for j in range(new_cluster_nums):
                for k in range(len(wind_k)):
                    wrap_num = wind_k[k]
                    likelihood = np.array([multivariate_normal.pdf(np.array([row[0], row[1] + 2 * np.pi * wrap_num]), mean=m[j,:], cov=c[j,:,:], allow_singular=True) for row in data]) * p[j]
                    ll += likelihood.reshape(-1,1)
                    
            log_likelihood = np.sum(np.log(ll + 1e-9))
            
            delta = old_log_likelihood - log_likelihood
            old_log_likelihood = log_likelihood
            
            iter += 1
        
        p = p / np.sum(p)
        
        self.mean = m
        self.cov = c
        self.mix = p