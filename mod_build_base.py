import os
import shutil
import numpy as np
from scipy.stats import multivariate_normal

from mean_shift import MeanShift
from expectation_maximization import ExpectationMaximization
from batch_process_data import BatchDataProcess, GridData
import utils


class BuildMoDBase:
    
    def __init__(
        self,
        config_file: str,
        current_cliff: str,
        output_cliff_folder: str,
        save_fig_folder: str,
    ) -> None:
        self.config_params = utils.read_config_file(config_file)
        if current_cliff is not None and os.path.exists(current_cliff):
            self.current_cliff = utils.read_cliff_map_data(current_cliff)
        else:
            self.current_cliff = None

        if os.path.exists(output_cliff_folder) and os.path.isdir(output_cliff_folder):
            shutil.rmtree(output_cliff_folder)

        self.cliff_csv_folder = output_cliff_folder
        self.save_fig_folder = save_fig_folder
        os.makedirs(self.cliff_csv_folder, exist_ok=True)
        os.makedirs(self.save_fig_folder, exist_ok=True)
        
        self.data_batches = BatchDataProcess(
            radius=float(self.config_params["radius"]),
            step=float(self.config_params["step"]),
            dataset_type=self.config_params["dataset_type"])

    def regularize_cov_matrix(self, cov_matrix, epsilon=1e-6):
        cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
        return cov_matrix

    def build_cliff(self, key, data: GridData, if_build_with_new_data=False) -> GridData:
        mean_shifter = MeanShift(
            grid_data=data, 
            convergence_threshold=float(self.config_params["convergence_thres_ms"]),
            group_distance_tolerance=float(self.config_params["group_distance_tolerance"]),
            cluster_all=bool(self.config_params["cluster_all"]),
            too_few_data_thres=int(self.config_params["too_few_data_thres"]),
            max_iteration=int(self.config_params["max_iter_ms"]),
            if_build_with_new_data=if_build_with_new_data,
        )
        mean_shifter.run_mean_shift()
        
        if len(mean_shifter.cluster_centers) == 0:
            return [], None, None, None
        
        if if_build_with_new_data:
            pruned_data = utils.pruned_data_after_ms(data.new_data, mean_shifter.data_cluster_labels)
        else:
            pruned_data = utils.pruned_data_after_ms(data.data, mean_shifter.data_cluster_labels)
        emv = ExpectationMaximization(
            grid_data=pruned_data,
            cluster_centers=mean_shifter.cluster_centers, 
            cluster_covariance=mean_shifter.covariances, 
            mixing_factors=mean_shifter.mixing_factors,
            wind_num=int(self.config_params["wind_num"]),
            convergence_thres=float(self.config_params["convergence_thres_em"]),
            max_iteration=int(self.config_params["max_iter_em"])
        )

        emv.run_em_algorithm()

        if len(emv.mean) == 0:
            return [], None, None, None

        cliffs = []
        
        for cluster_i in range(len(emv.mean)):
            save_row = [
                key[0], key[1],
                emv.mean[cluster_i,0], emv.mean[cluster_i,1],
                emv.cov[cluster_i,0,0], emv.cov[cluster_i,0,1], emv.cov[cluster_i,1,0], emv.cov[cluster_i,1,1],
                emv.mix[cluster_i,0], data.motion_ratio
            ]

            rounded_save_row = [round(value, 5) if not (value is None) else value for value in save_row]
            cliffs.append(rounded_save_row)

        wind_num = int(self.config_params["wind_num"])
        wind_k = np.arange(-wind_num, wind_num + 1)
        
        N_new, S_new, T_new = self.compute_sufficient_statistics(len(emv.mean), wind_k, data.new_data, emv.mean, emv.cov, emv.mix, if_check_sum_r=False)

        return cliffs, N_new, S_new, T_new

    def compute_sufficient_statistics(self, cluster_nums, wind_k, raw_data, m, c, p, if_check_sum_r=True):
        num_observations = len(raw_data)
        raw_data = np.array(raw_data)
        r_batch = np.zeros((cluster_nums, len(wind_k), num_observations), dtype=float)

        for j in range(cluster_nums):
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                try:
                    likelihood = np.array([multivariate_normal.pdf(np.array([row[0], row[1] + 2 * np.pi * wrap_num ]), mean=m[j,:], cov=c[j,:,:], allow_singular=True) for row in raw_data]) * p[j]
                except:
                    likelihood = 1e-9
                r_batch[j,k,:] = likelihood
        r_batch[r_batch < np.finfo(float).eps] = 0
        
        if if_check_sum_r:
            if np.sum(r_batch) < self.combine_thres:
                return False
        
        sum_r = np.tile(np.sum(r_batch, axis=(0, 1)), (cluster_nums, len(wind_k), 1))
        r_batch = np.divide(r_batch, sum_r, out=np.zeros_like(r_batch), where=sum_r!=0)
        
        ### N_k
        N_new = np.zeros((cluster_nums, 1), dtype=float)
        for j in range(cluster_nums):
            sum_r_j = np.sum(r_batch[j,:,:])
            N_new[j,:] = sum_r_j / num_observations

        ### S_k
        S_new = np.zeros((cluster_nums, 2), dtype=float)
        for j in range(cluster_nums):
            t = np.zeros((num_observations, 2), dtype=float)
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                data_copy = raw_data.copy()
                data_copy[:, 1] += 2 * np.pi * wrap_num
                t += data_copy * np.tile(r_batch[j,k,:].reshape(-1,1), (1, 2))
            S_new[j,:] = np.sum(t, axis=0) / num_observations
        
        ### T_k
        T_new = np.zeros((cluster_nums, 2, 2), dtype=float)
        for j in range(cluster_nums):
            t = np.zeros((num_observations, 2, 2), dtype=float)
            for k in range(len(wind_k)):
                wrap_num = wind_k[k]
                data_copy = raw_data.copy()
                data_copy[:, 1] += 2 * np.pi * wrap_num
                d_mod = data_copy
                t[:,0,0] += d_mod[:,0]**2 * r_batch[j,k,:]
                t[:,1,1] += d_mod[:,1]**2 * r_batch[j,k,:]
                t[:,0,1] += d_mod[:,0] * d_mod[:, 1] * r_batch[j,k,:]
                t[:,1,0] = t[:,0,1]
                
            T_new[j,:,:] = np.sum(t, axis=0) / num_observations
            
        return N_new, S_new, T_new