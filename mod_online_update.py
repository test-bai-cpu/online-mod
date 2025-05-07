from typing import List

from tqdm import tqdm

import numpy as np

from batch_process_data import GridData
from mod_build_base import BuildMoDBase
import utils


class OnlineUpdateMoD(BuildMoDBase):
    
    def __init__(
        self,
        config_file: str,
        current_cliff: str,
        output_cliff_folder: str,
        save_fig_folder: str,
    ) -> None:
        super().__init__(config_file, current_cliff, output_cliff_folder, save_fig_folder)
        
        self.decay_rate = float(self.config_params["decay_rate"])
        self.combine_thres = float(self.config_params["combine_thres"])
        
    def updateMoD(self, new_batch_file, output_file_name):
        print("------decay rate: ", self.decay_rate, "------")
        change_grid_centers = self.data_batches.process_new_batch(new_batch_file)

        new_build_cnt = 0
        update_cnt = 0
        
        for _, key in tqdm(enumerate(change_grid_centers), total=len(change_grid_centers), desc='Processing'):
            data = self.data_batches.grid_data[key]

            if len(data.data) == len(data.new_data):
                data.importance_value = len(data.new_data)
                cliffs, N_cur, S_cur, T_cur = self.build_cliff(key, data)
                data.cliff = cliffs
                data.N_cur = N_cur
                data.S_cur = S_cur
                data.T_cur = T_cur

                utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}.csv", cliffs)
                new_build_cnt += 1
            else:
                data.importance_value = data.importance_value * self.decay_rate

                learning_rate = len(data.new_data) / (data.importance_value + len(data.new_data))
                results = self.update_cliff(key, data, learning_rate, s_type="sEM")
                update_cnt += 1
                
                if results == None:
                    cliffs, _, _, _ = self.build_cliff(key, data, if_build_with_new_data=True)
                    new_build_cnt += 1
                    cliffs, N_cur, S_cur, T_cur = self.combine_cliff(data, cliffs)
                    utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}.csv", cliffs)
                else:
                    cliffs, N_cur, S_cur, T_cur = results
                    utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}.csv", cliffs)
                    
                data.cliff = cliffs
                data.N_cur = N_cur
                data.S_cur = S_cur
                data.T_cur = T_cur
                
                data.importance_value = data.importance_value + len(data.new_data)
        
        print("New build: ", new_build_cnt, " Update: ", update_cnt)
        
        # Check which grid has no new data
        unchange_grid_centers = set(self.data_batches.grid_data.keys()) - set(change_grid_centers)
        for key in unchange_grid_centers:
            data = self.data_batches.grid_data[key]
            data.importance_value = data.importance_value * self.decay_rate
            if data.cliff == []:
                continue
            utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}.csv", data.cliff)
        
    def combine_cliff(self, data, update_cliff):
        before_cliff = data.cliff
        before_p = np.array([row[8] for row in before_cliff])
        update_p = np.array([row[8] for row in update_cliff])
        
        update_count = len(data.new_data)
        total_count = data.importance_value + update_count
        before_count = data.importance_value
        
        before_p_new = before_p / (sum(before_p)) * (before_count/total_count)
        for i, row in enumerate(before_cliff):
            row[8] = before_p_new[i]
        update_p_new = update_p / (sum(update_p)) * (update_count/total_count)
        for i, row in enumerate(update_cliff):
            row[8] = update_p_new[i]
        
        total_cliff = before_cliff + update_cliff
        
        m = np.array([row[2:4] for row in total_cliff])
        c = np.array([[[row[4], row[5]], [row[6], row[7]]] for row in total_cliff])
        p = np.array([row[8] for row in total_cliff])
        wind_num = int(self.config_params["wind_num"])
        wind_k = np.arange(-wind_num, wind_num + 1)
        N_new, S_new, T_new = self.compute_sufficient_statistics(len(total_cliff), wind_k, data.new_data, m, c, p, if_check_sum_r=False)

        return total_cliff, N_new, S_new, T_new

    def update_cliff(self, key, data: GridData, learning_rate: float, s_type: str = "sEM") -> List:
        before_cliff = data.cliff

        if len(before_cliff) == 0:
            return self.build_cliff(key, data, if_build_with_new_data=True)

        raw_data = np.array(data.new_data)    
        
        wind_num = int(self.config_params["wind_num"])
        wind_k = np.arange(-wind_num, wind_num + 1)
        cluster_nums = len(before_cliff)
        
        m = np.array([row[2:4] for row in before_cliff])
        c = np.array([[[row[4], row[5]], [row[6], row[7]]] for row in before_cliff])
        p = np.array([row[8] for row in before_cliff])

        N_cur = np.array(data.N_cur)
        S_cur = np.array(data.S_cur)
        T_cur = np.array(data.T_cur)

        results = self.compute_sufficient_statistics(cluster_nums, wind_k, raw_data, m, c, p)
        if results is False:
            return
        else:
            N_new, S_new, T_new = results
        
        N_cur = N_cur + learning_rate * (N_new - N_cur)
        S_cur = S_cur + learning_rate * (S_new - S_cur)
        T_cur = T_cur + learning_rate * (T_new - T_cur)

        ### Update pi
        p = np.ones((cluster_nums)) * (1 / cluster_nums)
        for j in range(cluster_nums):
            p[j] = N_cur[j,:]
        p = p / np.sum(p)
        
        ### Update mu
        m = np.zeros((cluster_nums, 2), dtype=float)
        for j in range(cluster_nums):
            S_k = S_cur[j,:]
            N_k = N_cur[j,:]
            if N_k != 0:
                m[j,:] = np.divide(S_k, N_k)
            else:
                m[j,:] = np.zeros_like(m[j,:])

        ### Update cov
        c = np.zeros((cluster_nums,2,2), dtype=float)
        for j in range(cluster_nums):
            T_k = T_cur[j,:,:]
            N_k = N_cur[j,:]
            mu_k = m[j,:]
                    
            if N_k != 0:
                c[j,:,:] = np.divide(T_k, N_k) - np.outer(mu_k, mu_k)
            else:
                c[j,:,:] = np.zeros_like(c[j,:,:])

        cliffs = []
        for cluster_i in range(len(m)):
            save_row = [
                key[0], key[1],
                m[cluster_i,0], m[cluster_i,1],
                c[cluster_i,0,0], c[cluster_i,0,1], c[cluster_i,1,0], c[cluster_i,1,1],
                p[cluster_i], data.motion_ratio
            ]

            rounded_save_row = [round(value, 5) if not (value is None) else value for value in save_row]
            cliffs.append(rounded_save_row)
        
        return cliffs, N_cur, S_cur, T_cur
