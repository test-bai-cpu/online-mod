from typing import Tuple

import numpy as np

from sklearn.utils.parallel import Parallel, delayed

from batch_process_data import GridData
import utils


class MeanShift:
    
    def __init__(
        self,
        kernel: str = "gaussian",
        grid_data: GridData = None,
        convergence_threshold: float = 1e-6,
        group_distance_tolerance: float = 1e-2,
        cluster_all: bool = True,
        too_few_data_thres: int = 5,
        max_iteration: int = 500,
        if_build_with_new_data: bool = False,
        
    ) -> None:
        self.kernel = kernel
        if if_build_with_new_data:
            self.input_data = grid_data.new_data
        else:
            self.input_data = grid_data.data

        self.convergence_threshold = convergence_threshold
        self.group_distance_tolerance = group_distance_tolerance
        self.cluster_all = cluster_all
        self.too_few_data_thres = too_few_data_thres
        self.max_iteration = max_iteration
        
        self.kernel_bandwidth = self._compute_isotropic_kernel_bandwidth()
        self.cluster_centers = None
        self.data_cluster_labels = None
        self.mixing_factors = None
        self.mean_values = None
        self.covariances = None

    def run_mean_shift(self) -> None:
        self._fit_parallel()
        
        self._prune_cluster_step1()
        
        if len(self.cluster_centers) == 0:
            return

        self._compute_mean_shift_result()
        
        self._prune_cluster_step2()

    def _fit_parallel(self) -> None:
        shift_points = np.array(self.input_data)
        n_samples = self.input_data.shape[0]
        
        shift_points = Parallel()(delayed(self._mean_shift_single_point)(point) for point in shift_points)   

        center_intensity_dict = {}
        for point in shift_points:
            center_intensity_dict[tuple(point)] = self._get_nearest_neighbor_by_radius(point)
        
        sorted_by_intensity = sorted(
            center_intensity_dict.items(),
            key=lambda tup: (tup[1], tup[0]),
            reverse=True,
        )
        
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        cluster_centers_list = [sorted_centers[0]]

        for sorted_center in sorted_centers[1:]:
            if any(utils.distance_wrap_2d(x, sorted_center) < self.kernel_bandwidth
                   for x in cluster_centers_list):
                continue
            cluster_centers_list.append(sorted_center)
        
        cluster_centers = np.round(np.array(cluster_centers_list), 3)        

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        n_samples = self.input_data.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        distances, indices = self._get_nearest_cluster_center_and_distance(cluster_centers)
        if self.cluster_all:
            labels = indices.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= self.kernel_bandwidth
            labels[bool_selector] = indices.flatten()[bool_selector]
        
        self.cluster_centers = cluster_centers
        self.data_cluster_labels = labels

    def _compute_mean_shift_result(self) -> None:
        mixing_factors = []
        mean_values = []
        covariances = []
           
        # compute GMM parameters
        for index in range(len(self.cluster_centers)):
            data = self.input_data[self.data_cluster_labels == index]
            
            mixing_factors.append(len(data) / len(self.input_data[self.data_cluster_labels != -1]))
            mean, cov = utils.mean_cov_2d_vec(np.array(data))
            mean_values.append(mean)
            covariances.append(cov)
        
        self.mixing_factors = np.array(mixing_factors)
        self.mean_values = np.array(mean_values)
        self.covariances = np.array(covariances)

    def fit_v1(self) -> None:
        shift_points = np.array(self.input_data)
        n_samples = self.input_data.shape[0]
        still_shifting = [True] * n_samples
        max_dist = 1
        
        while max_dist > self.convergence_threshold:
            max_dist = 0
            for i in range(0, len(shift_points)):
                if not still_shifting[i]:
                    continue
                p_new = self._shift_point(shift_points[i])
                dist = utils.distance_wrap_2d(p_new, shift_points[i])
                
                if dist > max_dist:
                    max_dist = dist
                if dist < self.convergence_threshold:
                    still_shifting[i] = False
                shift_points[i] = p_new

        center_intensity_dict = {}
        for point in shift_points:
            center_intensity_dict[tuple(point)] = self._get_nearest_neighbor_by_radius(point)
        
        sorted_by_intensity = sorted(
            center_intensity_dict.items(),
            key=lambda tup: (tup[1], tup[0]),
            reverse=True,
        )
        
        sorted_centers = np.array([tup[0] for tup in sorted_by_intensity])
        cluster_centers_list = [sorted_centers[0]]

        for sorted_center in sorted_centers[1:]:
            if any(utils.distance_wrap_2d(x, sorted_center) < self.kernel_bandwidth
                   for x in cluster_centers_list):
                continue
            cluster_centers_list.append(sorted_center)
        
        cluster_centers = np.round(np.array(cluster_centers_list), 3)        

        # ASSIGN LABELS: a point belongs to the cluster that it is closest to
        n_samples = self.input_data.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        distances, indices = self._get_nearest_cluster_center_and_distance(cluster_centers)
        if self.cluster_all:
            labels = indices.flatten()
        else:
            labels.fill(-1)
            bool_selector = distances.flatten() <= self.kernel_bandwidth
            labels[bool_selector] = indices.flatten()[bool_selector]
        
        self.cluster_centers = cluster_centers
        self.data_cluster_labels = labels

    def _compute_non_isotropic_kernel_bandwidth(self) -> np.ndarray:
        data = np.array(self.input_data)
        _, cov = utils.mean_cov_2d_vec(np.array(data))
        
        sigma_l = cov[0, 0]
        sigma_c = cov[1, 1]
        data_count_size = len(data)
        h_l = ((4 * sigma_l ** 5) / (3 * data_count_size)) ** (1 / 5)
        h_c = ((4 * sigma_c ** 5) / (3 * data_count_size)) ** (1 / 5)
        if h_l < np.finfo(float).eps or h_c < np.finfo(float).eps:
            h_l += 10 ** -10
            h_c += 10 ** -10
            
        return np.array([[h_l, 0], [0, h_c]])

    def _compute_isotropic_kernel_bandwidth(self) -> float:
        data = np.array(self.input_data)
        std = utils.std_of_circular_linear_data(data)
        
        data_count_size = len(data)
        bandwidth = ((4 * std ** 5) / (3 * data_count_size)) ** (1 / 5)
        if bandwidth < np.finfo(float).eps:
            bandwidth += 10 ** -10
        
        return bandwidth

    def _mean_shift_single_point(self, point: np.ndarray) -> np.ndarray:
        dist = np.inf
        iter = 0
        current_point = point
        while dist > self.convergence_threshold and iter < self.max_iteration:
            previous_point = current_point
            current_point = self._shift_point(current_point)
            dist = utils.distance_wrap_2d(current_point, previous_point)
            iter += 1

        return current_point

    def _prune_cluster_step1(self) -> None:
        cluster_centers = self.cluster_centers
        labels = self.data_cluster_labels
        
        # Remove clusters when they are too close to each other
        for i in range(len(cluster_centers)):
            for j in range(i+1, len(cluster_centers)):
                if utils.distance_wrap_2d(cluster_centers[i], cluster_centers[j]) < self.group_distance_tolerance:
                    labels[labels == j] = i
        
        # Remove clusters with too few data points, and set to unassigned data label as -1
        counts = np.bincount(labels, minlength=len(cluster_centers))
        clusters_to_remove = np.where(counts < self.too_few_data_thres)[0]

        updated_labels = labels.copy()

        for cluster in clusters_to_remove:
            updated_labels = [-1 if label == cluster else label for label in updated_labels]

        new_cluster_centers = np.delete(cluster_centers, clusters_to_remove, axis=0)
        
        for new_index, old_index in enumerate(np.delete(np.arange(len(cluster_centers)), clusters_to_remove)):
            updated_labels = [new_index if label == old_index else label for label in updated_labels]

        final_labels = np.array(updated_labels)
        
        self.cluster_centers = new_cluster_centers
        self.data_cluster_labels = final_labels

    def _prune_cluster_step2(self) -> None: 
        cluster_centers = self.cluster_centers
        labels = self.data_cluster_labels
        clusters_to_remove = []
        
        for j in range(len(cluster_centers)):
            cov = self.covariances[j,:,:]
            try:
                np.linalg.cholesky(cov)
                chol_f = 0
            except np.linalg.LinAlgError:
                chol_f = 1
                
            if (chol_f != 0 or cov[0,0] < 10**(-10) or cov[1,1] < 10**(-10)):
                clusters_to_remove.append(j)
                
        clusters_to_remove = np.array(clusters_to_remove)

        if clusters_to_remove.size == 0:
            return
        
        self.covariances = np.delete(self.covariances, clusters_to_remove, axis=0)
        self.mean_values = np.delete(self.mean_values, clusters_to_remove, axis=0)
        self.mixing_factors = np.delete(self.mixing_factors, clusters_to_remove, axis=0)
        
        updated_labels = labels.copy()
        
        for cluster in clusters_to_remove:
            updated_labels = [-1 if label == cluster else label for label in updated_labels]
        
        new_cluster_centers = np.delete(cluster_centers, clusters_to_remove, axis=0)

        for new_index, old_index in enumerate(np.delete(np.arange(len(cluster_centers)), clusters_to_remove)):
            updated_labels = [new_index if label == old_index else label for label in updated_labels]
        
        final_labels = np.array(updated_labels)
        
        self.cluster_centers = new_cluster_centers
        self.data_cluster_labels = final_labels

    def _get_nearest_cluster_center_and_distance(
            self, cluster_centers: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Assign samples to the nearest cluster
        distances, indices = [], []

        for sample in np.array(self.input_data):
            distances = [
                utils.distance_wrap_2d(sample, cluster_center)
                for cluster_center in cluster_centers
            ]
            min_index = np.argmin(distances)
            distances.append(distances[min_index])
            indices.append(min_index)

        return np.array(distances), np.array(indices)
    
    def _shift_point(self, point: np.ndarray) -> np.ndarray:
        points = np.array(self.input_data)
        point_rep = np.tile(point, [len(points), 1])
        dist = utils.distance_wrap_2d_vectorized(point_rep, points)
        point_weights = utils.gaussian_kernel(dist, self.kernel_bandwidth)
        shifted_point = utils.weighted_mean_2d_vec(points, point_weights)
        
        return shifted_point

    def _get_nearest_neighbor_by_radius(self, cluster_center: np.ndarray) -> int:
        neighbor_count = sum(map(lambda x:
                                 utils.distance_wrap_2d(x, cluster_center) <
                                 self.kernel_bandwidth,
                                 np.array(self.input_data))
                             )

        return int(neighbor_count)