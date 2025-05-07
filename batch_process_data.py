import pandas as pd
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


class BatchDataProcess:
    
    def __init__(
        self,
        radius: float,
        step: float,
        dataset_type: str,
    ) -> None:
        self.radius = radius
        self.step = step
        self.dataset_type = dataset_type
        self.grid_data = {}
        self.initialized = False

    def process_new_batch(self, batch_data_file):
        if self.dataset_type == 'OBSERVE':
            data = batch_data_file
            data['motion_angle'] = np.round(np.mod(data['motion_angle'], 2 * np.pi), 3)
            new_data = data[['time', 'x', 'y', 'velocity', 'motion_angle']]

        change_grid_centers = self._split_data_to_grid(new_data)
        
        return change_grid_centers

    def _split_data_to_grid(self, data):
        change_grid_centers = []
        
        self._update_boundaries(data)

        grid_centers = self._create_grid_centers()
        
        # print(grid_centers)

        for grid_center in grid_centers:
            distances = np.sqrt((data['x'] - grid_center[0]) ** 2 + (data['y'] - grid_center[1]) ** 2)
            within_radius = data[distances <= self.radius]
            within_radius = within_radius[['velocity', 'motion_angle']]
            # within_radius = within_radius[['x', 'y', 'velocity', 'motion_angle']]
            # print(within_radius)

            if not within_radius.empty:
                if grid_center in self.grid_data:
                    self.grid_data[grid_center].data = pd.concat([self.grid_data[grid_center].data, within_radius], ignore_index=True)
                else:
                    self.grid_data[grid_center] = GridData(grid_center, within_radius)
                self.grid_data[grid_center].new_data = within_radius
                self.grid_data[grid_center].motion_ratio = len(self.grid_data[grid_center].data)
                    
                change_grid_centers.append(grid_center)
                    
        return change_grid_centers

    def _create_grid_centers(self):
        if self.x_min == self.x_max:
            self.x_max += self.step
        if self.y_min == self.y_max:
            self.y_max += self.step
            
        x_centers = np.arange(self.x_min, self.x_max, self.step)
        y_centers = np.arange(self.y_min, self.y_max, self.step)
        return [(round(x, 3), round(y, 3)) for x in x_centers for y in y_centers]

    def _adjust_boundaries_to_step(self, value, direction):
        if direction == 'lower':
            return np.floor(value / self.step) * self.step
        elif direction == 'upper':
            return np.ceil(value / self.step) * self.step

    def _update_boundaries(self, data):
        adj_x_min = self._adjust_boundaries_to_step(data['x'].min(), 'lower')
        adj_x_max = self._adjust_boundaries_to_step(data['x'].max(), 'upper')
        adj_y_min = self._adjust_boundaries_to_step(data['y'].min(), 'lower')
        adj_y_max = self._adjust_boundaries_to_step(data['y'].max(), 'upper')
        
        # print(f"adj_x_min: {adj_x_min}, adj_x_max: {adj_x_max}, adj_y_min: {adj_y_min}, adj_y_max: {adj_y_max}")
        
        if not self.initialized:
            self.x_min = adj_x_min
            self.x_max = adj_x_max
            self.y_min = adj_y_min
            self.y_max = adj_y_max
            self.initialized = True
        else:
            self.x_min = min(self.x_min, adj_x_min)
            self.x_max = max(self.x_max, adj_x_max)
            self.y_min = min(self.y_min, adj_y_min)
            self.y_max = max(self.y_max, adj_y_max)
        
        print(f"x_min: {self.x_min}, x_max: {self.x_max}, y_min: {self.y_min}, y_max: {self.y_max}")

    def _compute_motion_ratio(self):
        for grid_data in self.grid_data.values():
            grid_data.motion_ratio = len(grid_data.data)
        # max_motion = 0
        # for grid_data in self.grid_data.values():
        #     data_len = len(grid_data.data)
        #     if data_len > max_motion:
        #         max_motion = data_len
        
        # for grid_data in self.grid_data.values():
        #     grid_data.motion_ratio = len(grid_data.data) / max_motion
        
    def get_data_for_grid(self, grid_x, grid_y):
        grid_key = (grid_x, grid_y)
        
        grid_data = self.grid_data.get(grid_key, None)
        
        if grid_data is None:
            print("In procees_data.py, get_data_for_grid: grid_key not found")
        
        return grid_data
        
    def plot_data_by_color_all_grid(self):
        # Create a color map to color each grid's data differently
        # colors = plt.cm.rainbow(np.linspace(0, 1, len(self.grid_data)))
        # colors = plt.cm.rainbow(np.linspace(0, 1, 15))
        colors = [
            "blue", "green", "red", "cyan", "magenta", 
            "yellow", "black", "white", "orange", "purple",
            "brown", "pink", "lime", "navy", "gray",
            "olive", "maroon", "teal", "aqua", "silver", 
            "gold", "lavender", "beige", "ivory", "chocolate", 
            "salmon"]
        
        grid_centers = list(self.grid_data.keys())
        # print(len(grid_centers))
        
        for i in range(len(grid_centers)):
            grid_center = grid_centers[i]
            color = colors[i%26]
            self.plot_for_one_grid(grid_center[0], grid_center[1], color=color)
        
    def plot_for_one_grid(self, grid_x, grid_y, color='blue'):
        grid_data = self.get_data_for_grid(grid_x, grid_y)
        plt.scatter(grid_data.data['x'], grid_data.data['y'], color=color, label=f"Grid {grid_data.grid_center}", alpha=1)


class GridData:
    def __init__(
        self,
        grid_center: Tuple[float, float],
        data: pd.DataFrame,
    ) -> None:
        self.grid_center = grid_center
        self.data = data
        self.new_data = None
        self.motion_ratio = None
        self.importance_value = 0
        
        self.cliff = []
