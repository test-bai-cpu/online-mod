from tqdm import tqdm

from mod_build_base import BuildMoDBase
import utils


class IntervalBuildMoD(BuildMoDBase):
    
    def __init__(
        self,
        config_file: str,
        current_cliff: str,
        output_cliff_folder: str,
        save_fig_folder: str,
        build_type: str
    ) -> None:
        super().__init__(config_file, current_cliff, output_cliff_folder, save_fig_folder)
        self.build_type = build_type
        
    def updateMoD(self, new_batch_file, output_file_name):
        change_grid_centers = self.data_batches.process_new_batch(new_batch_file)
        
        for _, key in tqdm(enumerate(change_grid_centers), total=len(change_grid_centers), desc='Processing'):
            data = self.data_batches.grid_data[key]
            cliffs, _, _, _ = self.build_cliff(key, data, if_build_with_new_data=True)
            utils.save_cliff_csv_rows(f"{self.cliff_csv_folder}/{output_file_name}.csv", cliffs)