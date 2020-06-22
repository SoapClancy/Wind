from File_Management.path_and_file_management_Func import remove_win10_max_path_limit
import time
from datetime import datetime
import numpy as np
from initialise_wt_or_wf_Func import load_raw_wt_from_txt_file_and_temperature_from_csv
from WT_Class import WT
from File_Management.load_save_Func import load_npy_file

"""
This (remove_win10_max_path_limit) is a default call to make the code run on Windows platform 
(and only Windows 10 is supported!)
Because the Windows 10 default path length limitation (MAX_PATH) is 256 characters, many load/save functions in
this project may have errors in reading the path
To restore the path_limit, call File_Management.path_management_Func.restore_win10_max_path_limit yourself
"""
remove_win10_max_path_limit()


def region_number_and_pct(wind_turbine: WT, path_):
    model_boundary = load_npy_file(path_ + 'model_boundary.npy')
    model_boundary = np.concatenate((np.array([0]), model_boundary))
    model_boundary_roll = np.roll(model_boundary, -1)
    model_boundary_roll[-1] = np.inf
    flag = this_wind_turbine.outlier_category == 0
    total_num = np.sum(flag)
    for i in range(model_boundary.__len__()):
        min_ = model_boundary[i]
        max_ = model_boundary_roll[i]
        this_num = np.sum(np.bitwise_and(wind_turbine.measurements['wind speed'].values[flag] >= min_,
                                         wind_turbine.measurements['wind speed'].values[flag] < max_))
        this_num_pct = this_num / total_num * 100
        print('this_num = {}, this_num_pct = {}'.format(this_num, this_num_pct))


if __name__ == '__main__':
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()

    this_wind_turbine = wind_turbines[2]
    path = this_wind_turbine.results_path + '4d_cvine_gmcm_model/' + this_wind_turbine.__str__() + '/'
    region_number_and_pct(this_wind_turbine, path)
