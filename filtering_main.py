from initialise_wt_or_wf_Func import load_raw_wt_from_txt_file_and_temperature_from_csv
from WT_Class import WT
from Ploting.fast_plot_Func import series, scatter, time_series, hist
from numpy import ndarray
import numpy as np
import pandas as pd
from PowerCurve_Class import PowerCurveByMethodOfBins

if __name__ == '__main__':
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()

    # 看看温度和power output的相关性（散点图）
    # for this_wind_turbine in tuple([wind_turbines[2]]):
    for this_wind_turbine in wind_turbines:
        # this_wind_turbine.plot_wind_speed_to_active_power_output_mob((0,))

        this_wind_turbine.plot_wind_speed_to_active_power_output_scatter((0,))
        this_wind_turbine.plot_wind_speed_to_active_power_output_scatter((5, 4, 3, 2, 1, 0),
                                                                         show_category_color=(
                                                                             'indigo', 'r', 'b', 'aqua', 'k', 'g'),
                                                                         show_category_label=(
                                                                             'CAT-V', 'CAT-IV', 'CAT-III', 'CAT-II',
                                                                             'CAT-I', 'Normal'))
    # series(this_wind_turbine.measurements['environmental temperature'])
    # scatter(this_wind_turbine.measurements['environmental temperature'],
    #         this_wind_turbine.measurements['active power output'])
