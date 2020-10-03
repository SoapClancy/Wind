from WT_WF_Class import WF, WT
import pandas as pd
from project_utils import project_path_
import numpy as np
from numpy import ndarray
import datetime
import re
from File_Management.path_and_file_management_Func import list_all_specific_format_files_in_a_folder_path
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save
from scipy.io import loadmat
import os
from pathlib import Path
from Ploting.fast_plot_Func import *
from TimeSeries_Class import TimeSeries, merge_two_time_series_df
import re
from File_Management.path_and_file_management_Func import try_to_find_file
from PowerCurve_Class import PowerCurveFittedBy8PLF
from typing import Tuple
from collections import OrderedDict
import getpass

Croatia_RAW_DATA_PATH = Path(r"C:\Users\\" + getpass.getuser() + r"\OneDrive\PhD\01-PhDProject\Database\Croatia\03")


def load_croatia_data(this_wind_farm_name: str = None) -> OrderedDict:
    wind_farm = OrderedDict()
    # WF rated power mapper
    wf_rated_power_mapper = {
        'Zelengrad': 42,
        'Vratarusa': 42
    }

    for dir_name, subdir_list, file_list in os.walk(Croatia_RAW_DATA_PATH):
        if subdir_list:
            continue
        # May only need to fetch specific wind farm
        if this_wind_farm_name is not None:
            if Path(dir_name).name != this_wind_farm_name:
                continue

        wind_farm_name = Path(dir_name).name  # type: str

        # if try_to_find_file(''.join((wind_farm_name, '.png'))):
        #     continue

        def one_variable_reading(file_postfix: str):
            one_variable = pd.read_csv(Path(dir_name) / ''.join((wind_farm_name, file_postfix)))
            one_variable.index = pd.DatetimeIndex(
                pd.to_datetime(
                    one_variable.iloc[:, 1],
                    # format='%Y-%m-%d %H:%M:%S'
                )
            )
            one_variable.drop(one_variable.columns[[0, 1]], axis=1, inplace=True)
            return one_variable

        wind_farm_wind_speed = one_variable_reading('_scada_ws.csv')
        wind_farm_power_output = one_variable_reading('_scada_pow.csv')

        wind_farm_basic = pd.merge(wind_farm_wind_speed,
                                   wind_farm_power_output,
                                   how='outer',
                                   left_index=True,
                                   right_index=True)
        wind_farm_basic.rename(columns={'WS': 'wind speed',
                                        'POWER': 'active power output'},
                               errors='raise',
                               inplace=True)

        wind_farm[wind_farm_name] = WF(
            wind_farm_basic,
            obj_name=f'{wind_farm_name} WF',
            rated_active_power_output=wf_rated_power_mapper[wind_farm_name],
            predictor_names=('wind speed',),
            dependant_names=('active power output',)
        )

        # scatter(wind_farm_basic.iloc[365 * 24 * 30:365 * 24 * 90, 0].values,
        #         wind_farm_basic.iloc[365 * 24 * 30:365 * 24 * 90, 1].values / 42,
        #         color='b',
        #         x_label='Wind speed (m/s)',
        #         x_lim=(-0.05, 28.5),
        #         y_lim=(-0.005, 1.05),
        #         y_label='Power output (p.u.)',
        #         save_file_=wind_farm_name)
        return wind_farm


def convert_datetime_str_in_txt_to_datetime64(ts: ndarray, from_: str = 'txt'):
    time_stamp = []
    for i in ts:
        if from_ == 'txt':
            if re.search(r':', i) is None:
                i = i + ' 00:00:00'
            if re.search(r'.*\d+:\d+:\d+$', i) is None:
                i = i + ':00'
            time_stamp.append(datetime.datetime.strptime(i, '%d/%m/%Y %H:%M:%S'))
        else:
            if re.search(r':', i) is None:
                i = i + ' 00:00:00'
            if re.search(r'.*\d+:\d+:\d+$', i) is None:
                i = i + ':00'
            i = re.sub(r'-', '/', i)
            time_stamp.append(datetime.datetime.strptime(i, '%Y/%m/%d %H:%M:%S'))
    return pd.Series(time_stamp)


def __load_time_stamp_from_met_mast_txt():
    ts = pd.read_csv(project_path_ / 'Data/Raw_measurements/Darly/Met_Mast.txt', sep='\t')['PCTimeStamp'].values
    return convert_datetime_str_in_txt_to_datetime64(ts)


def load_weather_data():
    files = list_all_specific_format_files_in_a_folder_path(project_path_ / 'Data/Raw_measurements/Darly/', 'csv', '')
    pattern = re.compile(r'^.*\d+Weather')
    weather_data = pd.DataFrame({'time': [],
                                 'environmental temperature': [],
                                 'relative humidity': [],
                                 'barometric pressure': []})
    for file in files:
        regex = pattern.match(file)
        if regex is not None:
            one_reading = pd.read_csv(file, sep=',', skiprows=list(range(280)))
            one_reading_pd = pd.DataFrame(
                {
                    'time': convert_datetime_str_in_txt_to_datetime64(one_reading['ob_time'].values[:-1], ''),
                    'environmental temperature': one_reading['air_temperature'].values[:-1],
                    'relative humidity': one_reading['rltv_hum'].values[:-1],
                    'barometric pressure': one_reading['stn_pres'].values[:-1],
                }
            )
            weather_data = pd.concat([weather_data, one_reading_pd])
    return weather_data


def load_raw_wt_from_txt_file_and_temperature_from_csv() -> Tuple[WT, ...]:
    wind_turbines = []
    files = list_all_specific_format_files_in_a_folder_path(project_path_ / 'Data/Raw_measurements/Darly/', 'txt',
                                                            order='')
    pattern = re.compile(r'^.*Turbine(\d+)\.txt$')
    weather_data = load_weather_data()
    for file in files:
        regex = pattern.match(file)
        if regex is not None:
            name = 'Darly WT' + regex.group(1)

            @load_exist_pkl_file_otherwise_run_and_save(
                project_path_ / f'Data/Raw_measurements/Darly/{name} measurements.pkl'
            )
            def get_measurements():
                one_reading = pd.read_csv(file, sep='\t')
                measurements = pd.DataFrame({'time': convert_datetime_str_in_txt_to_datetime64(
                    one_reading['PCTimeStamp']),
                    'wind speed': one_reading.iloc[:, 1],
                    'wind speed std.': one_reading.iloc[:, 2],
                    'active power output': one_reading.iloc[:, 5] * 6 / 1000,
                    'reactive power output': one_reading.iloc[:, 6] * 6 / 1000,
                    'absolute wind direction': one_reading.iloc[:, 3],
                    'relative wind direction': one_reading.iloc[:, 4]})
                measurements = pd.merge(measurements, weather_data, on='time', how='left')
                # # 因为第一天的零点没有值，所以人工加上
                # measurements.loc[0, 'environmental temperature'] = float(
                #     measurements['environmental temperature'].values[144])
                measurements[['environmental temperature', 'relative humidity', 'barometric pressure']] = measurements[
                    ['environmental temperature', 'relative humidity', 'barometric pressure']
                ].interpolate()
                if float(np.sum(np.diff(measurements['time'].values.astype('float')) <
                                np.mean(np.diff(measurements['time'].values.astype('float'))) / 2)) > 0:
                    raise Exception('Duplicated time stamp found')
                return measurements.set_index('time')

            data = get_measurements()
            this_wt = WT(data,
                         obj_name=name,
                         predictor_names=tuple(data.columns.drop('active power output')),
                         dependant_names=('active power output',))
            wind_turbines.append(this_wt)
    return tuple(wind_turbines)


def create_dalry_wind_farm_obj_using_wf_filling_missing_old():
    @load_exist_pkl_file_otherwise_run_and_save(project_path_ /
                                                'Data/Results/filling_missing/WF_filling_missing_old.pkl')
    def get_measurements():
        ws_p = loadmat(os.path.join(project_path_, 'Data/Results/filling_missing/WF_filling_missing_old.mat'))
        ws_ = ws_p['corrected_wind_speed'].flatten()
        p_ = ws_p['corrected_power_output'].flatten()
        temp_wt = load_raw_wt_from_txt_file_and_temperature_from_csv()[0]
        time_ = temp_wt.measurements['time'].values[:ws_.size]
        temperature_ = temp_wt.measurements['environmental temperature'].values[:ws_.size]
        return time_, ws_, p_, temperature_

    time, ws, p, temperature = get_measurements()
    dalry_wf = WF(
        data=pd.DataFrame(
            {
                'time': time,
                'wind speed': ws,
                'active power output': p,
                'environmental temperature': temperature
            }).set_index('time'),
        obj_name='Dalry',
        predictor_names=('wind speed', 'environmental temperature'),
        dependant_names=('active power output',),
        rated_active_power_output=3000 * 6
    )
    return dalry_wf


def load_raw_36_wts_in_nez():
    wts = loadmat(Path(project_path_) / 'Data/Raw_measurements/NEZ/WT_Classification.mat')
    return wts


def load_high_resol_for_averaging_effects_analysis():
    folder_path = Path(project_path_) / 'Data/Raw_measurements/high_resol/'
    high_resol = {}
    for this_file in folder_path.iterdir():
        if this_file.match(r'*.mat'):
            file_name = re.match(r"\b.*(?=.mat)", this_file.name)[0]
            wts = loadmat(str(this_file))
            data = wts[file_name][:, [10, 11]]
            data[:, -1] *= 3000
            high_resol.setdefault(file_name, data)
    return high_resol


if __name__ == '__main__':
    # temp_func()
    # test_wf = create_dalry_wind_farm_obj_using_wf_filling_missing_old()
    # test_pc = test_wf.power_curve_by_method_of_bins()
    # test_pc = PowerCurveFittedBy8PLF.init_from_power_curve_by_method_of_bins(test_pc)
    # test_pc.update_params(*[1.0, -6.06147393e-03, -9.784173449e+00, 2.50094859e+01,
    #                         1.1163476e+01, 2.6918891e+01, 2.5509312e-01, .25560337e+00])
    # test_pc.fit(ga_algorithm_param={'max_num_iteration': 100,
    #                                 'max_iteration_without_improv': 100},
    #             params_init_scheme='self',
    #             run_n_times=5)
    # test_pc.plot()

    tt = load_croatia_data('Zelengrad')
    # tt = load_raw_wt_from_txt_file_and_temperature_from_csv()

    # for i in tt:
    #     scatter(i.measurements['wind speed'].values,
    #             i.measurements['active power output'].values)

    # scatter(tt[1].measurements['wind speed'].values,
    #         tt[1].measurements['active power output'].values / 3000,
    #         color='b', alpha=0.5, s=1,
    #         x_label='Wind speed [m/s]',
    #         y_label='Power output [p.u.]',
    #         x_lim=(-.05, 29.6),
    #         y_lim=(-0.01, 1.06)
    #         )

    # tt2 = load_raw_36_wts_in_nez()
    # for key, val in tt2.items():
    #     if not re.match(r"WT.*", key):
    #         continue
    #     if key!='WT15':
    #         continue
    #     scatter(val[:, 1],
    #             val[:, 2],
    #             s=0.5,
    #             color='b',
    #             x_label='Wind speed (m/s)',
    #             x_lim=(-0.05, 28.5),
    #             y_lim=(-0.005, 1.05),
    #             y_label='Power output (p.u.)',
    #             save_file_='wt15'
    #             )

    # hi = load_high_resol_for_averaging_effects_analysis()
