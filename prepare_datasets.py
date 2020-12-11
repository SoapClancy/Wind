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
from functools import reduce
from Ploting.data_availability_plot_Func import data_availability_plot
from Ploting.wind_rose_plot_Func import wind_rose_plot
import copy

Croatia_RAW_DATA_PATH = Path(r"C:\Users\\" + getpass.getuser() + r"\OneDrive\PhD\01-PhDProject\Database\Croatia\03")
Croatia_WF_LOCATION_MAPPER = {
    'Benkovac': (44.085, 15.608, 274.0),
    'Bruska': (44.093, 15.736, 612.4),
    'Jelinak': (43.554, 16.167, 536.7),
    'Katuni': (43.474, 16.900, 414.8),
    'Lukovac': (43.535, 16.911, 640.4),
    'Ogorje': (43.721, 16.518, 848.5),
    'Pometenobrdo': (43.614, 16.474, 601.9),
    'Ponikve': (42.867, 17.606, 436.2),
    'Rudine': (42.824, 17.796, 335.6),
    'VelikaGlava': (43.739, 16.086, 453.2),
    'VelikaPopina': (44.270, 16.010, 919.0),
    'Vratarusa': (45.050, 14.930, 700.0),
    'Zelengrad': (44.125, 15.738, 484.6),
}

this_wind_farm_name = 'Zelengrad'
ws_pout_only = False
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
def load_croatia_data(this_wind_farm_name: str = None, ws_pout_only: bool = True) -> OrderedDict:
    wind_farm = OrderedDict()
    # WF rated power mapper
    wf_rated_power_mapper = {
        'Benkovac': 9.2,
        'Bruska': 36.8,
        'Jelinak': 30,
        'Katuni': 34.2,
        'Lukovac': 48.75,
        'Ogorje': 42,
        'Pometenobrdo': 17.5,  # ###########please check
        'Ponikve': 36.8,  # ###########please check
        'Rudine': 34.2,
        'VelikaGlava': 43.7,
        'VelikaPopina': 9.2,
        'Vratarusa': 42,
        'Zelengrad': 42,
    }

    for dir_name, subdir_list, file_list in os.walk(Croatia_RAW_DATA_PATH):
        if subdir_list:
            continue
        # May only need to fetch specific wind farm
        if this_wind_farm_name is not None:
            if Path(dir_name).name != this_wind_farm_name:
                continue

        wind_farm_name = Path(dir_name).name  # type: str


        def one_variable_reading(file_postfix: str, col_name: str = None):
            file_path = Path(dir_name) / ''.join((wind_farm_name, file_postfix))
            if try_to_find_file(file_path):
                one_variable = pd.read_csv(file_path)
                one_variable.index = pd.DatetimeIndex(
                    pd.to_datetime(
                        one_variable.iloc[:, 1],
                        # format='%Y-%m-%d %H:%M:%S'
                    )
                )
                one_variable.drop(one_variable.columns[[0, 1]], axis=1, inplace=True)
            else:
                one_variable = pd.DataFrame(index=wind_farm_wind_speed.index,
                                            columns=[col_name],
                                            dtype=float)
            # try:
            #     one_variable = pd.read_csv(Path(dir_name) / ''.join((wind_farm_name, file_postfix)))
            # except UnicodeDecodeError:
            #     one_variable = pd.read_csv(Path(dir_name) / ''.join((wind_farm_name, file_postfix)), engine='python')

            return one_variable


        wind_farm_wind_speed = one_variable_reading('_scada_ws.csv')
        wind_farm_power_output = one_variable_reading('_scada_pow.csv')
        print(f"{wind_farm_name} Pout {wind_farm_power_output.index[0]} to {wind_farm_power_output.index[-1]}")
        print(f"{wind_farm_name} WS {wind_farm_wind_speed.index[0]} to {wind_farm_wind_speed.index[-1]}")

        wind_farm_basic = pd.merge(wind_farm_wind_speed,
                                   wind_farm_power_output,
                                   how='outer',
                                   left_index=True,
                                   right_index=True)
        rename_mapper = {'WS': 'wind speed',
                         'POWER': 'active power output'}
        if not ws_pout_only:
            wind_farm_wd = one_variable_reading('_scada_wd.csv', 'WD')
            print(f"{wind_farm_name} WD {wind_farm_wd.index[0]} to {wind_farm_wd.index[-1]}")

            wind_farm_temp = one_variable_reading('_scada_temp.csv', 'TEMP')
            print(f"{wind_farm_name} TEMP {wind_farm_temp.index[0]} to {wind_farm_temp.index[-1]}")

            wind_farm_press = one_variable_reading('_scada_press.csv', 'PRESS')
            print(f"{wind_farm_name} PRES {wind_farm_press.index[0]} to {wind_farm_press.index[-1]}")

            wind_farm_basic = reduce(lambda a, b: pd.merge(a,
                                                           b,
                                                           how='outer',
                                                           left_index=True,
                                                           right_index=True),
                                     [wind_farm_basic, wind_farm_press, wind_farm_temp, wind_farm_wd])
            rename_mapper.update({
                'PRESS': 'pressure',
                'TEMP': 'temperature',
                'WD': 'wind direction',
            })
        wind_farm_basic.rename(columns=rename_mapper,
                               errors='raise',
                               inplace=True)

        print("*" * 50)

        wind_farm[wind_farm_name] = WF(
            wind_farm_basic,
            obj_name=f'{wind_farm_name} WF',
            rated_active_power_output=wf_rated_power_mapper[wind_farm_name],
            predictor_names=('wind speed',),
            dependant_names=('active power output',)
        )

        this_wind_farm = wind_farm[wind_farm_name]
        # %% Zelengrad strange part analysis mask
        this_wind_farm = this_wind_farm[np.bitwise_and(
            this_wind_farm.index >= datetime.datetime(year=2014, month=1, day=1),
            this_wind_farm.index < datetime.datetime(year=2015, month=1, day=1)
        )]
        # %% Data availability plot
        # availability_ax = data_availability_plot(this_wind_farm,
        #                                          name_mapper={'wind speed': 'WS',
        #                                                       'active power output': 'P' + r'$\mathrm{_{out}}$',
        #                                                       'pressure': 'PRESS',
        #                                                       'temperature': 'TEMP',
        #                                                       'wind direction': 'WD'}, )
        # availability_ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))

        # %% WD rose plot
        # if wind_farm_name in ('Bruska', 'Zelengrad'):
        #     ws = this_wind_farm['wind speed'].values
        #     wd = this_wind_farm['wind direction'].values
            # wind_rose_plot(ws, wd)
            # n = 8
            # wd_range_mask = [
            #     np.bitwise_and(wd >= i * 90 / n, wd < (i + 1) * 90 / n) for i in range(n)
            # ]
            # for range_i, this_mask in enumerate(wd_range_mask):
            #     scatter(ws[this_mask],
            #             this_wind_farm['active power output'].values[this_mask] / wf_rated_power_mapper[wind_farm_name],
            #             color='royalblue',
            #             x_label='Wind Speed [m/s]',
            #             x_lim=(-0.5, 33.5),
            #             y_lim=(-0.05, 1.05),
            #             y_label='Power Output [p.u.]',
            #             save_file_=wind_farm_name + f"_{range_i}", save_format='svg')

        # %% Pout-WS scatter plot
        # scatter(this_wind_farm.iloc[:, 0].values,
        #         this_wind_farm.iloc[:, 1].values,
        #         color='royalblue',
        #         x_label='Wind Speed [m/s]',
        #         x_lim=(-0.5, 33.5),
        #         y_lim=(-0.05 * this_wind_farm.rated_active_power_output, 1.05 * this_wind_farm.rated_active_power_output),
        #         y_label='Power Output [MW]',
        #         save_file_=wind_farm_name, save_format='svg')
        return wind_farm

# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# return_val = wind_farm[wind_farm_name]
# return_val_df = return_val.pd_view()


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

    pass
    # tt = load_croatia_data(this_wind_farm_name='Benkovac', ws_pout_only=False)
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
