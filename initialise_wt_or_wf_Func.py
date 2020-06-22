from WF_Class import WF
import pandas as pd
from project_path_Var import project_path_
import numpy as np
from numpy import ndarray
import datetime
import re
from File_Management.path_and_file_management_Func import list_all_specific_format_files_in_a_path
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save
from WT_Class import WT
from scipy.io import loadmat
import os
from pathlib import Path
from Ploting.fast_plot_Func import *


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
    ts = pd.read_csv(project_path_ + 'Data/Raw_measurements/Darly/Met_Mast.txt', sep='\t')['PCTimeStamp'].values
    return convert_datetime_str_in_txt_to_datetime64(ts)


def load_temperature_data():
    files = list_all_specific_format_files_in_a_path(project_path_ + 'Data/Raw_measurements/Darly/', 'csv', '')
    pattern = re.compile(r'^.*\d+Weather')
    temperature_data = pd.DataFrame({'time': [], 'environmental temperature': []})
    for file in files:
        regex = pattern.match(file)
        if regex is not None:
            one_reading = pd.read_csv(file, sep=',', skiprows=list(range(280)))
            one_reading_pd = pd.DataFrame({'time': convert_datetime_str_in_txt_to_datetime64(
                one_reading['ob_time'].values[:-1], ''),
                'environmental temperature': one_reading['air_temperature'].values[:-1]})
            temperature_data = pd.concat([temperature_data, one_reading_pd])
    return temperature_data


def load_raw_wt_from_txt_file_and_temperature_from_csv():
    wind_turbines = []
    files = list_all_specific_format_files_in_a_path(project_path_ + 'Data/Raw_measurements/Darly/', 'txt', order='')
    pattern = re.compile(r'^.*Turbine(\d+)\.txt$')
    temperature_data = load_temperature_data()
    for file in files:
        regex = pattern.match(file)
        if regex is not None:
            name = 'Darly WT' + regex.group(1)

            @load_exist_pkl_file_otherwise_run_and_save(project_path_ + 'Data/Raw_measurements/Darly/' +
                                                        name + " measurements.pkl")
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
                measurements = pd.merge(measurements, temperature_data, on='time', how='left')
                # 因为第一天的零点没有值，所以人工加上
                measurements.loc[0, 'environmental temperature'] = float(
                    measurements['environmental temperature'].values[144])
                measurements['environmental temperature'] = measurements['environmental temperature'].interpolate()
                if float(np.sum(np.diff(measurements['time'].values.astype('float')) <
                                np.mean(np.diff(measurements['time'].values.astype('float'))) / 2)) > 0:
                    raise Exception('Duplicated time stamp found')
                return measurements

            wind_turbines.append(WT(name=name, measurements=get_measurements))
    return tuple(wind_turbines)


def create_dalry_wind_farm_obj_using_wf_filling_missing_old():
    @load_exist_pkl_file_otherwise_run_and_save(os.path.join(project_path_,
                                                             'Data/Results/filling_missing/WF_filling_missing_old.pkl'))
    def get_measurements():
        ws_p = loadmat(os.path.join(project_path_, 'Data/Results/filling_missing/WF_filling_missing_old.mat'))
        ws_ = ws_p['corrected_wind_speed'].flatten()
        p_ = ws_p['corrected_power_output'].flatten()
        temp_wt = load_raw_wt_from_txt_file_and_temperature_from_csv()[0]
        time_ = temp_wt.measurements['time'].values[:ws_.size]
        temperature_ = temp_wt.measurements['environmental temperature'].values[:ws_.size]
        return time_, ws_, p_, temperature_

    time, ws, p, temperature = get_measurements
    dalry_wf = WF(name='Dalry',
                  measurements=pd.DataFrame(
                      {
                          'time': time,
                          'wind speed': ws,
                          'active power output': p * 3000 * 6,
                          'environmental temperature': temperature
                      }),
                  outlier_category=np.full(ws.shape, 0),
                  outlier_category_detailed=pd.DataFrame(
                      {
                          'time': np.full(ws.shape, 0),
                          'wind speed': np.full(ws.shape, 0),
                          'active power output': np.full(ws.shape, 0),
                          'environmental temperature': np.full(ws.shape, 0)
                      })
                  )
    WF.rated_active_power_output = 3000 * 6
    return dalry_wf


def load_raw_36_wts_in_nez():
    wts = loadmat(Path(project_path_) / 'Data/Raw_measurements/NEZ/WT_Classification.mat')
    return wts


def temp_func():
    # 画图，energies review paper
    # 垃圾WT和WF类，必须重写
    wt_15 = load_raw_36_wts_in_nez()['WT15']
    scatter(wt_15[:, 1], wt_15[:, 2], color='b', alpha=0.5, s=1,
            x_label='Wind speed (m/s)',
            y_label='Power output (p.u.)',
            x_lim=(-.05, 29.6),
            y_lim=(-0.01, 1.06))
    wf = loadmat(Path(project_path_) / 'Data/Raw_measurements/NEZ/WF_Classification.mat').get('WF_Classfication')
    scatter(wf[:, 1], wf[:, 2], color='b', alpha=0.5, s=1,
            x_label='Wind speed (m/s)',
            y_label='Power output (p.u.)',
            x_lim=(-.05, 29.6),
            y_lim=(-0.01, 1.06))
