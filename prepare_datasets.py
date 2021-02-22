from WT_WF_Class import WF, WT
import pandas as pd
from project_utils import *

from scipy.io import loadmat

from Ploting.fast_plot_Func import *
from TimeSeries_Class import TimeSeries
import re
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from PowerCurve_Class import PowerCurveFittedBy8PLF
from typing import Tuple
import getpass
from functools import reduce
from Ploting.data_availability_plot_Func import data_availability_plot
from Ploting.wind_rose_plot_Func import wind_rose_plot
import copy
from Writting import *
from Writting.utils import put_picture_into_a_docx
from collections import OrderedDict
from pandasql import sqldf
from Data_Preprocessing.TruncatedOrCircularToLinear_Class import CircularToLinear


def pysqldf(q):
    return sqldf(q, globals())


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
        #         y_lim=(-0.05 * this_wind_farm.rated_active_power_output,
        #               1.05 * this_wind_farm.rated_active_power_output),
        #         y_label='Power Output [MW]',
        #         save_file_=wind_farm_name, save_format='svg')
        return wind_farm


# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# return_val = wind_farm[wind_farm_name]
# return_val_df = return_val.pd_view()


def load_dalry_wind_farm_toy() -> WF:
    data_folder = project_path_ / Path(r"Data\Results\Forecasting\DalryWFToy\\")

    @load_exist_pkl_file_otherwise_run_and_save(data_folder / Path(r"Dalry Hourly.pkl"))
    def func():
        csv_data = pd.read_csv(data_folder / Path("Dalry Hourly.csv"))
        csv_data.set_index("time",  inplace=True)
        csv_data.index = pd.to_datetime(csv_data.index)
        csv_data["wind direction"] = CircularToLinear(360).inverse_transform(
            *csv_data[["wind direction sin", "wind direction cos"]].values.T)

        return csv_data

    data = func()
    wf_obj = WF(rated_active_power_output=18_000, number_of_wind_turbine=6,
                obj_name="dalry wind farm toy",
                predictor_names=("wind speed", "air density", "wind direction"),
                dependant_names=("active power output",),
                data=data)  # type: WF
    return wf_obj


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


def load_dalry_wind_farm_met_mast():
    @load_exist_pkl_file_otherwise_run_and_save(project_path_ / f'Data/Raw_measurements/Darly/Met_Mast.pkl')
    def get_measurements():
        one_reading = pd.read_csv(project_path_ / f'Data/Raw_measurements/Darly/Met_Mast.txt', sep='\t')
        one_reading.rename(mapper={"PCTimeStamp": "datetime",
                                   "VMET_Avg. Wind speed 1 (1)": "wind speed_1",
                                   "VMET_Avg. Wind speed 2 (2)": "wind speed_2",
                                   "VMET_Avg. Wind dir. 1 (3)": "wind direction"}, axis=1, inplace=True)
        index = pd.to_datetime(one_reading.iloc[:, 0])
        one_reading.index = index
        one_reading.drop(labels='datetime', axis=1, inplace=True)
        mask = np.bitwise_or(one_reading.loc[:, "wind direction"] < 0, one_reading.loc[:, "wind direction"] > 360)
        one_reading.loc[mask, "wind direction"] = np.nan
        return one_reading

    return get_measurements()


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


def load_high_resol_for_averaging_effects_analysis(load_all_wind_turbines: bool = False):
    folder_path = Path(project_path_) / 'Data/Raw_measurements/high_resol/'
    # high_resol = OrderedDict()
    high_resol = dict()
    if load_all_wind_turbines:
        if try_to_find_file(folder_path / 'high_resol_all.pkl'):
            return load_pkl_file(folder_path / 'high_resol_all.pkl')  # type:pd.DataFrame
    for this_file in folder_path.iterdir():
        if not load_all_wind_turbines:
            # For compatibility of old signature
            if this_file.match(r'*.mat'):
                file_name = re.match(r"\b.*(?=.mat)", this_file.name)[0]
                wts = loadmat(str(this_file))
                data = wts[file_name][:, [10, 11]]
                data[:, -1] *= 3000
                high_resol.setdefault(file_name, data)
        else:
            if re.match(r'^.*(xlsx|xls)$', this_file.__str__(), re.I):
                file_name = this_file.stem
                if this_file.suffix == ".xls":
                    data = pd.read_excel(this_file)
                else:
                    data = pd.read_excel(this_file, engine='openpyxl')
                # Rename columns
                old_column_names = data.columns
                assert len(old_column_names) == len(set(old_column_names))
                new_column_names = []
                for i, this_old_column_name in enumerate(old_column_names):
                    if "WTG" in this_old_column_name:
                        if 'kW' in this_old_column_name:
                            this_new_column_name = ("kW", new_column_names[-1][1])
                        elif ('Kvar' in this_old_column_name) or ('kVAR' in this_old_column_name):
                            this_new_column_name = ("kVAR", new_column_names[-2][1])
                        else:
                            this_new_column_name = ("WT_WS", str(int(re.findall(r"\d+", this_old_column_name)[0])))
                    elif "kW" in this_old_column_name:
                        this_new_column_name = ("kW", new_column_names[-1][1])
                    elif ('Kvar' in this_old_column_name) or ('kVAR' in this_old_column_name):
                        this_new_column_name = ("kVAR", new_column_names[-2][1])
                    else:
                        raise KeyError
                    new_column_names.append(this_new_column_name)
                assert len(new_column_names) == len(set(new_column_names)) == len(old_column_names) == len(
                    set(old_column_names))
                data.columns = pd.MultiIndex.from_tuples(new_column_names, names=('dimension', 'WT_no'))
                data.index = pd.MultiIndex.from_product([[file_name], data.index], names=('file name', 'id'))
                high_resol.setdefault(file_name, data)
    # Merge the data sets (only if load all wind turbines)
    if load_all_wind_turbines:
        wind_farm_basic = pd.concat(high_resol.values())
        save_pkl_file(folder_path / 'high_resol_all.pkl', wind_farm_basic)
        return wind_farm_basic  # type:pd.DataFrame
    return high_resol  # type:list


def check_possible_hysteresis_rules():
    """
    备注：cut out和restart可在以下找到：WT6['110406 Aik Data']
    :return:
    """
    resol = 1
    high_resol = load_high_resol_for_averaging_effects_analysis(load_all_wind_turbines=True)
    hi_wts = {}
    visions = ("Cut in", "Cut in back", "Cut out", "Restart")
    rules = [f"Past {x} seconds {y}" for x in (600, 60, 3) for y in ("average", "max", "min")] + \
            [f"Past {i} sec" for i in range(1, 2)]

    # %% Prepare individual data source
    high_resol.columns.__getattribute__('get_loc_level')('1', level=1)
    file_names = list(set(high_resol.index.get_level_values(0)))
    file_names.sort()
    wt_nos = set(high_resol.columns.get_level_values(1))
    wt_keys = [f"WT{wt_no}" for wt_no in wt_nos]
    wt_keys.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))
    for file_name in file_names:
        for wt_no in wt_nos:
            data = high_resol.iloc[
                high_resol.index.__getattribute__('get_locs')([file_name, slice(None)]),
                high_resol.columns.__getattribute__('get_locs')([slice(None), wt_no])
            ]
            data.index = data.index.droplevel(0)
            data.columns = data.columns.droplevel(1)
            data = data.drop('kVAR', axis=1)
            data = data.rename(columns={
                'WT_WS': 'wind speed',
                'kW': 'active power output'
            })
            wt_obj = WT(obj_name=f"WT{wt_no} in {file_name}",
                        predictor_names=('wind speed',),
                        dependant_names=('active power output',),
                        data=data)
            if np.all(np.isnan(wt_obj.values)):
                wt_obj = None
            else:
                wt_obj.loc[:, 'active power output'] /= wt_obj.rated_active_power_output

            if f"WT{wt_no}" not in hi_wts:
                hi_wts.setdefault(f"WT{wt_no}", {file_name: wt_obj})
            else:
                hi_wts[f"WT{wt_no}"].update({file_name: wt_obj})

    # %% Rules to check
    def checks_portfolio(_wt_obj: WT, _vision: str, _rule: str) -> list:
        assert _vision in visions
        result = []

        # Check indices for interest event happen
        interest_event_index = []
        pout = _wt_obj['active power output'].values
        ws = _wt_obj['wind speed'].values
        for i in range(1, _wt_obj.shape[0]):
            if np.all(pout > 0) or np.all(pout <= 0):
                break
            if _vision == "Cut out":
                if all((pout[i] <= 0, pout[i - 1] > 0, ws[i] > 10)):
                    interest_event_index.append(_wt_obj.index[i])
            elif _vision == "Restart":
                if all((pout[i] > 0, pout[i - 1] <= 0, ws[i] > 10)):
                    interest_event_index.append(_wt_obj.index[i])
            elif _vision == "Cut in":
                if all((pout[i] > 0, pout[i - 1] <= 0, ws[i] < 10)):
                    interest_event_index.append(_wt_obj.index[i])
            else:  # Cut in back
                if all((pout[i] <= 0, pout[i - 1] > 0, ws[i] < 10)):
                    interest_event_index.append(_wt_obj.index[i])

        # Check all rules
        now_sec = int(re.match(r"^.*\s(\d+)\s\w*(\s?)(.*)$", _rule).group(1))
        now_case = re.match(r"^.*\s(\d+)\s\w*(\s?)(.*)$", _rule).group(3)
        if now_sec == 1:
            for index in interest_event_index:
                result.append(_wt_obj.loc[[index - 1]]['wind speed'].mean())
        else:
            for index in interest_event_index:
                if now_case == "average":
                    result.append(_wt_obj.loc[index - 1 - now_sec // resol:index - 1]['wind speed'].mean())
                elif now_case == "max":
                    result.append(_wt_obj.loc[index - 1 - now_sec // resol:index - 1]['wind speed'].max())
                else:
                    result.append(_wt_obj.loc[index - 1 - now_sec // resol:index - 1]['wind speed'].min())

        return result

    # Iterate over the individual data source
    results = {key: {key_2: [] for key_2 in rules} for key in visions}
    for wt_key in wt_keys:
        for file_name in file_names:
            wt_obj = hi_wts[wt_key][file_name]  # type:Union[WT, None]
            if wt_obj is None:
                continue

            for vision in visions:
                for rule in rules:
                    results[vision][rule].extend(checks_portfolio(wt_obj, vision, rule))

    # Collect the plot buffer
    results_plot = OrderedDict()
    for vision in visions:
        results_plot[vision] = OrderedDict()
        for rule in rules:
            plot_data = np.array(results[vision][rule])
            plot_data = np.around(plot_data, 1)
            rule_extend = (rule + "\n" + f"Average = {np.mean(plot_data):.3f} m/s" +
                           "\n" + f"Min = {np.min(plot_data):.3f} m/s" +
                           "\n" + f"Max = {np.max(plot_data):.3f} m/s")
            ax = hist(plot_data, bins=np.arange(-0.05, 35, 0.1), density=True,
                      edgecolor='royalblue', color='royalblue',
                      x_label=WS_POUT_2D_PLOT_KWARGS['x_label'], y_label='Probability Density')
            y_lim = ax.get_ylim()
            ax = vlines(np.mean(plot_data), color='r', ax=ax, y_lim=y_lim, save_to_buffer=True, label="Mean")
            results_plot[vision].update(OrderedDict([(f"{rule_extend}", (ax, 7.5))]))
    # Write and save
    put_picture_into_a_docx(results_plot,
                            project_path_ / r"Data\Results\transient_study\TSE2020\possible_hysteresis_rules.docx")


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

    # hi = load_high_resol_for_averaging_effects_analysis(load_all_wind_turbines=True)
    # check_possible_hysteresis_rules()
    # load_dalry_wind_farm_toy()
    # %% SQL practice
    # df_1 = load_weather_data()
    # sql_str = """
    # SELECT `time`, `environmental temperature`
    # FROM df_1
    # WHERE (`time` BETWEEN date('2007-01-01') AND date('2008-01-01')) AND (
    # `environmental temperature` < 0)
    # """
    # WHERE `time` BETWEEN strftime('%Y-%m-%d %H:%M:%S','2007-01-01', 'start of day', '+14 hours')
    # AND date('2008-01-01')
    # WHERE `time` BETWEEN datetime('2007-01-01', 'start of day', '+13 hours') AND date('2008-01-01')
    # WHERE `time` BETWEEN strftime('%Y-%m-%d %H:%M:%S', '2007-01-01 12:00:00') AND date('2008-01-01')

    # WHERE `time` BETWEEN date('2007-01-01') AND date('2008-01-01')

    # tt2 = sqldf(sql_str)
