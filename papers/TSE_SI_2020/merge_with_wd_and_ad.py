from project_utils import *
from Ploting.fast_plot_Func import *
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
import datetime
from WT_WF_Class import WF
from PhysicalInstance_Class import *
import pandas as pd
from operating_regime_data_prepare import _print_basic_info
from Data_Preprocessing.TruncatedOrCircularToLinear_Class import CircularToLinear
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, WF_TO_CLUSTER_MAPPER

MEASUREMENTS_DIR = project_path_ / Path(r'./Data/Raw_measurements/TSE_SI_2020_WF')


def agg_wd_and_merge(wf_obj):
    file_name = re.search(r"cluster\s(.*)\sWF", wf_obj.obj_name).group(1)
    wd_df = pd.read_csv(MEASUREMENTS_DIR / fr"./WD/{file_name}.csv")
    wd_df.index = pd.to_datetime(wd_df['TIMESTAMP'])
    wd_df.drop('TIMESTAMP', axis=1, inplace=True)
    wd_df.rename({'WD': 'wind direction'}, axis=1, inplace=True)
    # Outlier
    mask = ~np.bitwise_and(wd_df['wind direction'].values <= 360,
                           wd_df['wind direction'].values >= 0)
    # print(f"{file_name} WD outlier number = {np.sum(mask)}")
    # print(f"{file_name} WD missing value number = {np.sum(np.isnan(wd_df['wind direction'].values))}")
    wd_df[mask] = np.nan
    # transform
    c_to_l = CircularToLinear(360)
    temp = c_to_l.transform(wd_df['wind direction'].values)
    wd_df['wind direction cos'] = temp['cos']
    wd_df['wind direction sin'] = temp['sin']
    wd_df.drop('wind direction', axis=1, inplace=True)
    del temp
    # resample
    wd_df = wd_df.resample('60T').mean()
    wd_df['wind direction'] = c_to_l.inverse_transform(sin_val=wd_df['wind direction sin'].values,
                                                       cos_val=wd_df['wind direction cos'].values)
    # print(f"{file_name} WD missing value number after resample = {np.sum(np.isnan(wd_df['wind direction'].values))}")
    # get MERRA-II
    pass
    # merge
    wf_obj_with_wd = pd.merge(wf_obj, wd_df, right_index=True, left_index=True, how='left')
    # print("--------------Before merge with WD---------------------")
    # _print_basic_info(wf_obj)
    # print("--------------After merge with WD---------------------")
    # _print_basic_info(wf_obj_with_wd)
    # print("==============================================================================")
    # print("\n")
    return wf_obj_with_wd


def get_ad_and_merge(wf_obj):
    cluster_name = re.search(r"(.*)\scluster\s(.*)\sWF", wf_obj.obj_name).group(1)
    file_name = re.search(r"cluster\s(.*)\sWF", wf_obj.obj_name).group(1)

    ad_df = pd.DataFrame()
    for year in (2017, 2018, 2019):
        one_reading = pd.read_csv(MEASUREMENTS_DIR / fr"./MERRA/{cluster_name}/{file_name}_{year}.csv",
                                  skiprows=3)
        one_reading.index = pd.to_datetime(one_reading['local_time'])
        one_reading = one_reading[['air_density']]
        one_reading.rename({'air_density': 'air density'}, axis=1, inplace=True)
        ad_df = pd.concat([ad_df, one_reading])
    wf_obj_final = pd.merge(wf_obj, ad_df, right_index=True, left_index=True, how='left')

    return wf_obj_final


def _to_wf_obj(wf_reading, cluster_name, wind_farm_name):
    wf_obj = WF(
        data=wf_reading,
        obj_name=f'{cluster_name} cluster {wind_farm_name} WF',
        rated_active_power_output=WF_RATED_POUT_MAPPER[wind_farm_name],
        predictor_names=('wind speed',),
        dependant_names=('active power output',),
        number_of_wind_turbine=NUMBER_OF_WT_MAPPER[wind_farm_name],
    )
    wf_obj.index = pd.DatetimeIndex(
        pd.to_datetime(
            wf_obj.iloc[:, 0],
            # format='%Y-%m-%d %H:%M:%S'
        )
    )
    wf_obj.drop(wf_obj.columns[0], axis=1, inplace=True)

    return wf_obj


def merge_and_save():
    for wind_farm_name, cluster_name in WF_TO_CLUSTER_MAPPER.items():
        if wind_farm_name == 'Vostane_Kamensko':
            continue
        # elif wind_farm_name == 'Bruska':  # 812 hrs
        #     continue
        # elif wind_farm_name == 'Benkovac':  # 2105 hrs
        #     continue
        # elif wind_farm_name == 'Glunca':  # 738 hrs
        #     continue

        wf_reading = pd.read_csv(MEASUREMENTS_DIR / fr"./OPR/{cluster_name} cluster {wind_farm_name} WF_with_OPR.csv")
        wf_obj = _to_wf_obj(wf_reading, cluster_name, wind_farm_name)
        wf_obj_with_wd = agg_wd_and_merge(wf_obj)
        wf_obj_final = get_ad_and_merge(wf_obj_with_wd)
        # Remove duplicate
        wf_obj_final = wf_obj_final[~wf_obj_final.index.duplicated()]
        wf_obj_final = wf_obj_final.reindex(
            pd.period_range(start='2018-01-01', end='2020-01-01', freq='60T').to_timestamp()[:-1]
        )
        wf_obj_final[np.any(np.isnan(wf_obj_final.values), axis=1)].pd_view()
        print(f"=============================={wind_farm_name}=====================================")
        _print_basic_info(wf_obj_final)
        print("==============================================================================")
        wf_obj_final.pd_view().to_csv(f"{wf_obj.obj_name}_pre_FINAL.csv")


if __name__ == '__main__':
    merge_and_save()
