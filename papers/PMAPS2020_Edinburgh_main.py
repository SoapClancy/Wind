"""
PMAPS2020_Edinburgh_main.py用于处理PMAPS2020的Edinburgh的文章的数据。
__author__      = "Mingzhe Zou"
__copyright__   = "Copyright 2020, Mingzhe Zou"
__email__ = "mingzhe.zou@ed.ac.uk"
"""
import pandas as pd
from project_utils import project_path_
import numpy as np
from numpy import ndarray
import re
import datetime
from Ploting.fast_plot_Func import series, scatter
from BivariateAnalysis_Class import Bivariate
from Time_Processing.Season_Enum import SeasonTemplate1
import copy
from WT_WF_Class import WT
from MarkovChain.MarkovChain_Class import OneDimMarkovChain
from BivariateAnalysis_Class import MethodOfBins
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save, save_pkl_file, load_npy_file, \
    load_pkl_file
from File_Management.path_and_file_management_Func import try_to_find_folder_path_otherwise_make_one
from papers.rubbish.correlation_modeling_main import load_test_data_for_iet_paper
from Ploting.uncertainty_plot_Func import series_uncertainty_plot
from Ploting.fast_plot_Func import vlines
from PowerCurve_Class import PowerCurve
from ErrorEvaluation_Class import ProbabilisticErrorIETPaperMethod, EnergyBasedError, DeterministicError
from scipy.io import loadmat, savemat
from UnivariateAnalysis_Class import UnivariatePDFOrCDFLike, MixtureUnivariatePDFOrCDFLike, \
    DeterministicUnivariateProbabilisticModel
import csv


@load_exist_pkl_file_otherwise_run_and_save(project_path_ + 'Data/Raw_measurements/PMAPS2020_Edinburgh/' +
                                            "Merra-2 data.pkl")
def load_merra_2_data():
    """
    载入merra 2的数据。这个.csv来自Duo。把这个.csv转换成标准的pd.DataFrame对象
    """
    merra_2_data_file_ = ''.join((project_path_, 'Data/Raw_measurements/PMAPS2020_Edinburgh/Merra-2 data.csv'))
    one_reading = pd.read_csv(merra_2_data_file_, sep=',', skiprows=list(range(24)))

    # 将one_reading中的date和time的格式和类型标准化
    def convert_date_and_time_str_to_datetime64():
        date_time_stamp = []
        for i in range(one_reading.shape[0]):
            this_date_in_one_reading = one_reading['# Date'].iloc[i]
            this_time_in_one_reading = one_reading['UT time'].iloc[i]
            if re.findall(r':', this_time_in_one_reading).__len__() == 2:
                this_time_in_one_reading = this_time_in_one_reading[:-3]
            if this_time_in_one_reading[:2] == '24':
                this_time_in_one_reading = '0:00'
                if i != one_reading.shape[0] - 1:
                    this_date_in_one_reading = one_reading['# Date'].iloc[i + 1]
                else:
                    this_date_in_one_reading = '2013/1/1'
            date_time_stamp.append(datetime.datetime.strptime(this_date_in_one_reading + ' ' + this_time_in_one_reading,
                                                              '%Y/%m/%d %H:%M'))
        return date_time_stamp

    date_time = pd.DataFrame({'date time': convert_date_and_time_str_to_datetime64()})
    # 复制其它的col到reading_results中
    other_cols = one_reading.iloc[:, 2:]
    reading_results = pd.concat([date_time, other_cols], axis=1)
    return reading_results


def load_old_wind_farm_corrected():
    ws = loadmat(project_path_ + 'Data/Results/fill_missing_and_correct/old/WF corrected_WS.mat')[
        'corrected_wind_speed']
    pout = loadmat(project_path_ + 'Data/Results/fill_missing_and_correct/old/WF corrected_power_output.mat')[
               'corrected_power_output'] * 3000
    return pd.DataFrame({'Wind speed': ws.reshape(-1, ),
                         'Wind power': pout.reshape(-1, )})


def find_extreme_wind_speed_for_rated_power(rated_power=2940, bin_step=0.5):
    # measurements = load_merra_2_data
    measurements = load_old_wind_farm_corrected()

    # 建立Bivariate类，通过mob去寻找符合要求的wind speed bin：bin中的95%的power output都不低于rated power
    bivariate = Bivariate(measurements['Wind speed'].values, measurements.iloc[:, -1].values, bin_step=bin_step)
    _95_percentile_in_mob = bivariate.cal_mob_statistic_eg_quantile(np.array([0.95]))
    _95_percentile_in_mob_reach_rated_mask = _95_percentile_in_mob[:, -1] > rated_power

    min_wind_speed_bin_idx = bivariate.find_mob_key_according_to_mob_or_mob_fitting_like_dict(
        _95_percentile_in_mob[_95_percentile_in_mob_reach_rated_mask, 0][0],
        bivariate.mob
    )['accurate_bin_key']

    max_wind_speed_bin_idx = bivariate.find_mob_key_according_to_mob_or_mob_fitting_like_dict(
        _95_percentile_in_mob[_95_percentile_in_mob_reach_rated_mask, 0][-1],
        bivariate.mob
    )['accurate_bin_key']

    print('Minimum wind speed bin where 95% power output data reach rated: {} (format = [start medium end])'.format(
        bivariate.mob[min_wind_speed_bin_idx]['this_bin_boundary']
    ))
    print('Maximum wind speed bin where 95% power output data reach rated: {} (format = [start medium end])'.format(
        bivariate.mob[max_wind_speed_bin_idx]['this_bin_boundary']
    ))

    # 基于所有的原始数据，画风速-power的scatter
    # ax = scatter(measurements['Wind speed'].values, measurements.iloc[:, -1] / 3000)
    # ax = vlines(bivariate.mob[min_wind_speed_bin_idx]['this_bin_boundary'], ax=ax)
    # ax = vlines(bivariate.mob[max_wind_speed_bin_idx]['this_bin_boundary'], ax=ax)
    # hlines(rated_power, ax, label='Rated power = {} kW'.format(rated_power), color='r',
    #        x_label='Wind speed (m/s)', y_label='Power output (p.u.)',
    #        x_lim=(0, 28.5),
    #        y_lim=(-0.0125, 1.03))


def find_extreme_wind_speed_for_rated_power_sasa(rated_power=3000, bin_step=0.5):
    # histogram
    measurements = load_old_wind_farm_corrected()
    measurements.iloc[:, -1] /= 3000
    measurements.iloc[:, -1] /= 0.98

    # 画10-min的
    bivariate = Bivariate(measurements['Wind speed'].values, measurements.iloc[:, -1].values, bin_step=bin_step)
    key = bivariate.find_mob_key_according_to_mob_or_mob_fitting_like_dict(23,
                                                                           bivariate.mob
                                                                           )['accurate_bin_key']
    _5_percentile_in_mob = bivariate.cal_mob_statistic_eg_quantile(np.array([0.05]))
    _5_percentile_in_mob *= 0.99
    _95_percentile_in_mob = bivariate.cal_mob_statistic_eg_quantile(np.array([0.95]))
    # ax = vlines(_5_percentile_in_mob[key, 1], color='k', linestyles=':',
    #             label='5$^\mathrm{th}$ percentile')
    # ax = vlines(_95_percentile_in_mob[key, 1], color='r', linestyles='-',
    #             label='95$^\mathrm{th}$ percentile',
    #             ax=ax)
    # hist(bivariate.mob[key]['dependent_var_in_this_bin'],
    #      x_label='Power output (p.u.) in wind speed bin = [22.75, 23.25) (m/s)',
    #      y_label='Probability density',
    #      x_lim=(0, 1.03),
    #      y_lim=(0, 26),
    #      bins=np.arange(0, 1.025, 0.0125) + 0.0125 / 2,
    #      normed=True,
    #      ax=ax,
    #      label='WF data',
    #      legend_loc='upper left')
    tt = 1
    # resample
    # ws, pout = measurements['Wind speed'].values, measurements.iloc[:, -1].values
    # ws = ws.reshape(-1, 6)
    # pout = pout.reshape(-1, 6)
    # ws = np.nanmean(ws, axis=1)
    # pout = np.nanmean(pout, axis=1)
    # scatter(ws, pout)
    # bivariate = Bivariate(ws, pout, bin_step=bin_step)

    # 基于所有的原始数据，画风速-power的scatter
    ax = scatter(measurements['Wind speed'].values, measurements.iloc[:, -1],
                 x_lim=(0, 28.5),
                 y_lim=(-0.0125, 1.05),
                 c=(0.6, 0.6, 0.6),
                 # label='WF wind speed - power output data',
                 alpha=0.2,
                 figure_size=(5, 5 * 0.618 + 0.22)
                 )

    # 画5/95

    ax = series(bivariate.array_of_bin_boundary[:, 1], _5_percentile_in_mob[:, -1], linestyle=':', color='k',
                label='5$^\mathrm{th}$ percentile power curve', ax=ax, linewidth=2.5,
                )
    ax = vlines(bivariate.mob[28]['this_bin_boundary'][1], ax=ax, color='cyan', linewidth=2.5)
    ax = vlines(bivariate.mob[45]['this_bin_boundary'][1], ax=ax, color='g', linestyles='-.', linewidth=2.5)
    ax = series(bivariate.array_of_bin_boundary[:, 1], _95_percentile_in_mob[:, -1], ax=ax, color='r',
                x_label='Wind Speed (m/s)', y_label='Power Output (p.u.)',
                x_lim=(0, 28.5),
                y_lim=(-0.0125, 1.05),
                label='95$^\mathrm{th}$ percentile power curve',
                legend_loc='upper left',
                linewidth=2.5,
                )
    ax.legend(bbox_to_anchor=(-0.12, 1.02, 1.12, .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.)
    ax.annotate("14.5 m/s", xy=(14.5, -0.0125), xytext=(10.5, 0.2),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("23 m/s", xy=(23, -0.0125), xytext=(19, 0.2),
                arrowprops=dict(arrowstyle="->"))
    tt = 1


def train_3d_cvine_gmcm_using_first_2_years(full_this_wind_turbine: WT):
    """
    这是PMAPS Edinburgh的training
    """
    full_this_wind_turbine_pro = copy.deepcopy(full_this_wind_turbine)
    full_this_wind_turbine_pro.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                           datetime.datetime(2008, 11, 30, 23, 50))
    for this_season in SeasonTemplate1.__members__:
        specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine_pro)
        specific_this_wind_turbine.do_truncate_by_season(this_season)
        # %% re-calculate outliers
        specific_this_wind_turbine.identify_outlier()
        specific_this_wind_turbine.fit_3d_cvine_gmcm_model(use_ws_ahead=1)


def __encode_markov_chain(ws_series: ndarray, wd_series: ndarray, ws_resol=0.5, wd_resol=90):
    def one_encoder(data, resol, force_lower_boundary=None, force_upper_boundary=None):
        if (force_lower_boundary is not None) and (force_upper_boundary is not None):
            boundary_array = np.arange(force_lower_boundary, force_upper_boundary + resol, resol)
        else:
            boundary_array = np.arange(float(np.nanmin(data)), float(np.nanmax(data)) + resol, resol)
        # print("float(np.nanmin(data))={}, float(np.nanmax(data))={}".format(
        #     float(np.nanmin(data)), float(np.nanmax(data))
        # ))
        # print("boundary_array[0]={}, boundary_array[-1]={}".format(
        #     boundary_array[0], boundary_array[-1]
        # ))
        encoded_data = np.full(data.shape, np.nan)
        for i_ in range(boundary_array.size - 1):
            lower_boundary = boundary_array[i_]
            upper_boundary = boundary_array[i_ + 1]
            encoded_data[np.bitwise_and(data >= lower_boundary, data < upper_boundary)] = i_
        return encoded_data

    # 将来test set的所有out of 这个range的数据必须置为nan，否则可能编码错误！这里用的是统一固定的编码！
    ws_series_encoded = one_encoder(ws_series, ws_resol, 3, 18)
    wd_series_encoded = one_encoder(wd_series, wd_resol, 0, 360)
    # 检测特殊状态
    ws_series_encoded_state_num = np.unique(ws_series_encoded[~np.isnan(ws_series_encoded)]).size
    wd_series_encoded_state_num = np.unique(wd_series_encoded[~np.isnan(wd_series_encoded)]).size
    two_dim_encoded_state_num = ws_series_encoded_state_num * wd_series_encoded_state_num
    # 降维编码，将wd_series_encoded_state_num（即：wd_series_encoded的状态的个数）作为偏移进制
    two_dim_encoded = np.full(ws_series.shape, np.nan)
    for i in range(two_dim_encoded.size):
        offset = ws_series_encoded[i] * (int((360 - 0) / 90))  # force有四个状态！！！
        value = wd_series_encoded[i]
        two_dim_encoded[i] = offset + value
    return two_dim_encoded


def train_two_dim_mc_using_first_2_years(full_this_wind_turbine: WT):
    full_this_wind_turbine_pro = copy.deepcopy(full_this_wind_turbine)
    full_this_wind_turbine_pro.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                           datetime.datetime(2008, 11, 30, 23, 50))
    mc_matrix, mob = {}, {}
    for this_season in SeasonTemplate1.__members__:
        specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine_pro)
        specific_this_wind_turbine.do_truncate_by_season(this_season)
        ws_wd_time_series = np.array([specific_this_wind_turbine.measurements['wind speed'].values,
                                      specific_this_wind_turbine.measurements['absolute wind direction'].values,
                                      specific_this_wind_turbine.measurements['active power output'].values]).T
        all_not_outlier = np.all((specific_this_wind_turbine.outlier_category_detailed['wind speed'].values == 0,
                                  specific_this_wind_turbine.outlier_category_detailed[
                                      'absolute wind direction'].values == 0,
                                  specific_this_wind_turbine.outlier_category_detailed[
                                      'active power output'].values == 0), axis=0)

        # 将有outlier的数据行置为nan
        ws_wd_time_series[~all_not_outlier, :] = np.nan
        # 生成mc矩阵
        two_dim_encoded = __encode_markov_chain(ws_wd_time_series[:, 0], ws_wd_time_series[:, 1])

        path_ = ''.join((specific_this_wind_turbine.results_path,
                         'two_dim_mc/',
                         specific_this_wind_turbine.__str__(),
                         '/'))
        try_to_find_folder_path_otherwise_make_one(path_)

        # mc_matrix
        @load_exist_pkl_file_otherwise_run_and_save(path_ + 'mc_matrix.pkl')
        def cal_two_dim_mc_matrix():
            mc_matrix_ = OneDimMarkovChain(current_state=two_dim_encoded[:-1],
                                           next_state=two_dim_encoded[1:])
            return mc_matrix_.state_markov_chain_in_matrix

        mc_matrix.setdefault(this_season, cal_two_dim_mc_matrix)

        # unique_state
        @load_exist_pkl_file_otherwise_run_and_save(path_ + 'unique_state.pkl')
        def cal_unique_state():
            return np.unique(two_dim_encoded[:-1][~np.isnan(two_dim_encoded[:-1])])

        mc_matrix.setdefault(this_season, cal_unique_state)

        # 找到编码2维变量和同一时刻功率的关系：
        mob_ = MethodOfBins(two_dim_encoded, ws_wd_time_series[:, 2], first_bin_left_boundary=-0.5, bin_step=1)
        save_pkl_file(path_ + 'mob.pkl', mob_)
        mob.setdefault(this_season, mob_)

    return mc_matrix, mob


def estimate_active_power_output_by_two_dim_mc_model(specific_this_wind_turbine, ws_wd_time_series):
    path_ = ''.join((specific_this_wind_turbine.results_path,
                     'two_dim_mc/',
                     specific_this_wind_turbine.__str__(),
                     '/'))

    two_dim_encoded = __encode_markov_chain(ws_wd_time_series[:, 0], ws_wd_time_series[:, 1])

    mc_matrix = load_pkl_file(path_ + 'mc_matrix.pkl')
    mc_obj = OneDimMarkovChain(state_markov_chain_in_matrix=mc_matrix)
    mc_obj.unique_state = load_pkl_file(path_ + 'unique_state.pkl')
    mob = load_pkl_file(path_ + 'mob.pkl')  # type:MethodOfBins

    pout = []
    for i, this_two_dim_encoded in enumerate(two_dim_encoded):
        if np.isnan(this_two_dim_encoded):
            if (ws_wd_time_series[i, 0] > 25) or (ws_wd_time_series[i, 0] < 3):
                pout.append(DeterministicUnivariateProbabilisticModel(0))
            else:
                pout.append(DeterministicUnivariateProbabilisticModel(3000))
            continue

        # 选取每一个可能的下一个状态
        next_state_pdf = mc_obj.get_next_state_pdf_from_current_state(this_two_dim_encoded).flatten()

        pout_this = []
        for this_next_possible_two_dim_encoded in range(next_state_pdf.size):
            key = mob.find_mob_key_according_to_mob_or_mob_fitting_like_dict(
                this_next_possible_two_dim_encoded,
                mob.mob)['nearest_not_none_bin_keys']
            pout_this.append(mob.out_put_mob_row_as_univariate_cdf_like(key, cdf_x=np.linspace(0, 3000, 100)))

        pout.append(MixtureUnivariatePDFOrCDFLike.do_mixture(
            univariate_pdf_or_cdf_like=np.array(pout_this),
            weights=next_state_pdf
        ))
    return pout


def estimate_active_power_output_by_empirical_model(specific_this_wind_turbine, ws_wd_time_series):
    path_ = ''.join((specific_this_wind_turbine.results_path,
                     'two_dim_mc/',
                     specific_this_wind_turbine.__str__(),
                     '/'))
    two_dim_encoded = __encode_markov_chain(ws_wd_time_series[:, 0], ws_wd_time_series[:, 1])

    mob = load_pkl_file(path_ + 'mob.pkl')  # type:MethodOfBins

    pout = []
    for i, this_two_dim_encoded in enumerate(two_dim_encoded):
        if np.isnan(this_two_dim_encoded):
            if (ws_wd_time_series[i, 0] > 25) or (ws_wd_time_series[i, 0] < 3):
                pout.append(DeterministicUnivariateProbabilisticModel(0))
            else:
                pout.append(DeterministicUnivariateProbabilisticModel(3000))
            continue
        else:
            key = mob.find_mob_key_according_to_mob_or_mob_fitting_like_dict(
                this_two_dim_encoded, mob.mob)['nearest_not_none_bin_keys']
            pout.append(mob.out_put_mob_row_as_univariate_cdf_like(key, cdf_x=np.linspace(0, 3000, 100)))
    return pout


def test_on_year3_and_by_seasons(full_this_wind_turbine: WT, doing_day_analysis: bool = False):
    """
    这是PMAPS paper的test
    """
    full_this_wind_turbine_pro = copy.deepcopy(full_this_wind_turbine)
    # %% 只选择用前两年数据生成的模型
    full_this_wind_turbine_pro.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                           datetime.datetime(2008, 11, 30, 23, 50))
    # 全年的结果
    whole_year_test_buff = {}.fromkeys(('mean', '5', '95'))
    for key in whole_year_test_buff.keys():
        whole_year_test_buff[key] = []

    whole_year_model_buff = {}.fromkeys(('mfr', '2d_gmcm', 'cvine_gmcm'))
    for key in whole_year_model_buff.keys():
        whole_year_model_buff[key] = {'mean': [], '5': [], '95': []}

    for this_season_idx, this_season in enumerate(SeasonTemplate1.__members__):
        specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine_pro)
        specific_this_wind_turbine.do_truncate_by_season(this_season)
        # %% 提取测试用的数据。给IET paper
        test_data = load_test_data_for_iet_paper(this_season_idx)
        # shift，因为这是forecast
        test_data['test_pout_actual_5'] = np.roll(test_data['test_pout_actual_5'], -1)
        test_data['test_pout_actual_5'][-1] = np.nan
        test_data['test_pout_actual_95'] = np.roll(test_data['test_pout_actual_95'], -1)
        test_data['test_pout_actual_95'][-1] = np.nan
        test_data['test_pout_actual'] = np.roll(test_data['test_pout_actual'], -1)
        test_data['test_pout_actual'][-1] = np.nan

        # %% 提取boundary的数据，因为只考虑region_a
        boundary_path_ = ''.join((specific_this_wind_turbine.results_path,
                                  '3d_cvine_gmcm_model_use_ws_ahead_1/' + specific_this_wind_turbine.__str__() + '/'))
        model_boundary = load_npy_file(boundary_path_ + 'model_boundary.npy')
        (region_1_mask, region_a_mask_in_input_data, region_rated_mask,
         region_b_mask_in_input_data, region_5_mask, hard_rated_mask, hard_cut_off_mask) = \
            PowerCurve.cal_region_boundary_mask(model_boundary,
                                                test_data.get('test_ws_actual'),
                                                test_data.get('test_outlier_category') == 0)

        # Considering_one_day_mask
        time_mask_pure = np.full(region_1_mask.shape, False)
        if this_season == 'winter':
            day_start_idx, day_end_idx = 7862, 8007
        elif this_season == 'spring':
            day_start_idx, day_end_idx = 7043, 7187
        elif this_season == 'summer':
            day_start_idx, day_end_idx = 3993, 4137
        elif this_season == 'autumn':
            day_start_idx, day_end_idx = 3849, 3993
        else:
            raise
        time_mask_pure[day_start_idx:day_end_idx] = True
        if not doing_day_analysis:
            time_mask_pure[:] = True

        region_a_mask_and_time_mask = np.bitwise_and(region_a_mask_in_input_data,
                                                     time_mask_pure)
        time_mask_and_cat_0_mask = np.bitwise_and(time_mask_pure,
                                                  test_data.get('test_outlier_category') == 0)

        def cal_using_what_model(what_model_name):
            pout_by_what_model_mean = np.full(test_data.get('test_ws_actual').size, np.nan)
            pout_by_what_model_5 = np.full(test_data.get('test_ws_actual').size, np.nan)
            pout_by_what_model_95 = np.full(test_data.get('test_ws_actual').size, np.nan)
            if what_model_name == 'two_dim_mc':
                pout_by_what_model = estimate_active_power_output_by_two_dim_mc_model(
                    specific_this_wind_turbine,
                    np.stack((test_data.get('test_ws_actual'), test_data.get('test_wd_actual')),
                             axis=1)[region_a_mask_and_time_mask, :])
            elif what_model_name == 'cvine_gmcm':
                pout_by_what_model = specific_this_wind_turbine.estimate_active_power_output_by_3d_cvine_gmcm_model(
                    np.stack((test_data.get('test_ws_actual'), test_data.get('test_wd_actual')),
                             axis=1)[region_a_mask_and_time_mask, :],
                    use_ws_ahead=1)
            elif what_model_name == 'empirical':
                pout_by_what_model = estimate_active_power_output_by_empirical_model(
                    specific_this_wind_turbine,
                    np.stack((test_data.get('test_ws_actual'), test_data.get('test_wd_actual')),
                             axis=1)[region_a_mask_and_time_mask, :])
            else:
                raise Exception("Check model name")

            pout_by_what_model_mean[region_a_mask_and_time_mask] = np.array([x.mean_ for x in pout_by_what_model])
            # 删除pdf信息，只要cdf就可以计算icdf。这样可以节约一半内存
            for i in range(pout_by_what_model_mean.__len__()):
                if isinstance(pout_by_what_model_mean[i], UnivariatePDFOrCDFLike):
                    pout_by_what_model_mean[i].pdf_like_ndarray = None
            # 计算5-95percentiles
            pout_by_what_model_5_95 = np.array(
                [x.find_nearest_inverse_cdf(np.array([0.05, 0.95])) for x in pout_by_what_model])
            pout_by_what_model_5[region_a_mask_and_time_mask] = pout_by_what_model_5_95[:, 0]
            pout_by_what_model_95[region_a_mask_and_time_mask] = pout_by_what_model_5_95[:, 1]
            return {'pout_by_what_model_mean': pout_by_what_model_mean,
                    'pout_by_what_model_5': pout_by_what_model_5,
                    'pout_by_what_model_95': pout_by_what_model_95}

        # %% c-vine GMCM
        cvine_gmcm_path = ''.join((specific_this_wind_turbine.results_path,
                                   '3d_cvine_gmcm_model_use_ws_ahead_1/',
                                   specific_this_wind_turbine.__str__(),
                                   '/PMAPS_paper/'))
        try_to_find_folder_path_otherwise_make_one(cvine_gmcm_path)

        @load_exist_pkl_file_otherwise_run_and_save(cvine_gmcm_path + 'cvine_gmcm_results_{}.pkl'.format(
            doing_day_analysis
        ))
        def cal_using_cvine_gmcm_wrapper():
            return cal_using_what_model('cvine_gmcm')

        cvine_gmcm_results = cal_using_cvine_gmcm_wrapper

        # %% two_dim_mc
        two_dim_mc_path = ''.join((specific_this_wind_turbine.results_path,
                                   'two_dim_mc/',
                                   specific_this_wind_turbine.__str__(), '/'))
        try_to_find_folder_path_otherwise_make_one(two_dim_mc_path)

        @load_exist_pkl_file_otherwise_run_and_save(two_dim_mc_path + 'two_dim_mc_results_{}.pkl'.format(
            doing_day_analysis))
        def cal_using_two_dim_mc_wrapper():
            return cal_using_what_model('two_dim_mc')

        two_dim_mc_results = cal_using_two_dim_mc_wrapper

        # %% empirical
        empirical_path = ''.join((specific_this_wind_turbine.results_path,
                                  '3d_cvine_gmcm_model_use_ws_ahead_1/',
                                  specific_this_wind_turbine.__str__(),
                                  '/PMAPS_paper/'))
        try_to_find_folder_path_otherwise_make_one(empirical_path)

        @load_exist_pkl_file_otherwise_run_and_save(empirical_path + 'empirical_results_{}.pkl'.format(
            doing_day_analysis
        ))
        def cal_using_empirical_wrapper():
            return cal_using_what_model('empirical')

        empirical_results = cal_using_empirical_wrapper

        """
        计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值计算值
        """

        def cal_what_model_errors(what_model_name: str):
            nonlocal whole_year_test_buff
            what_model_errors = {}.fromkeys(('mae', 'rmse', 'epsilon_mae', 'epsilon_rmse', 'delta_u',
                                             'over_estimate', 'over_estimate_in_pct',
                                             'under_estimate', 'under_estimate_in_pct',
                                             'model_output_total', 'model_output_total_dividing_target_total',
                                             'model_output_total_plus', 'model_output_total_plus_dividing_target_total'
                                             ), np.nan)

            time_mask_and_cat_0_mask_model = time_mask_and_cat_0_mask

            if what_model_name == 'two_dim_mc':
                what_results = two_dim_mc_results
            elif what_model_name == 'cvine_gmcm':
                what_results = cvine_gmcm_results
            elif what_model_name == 'empirical':
                what_results = empirical_results
            else:
                raise
            what_mean = 'pout_by_what_model_mean'
            what_5 = 'pout_by_what_model_5'
            what_95 = 'pout_by_what_model_95'

            # %% 有些模型只考虑region_a，所以这里对其它region进行修正
            what_results[what_5][np.bitwise_or(region_1_mask, hard_cut_off_mask)] = 0.
            what_results[what_mean][np.bitwise_or(region_1_mask, hard_cut_off_mask)] = 0.
            what_results[what_95][np.bitwise_or(region_1_mask, hard_cut_off_mask)] = 0.

            what_results[what_5][hard_rated_mask] = 3000.
            what_results[what_mean][hard_rated_mask] = 3000.
            what_results[what_95][hard_rated_mask] = 3000.

            # %% 正式计算errors的代码
            what_model_errors['DeterministicError'] = DeterministicError(
                target=test_data.get('test_pout_actual')[time_mask_and_cat_0_mask] / 3000,
                model_output=what_results[what_mean][time_mask_and_cat_0_mask_model].flatten() / 3000)
            what_model_errors.update({
                'mae': what_model_errors['DeterministicError'].cal_mean_absolute_error(),
                'rmse': what_model_errors['DeterministicError'].cal_root_mean_square_error()})
            if what_model_name != 'mfr':
                what_model_errors.update(ProbabilisticErrorIETPaperMethod(
                    target=np.stack((test_data.get('test_pout_actual_5')[time_mask_and_cat_0_mask] / 3000,
                                     test_data.get('test_pout_actual_95')[time_mask_and_cat_0_mask] / 3000),
                                    axis=1),
                    model_output=np.stack((what_results[what_5][time_mask_and_cat_0_mask_model].flatten() / 3000,
                                           what_results[what_95][time_mask_and_cat_0_mask_model].flatten() / 3000),
                                          axis=1)
                ).do_calculation())
            what_model_errors.update(EnergyBasedError(
                target=test_data.get('test_pout_actual')[time_mask_and_cat_0_mask] / 1000,
                model_output=what_results[what_mean][time_mask_and_cat_0_mask_model].flatten() / 1000,
                time_step=1 / 6).do_calculation())

            return what_model_errors

        two_dim_mc_error = cal_what_model_errors('two_dim_mc')
        cvine_gmcm_error = cal_what_model_errors('cvine_gmcm')
        empirical_error = cal_what_model_errors('empirical')

        # Form a table
        errors_table = []
        for this_error in (empirical_error, two_dim_mc_error, cvine_gmcm_error):
            errors_table.append([this_error['mae'], this_error['rmse'],
                                 this_error['epsilon_mae'], this_error['epsilon_rmse'], this_error['delta_u'],
                                 this_error['over_estimate'], this_error['over_estimate_in_pct'],
                                 this_error['under_estimate'], this_error['under_estimate_in_pct'],
                                 this_error['model_output_total'],
                                 this_error['model_output_total_dividing_target_total'],
                                 this_error['model_output_total_plus'],
                                 this_error['model_output_total_plus_dividing_target_total']])
        errors_table = np.array(errors_table)

        if all(time_mask_pure):
            savemat(cvine_gmcm_path + 'errors_table.mat', {'errors_table': errors_table})
        else:
            savemat(cvine_gmcm_path + 'errors_table_day.mat', {'errors_table': errors_table})

        """
        画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图
        """

        xx = np.linspace(day_start_idx, day_end_idx, day_end_idx - day_start_idx) / 144

        """
        # %% 大图
        def plot_day_comparison_of_model_means(with_uncertain_inputs=False):
            ax = time_series(x=xx,
                             y=test_data.get('test_pout_actual')[time_mask_pure] / 3000,
                             label='Actl-M.',
                             linestyle='-',
                             color=[0, 1, 0])
            if not with_uncertain_inputs:
                ax = time_series(x=xx,
                                 y=pout_by_mfr[time_mask_pure] / 3000,
                                 ax=ax,
                                 label='Mfr PC',
                                 linestyle='-.',
                                 color='k')
                ax = time_series(x=xx,
                                 y=_2d_gmcm_results['pout_by_what_model_mean'][time_mask_pure] / 3000,
                                 ax=ax,
                                 label='2d_gmcm',
                                 linestyle=':',
                                 color='b',
                                 linewidth=1.5)
                ax = time_series(x=xx,
                                 y=cvine_gmcm_results['pout_by_cvine_gmcm_mean'][time_mask_pure] / 3000,
                                 ax=ax,
                                 label='cvine_gmcm',
                                 linestyle='--',
                                 color='b',
                                 save_file_=cvine_gmcm_path + 'whole_day',
                                 x_label='Time stamp (x-th day in the season)',
                                 y_label='Power output (p.u.)',
                                 x_lim=(xx[0] - 0.01, xx[-1] + 0.01),
                                 y_lim=(-0.01, 1.01))
            else:
                ax = time_series(x=xx,
                                 y=_2d_grid_search_results['grid_results_mean_'] / 3000,
                                 ax=ax,
                                 label='_2d_grid_search_results',
                                 linestyle=':',
                                 color='b',
                                 linewidth=1.5)
                ax = time_series(x=xx,
                                 y=_3d_grid_search_results['grid_results_mean_'] / 3000,
                                 ax=ax,
                                 label='_3d_grid_search_results',
                                 linestyle='--',
                                 color='b',
                                 save_file_=cvine_gmcm_path + 'whole_day_with_uncertain',
                                 x_label='Time stamp (x-th day in the season)',
                                 y_label='Power output (p.u.)',
                                 x_lim=(xx[0] - 0.01, xx[-1] + 0.01),
                                 y_lim=(-0.01, 1.01))
        """

        # %% 小图
        def plot_one_model_uncertainty_and_save_csv(model_name, y_linestyle: str, *,
                                                    save_csv_name: str = None):
            if model_name == 'two_dim_mc':
                model_results = two_dim_mc_results
            elif model_name == 'cvine_gmcm':
                model_results = cvine_gmcm_results
            elif model_name == 'empirical':
                model_results = empirical_results
            else:
                raise

            y1 = model_results['pout_by_what_model_5'][time_mask_pure] / 3000
            y2 = model_results['pout_by_what_model_95'][time_mask_pure] / 3000
            y_mean = model_results['pout_by_what_model_mean'][time_mask_pure] / 3000

            ax = series_uncertainty_plot(xx,
                                         y1=test_data['test_pout_actual_5'][time_mask_pure] / 3000,
                                         y2=test_data['test_pout_actual_95'][time_mask_pure] / 3000,
                                         facecolor='g',
                                         edgecolor='g',
                                         hatch='/' * 6,
                                         linestyle='-',
                                         linewidth=1,
                                         alpha=0.2)
            ax = series_uncertainty_plot(xx,
                                         y1=y1,
                                         y2=y2,
                                         ax=ax,
                                         facecolor='b',
                                         edgecolor='b',
                                         hatch='\\' * 6,
                                         linestyle='-',
                                         linewidth=1,
                                         alpha=0.2)
            ax = series(x=xx,
                        y=test_data.get('test_pout_actual')[time_mask_pure] / 3000, ax=ax,
                        linestyle='-',
                        color=[0, 1, 0])
            ax = series(xx, y_mean, ax=ax,
                        linestyle=y_linestyle,
                        color='b',
                        save_file_=cvine_gmcm_path + model_name,
                        x_label='Time stamp (x-th day in the season)',
                        y_label='Power output (p.u.)',
                        x_lim=(xx[0] - 0.01, xx[-1] + 0.01),
                        y_lim=(-0.01, 1.01))
            if save_csv_name is not None:
                f = open(cvine_gmcm_path + save_csv_name + '.csv', 'w', encoding='utf-8')
                csv_writer = csv.writer(f)
                header = ['actual_Pout_mean', 'actual_Pout_5_percentile', 'actual_Pout_95_percentile']
                header_extend = ['_model_mean', '_model_5_percentile', '_model_95_percentile']
                header_extend = [model_name + x for x in header_extend]
                header.extend(header_extend)
                csv_writer.writerow(header)

                rows = np.stack((test_data['test_pout_actual'][time_mask_pure] / 3000,
                                 test_data['test_pout_actual_5'][time_mask_pure] / 3000,
                                 test_data['test_pout_actual_95'][time_mask_pure] / 3000,
                                 y_mean,
                                 y1,
                                 y2), axis=1)
                csv_writer.writerows(rows.tolist())

        if not (all(time_mask_pure)):
            # plot_day_comparison_of_model_means()
            # plot_day_comparison_of_model_means(True)
            plot_one_model_uncertainty_and_save_csv('two_dim_mc', ':', save_csv_name='two_dim_mc')
            plot_one_model_uncertainty_and_save_csv('cvine_gmcm', '--', save_csv_name='cvine_gmcm')
            plot_one_model_uncertainty_and_save_csv('empirical', '-.', save_csv_name='empirical')

        """
        if this_season_idx == 3:
            whole_year_mfr_error = cal_what_model_errors('whole_year_mfr')
            whole_year_2d_error = cal_what_model_errors('whole_year_2d_gmcm')
            whole_year_3d_error = cal_what_model_errors('whole_year_cvine_gmcm')
            tt = 1
            errors_table = []
            for this_error in (whole_year_mfr_error, whole_year_2d_error, whole_year_3d_error):
                errors_table.append([this_error['mae'], this_error['rmse'],
                                     this_error['epsilon_mae'], this_error['epsilon_rmse'], this_error['delta_u'],
                                     this_error['over_estimate'], this_error['over_estimate_in_pct'],
                                     this_error['under_estimate'], this_error['under_estimate_in_pct'],
                                     this_error['model_output_total'],
                                     this_error['model_output_total_dividing_target_total'],
                                     this_error['model_output_total_plus'],
                                     this_error['model_output_total_plus_dividing_target_total']])
            errors_table = np.array(errors_table)
            savemat(cvine_gmcm_path + 'errors_table_whole_year.mat',
                    {'errors_table_whole_year': errors_table})
            """
    # 计算全年的error


def compare_training_test(full_this_wind_turbine: WT):
    pass


if __name__ == '__main__':
    # 这个.csv的数据很奇怪，最大的wind power只有2941.XXX。根据Duo，按0.98 p.u.作为rated power
    # find_extreme_wind_speed_for_rated_power(3000, 0.5)
    find_extreme_wind_speed_for_rated_power_sasa()
    """
    WT分析
    """
    # wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
    # wind_turbine_2 = wind_turbines[2]
    # del wind_turbines
    # train_3d_cvine_gmcm_using_first_2_years(wind_turbine_2)
    # train_two_dim_mc_using_first_2_years(wind_turbine_2)
    # test_on_year3_and_by_seasons(wind_turbine_2, True)
    # compare_training_test(wind_turbine_2)

    """
    WF分析
    """
    # dalry_wf = create_dalry_wind_farm_obj_using_wf_filling_missing_old()
    # dalry_wf.down_sample(6)
    # #
    # dalry_wf.train_lstm_model_to_forecast_active_power_output((datetime.datetime(2007, 1, 1, 0, 0),
    #                                                            datetime.datetime(2008, 12, 31, 23, 50)),
    #                                                           x_time_step=24 * 28,
    #                                                           y_time_step=24,
    #                                                           train_times=3)
    # dalry_wf.train_lstm_model_to_forecast_wind_speed((datetime.datetime(2007, 1, 1, 0, 0),
    #                                                   datetime.datetime(2008, 12, 31, 23, 50)),
    #                                                  x_time_step=24 * 28,
    #                                                  y_time_step=24,
    #                                                  train_times=3)
    #
    # dalry_wf.train_lstm_model_to_forecast_temperature((datetime.datetime(2007, 1, 1, 0, 0),
    #                                                    datetime.datetime(2008, 12, 31, 23, 50)),
    #                                                   x_time_step=24 * 28,
    #                                                   y_time_step=24,
    #                                                   train_times=3)
    #
    # dalry_wf.test_lstm_model_to_forecast_active_power_output((datetime.datetime(2009, 1, 1, 0, 0),
    #                                                           datetime.datetime(2009, 12, 31, 23, 50)),
    #                                                          x_time_step=24 * 28,
    #                                                          y_time_step=24,
    #                                                          lstm_file_name='training_2.mat')
    #
    # dalry_wf.test_lstm_model_to_forecast_wind_speed((datetime.datetime(2009, 1, 1, 0, 0),
    #                                                  datetime.datetime(2009, 12, 31, 23, 50)),
    #                                                 x_time_step=24 * 28,
    #                                                 y_time_step=24,
    #                                                 lstm_file_name='training_2.mat')
    #
    # dalry_wf.test_lstm_model_to_forecast_temperature((datetime.datetime(2009, 1, 1, 0, 0),
    #                                                   datetime.datetime(2009, 12, 31, 23, 50)),
    #                                                  x_time_step=24 * 28,
    #                                                  y_time_step=24,
    #                                                  lstm_file_name='training_2.mat')

    # dalry_wf.train_lstm_model_to_forecast_wind_speed_and_temperature((datetime.datetime(2007, 1, 1, 0, 0),
    #                                                                   datetime.datetime(2008, 12, 31, 23, 50)))

    # dalry_wf.train_lstm_model_to_forecast_wind_speed((datetime.datetime(2007, 1, 1, 0, 0),
    #                                                   datetime.datetime(2008, 12, 31, 23, 50)))

    # dalry_wf.test_lstm_model_to_forecast_active_power_output((datetime.datetime(2009, 1, 1, 0, 0),
    #                                                           datetime.datetime(2009, 12, 31, 23, 50)),
    #                                                          lstm_file_name='training_2.mat')

    # dalry_wf.test_lstm_model_to_forecast_wind_speed((datetime.datetime(2009, 1, 1, 0, 0),
    #                                                  datetime.datetime(2009, 12, 31, 23, 50)),
    #                                                 lstm_file_name='training_0.mat')

    # dalry_wf.test_lstm_model_to_forecast_wind_speed_and_temperature((datetime.datetime(2009, 1, 1, 0, 0),
    #                                                                  datetime.datetime(2009, 12, 31, 23, 50)),
    #                                                                 lstm_file_name='training_0.mat')
