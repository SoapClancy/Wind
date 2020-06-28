from initialise_wt_or_wf_Func import load_raw_wt_from_txt_file_and_temperature_from_csv
from WT_Class import WT
from Ploting.fast_plot_Func import series, scatter, time_series, hist, scatter_density
from numpy import ndarray
import numpy as np
import pandas as pd
from PowerCurve_Class import PowerCurveByMethodOfBins, PowerCurveByMfr
from UnivariateAnalysis_Class import UnivariatePDFOrCDFLike, UnivariateGaussianMixtureModel
import matplotlib as plt
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple
import copy
import datetime
from Time_Processing.Season_Enum import SeasonTemplate1
from scipy.io import loadmat, savemat
from project_path_Var import project_path_
from File_Management.load_save_Func import load_npy_file, load_exist_pkl_file_otherwise_run_and_save, load_pkl_file, \
    save_pkl_file
from File_Management.path_and_file_management_Func import try_to_find_path_otherwise_make_one
from BivariateAnalysis_Class import Bivariate
from ErrorEvaluation_Class import ProbabilisticErrorIETPaperMethod, EnergyBasedError, DeterministicError
from Ploting.fast_plot_Func import time_series, vlines
from Ploting.uncertainty_plot_Func import series_uncertainty_plot
from Correlation_Modeling.Copula_Class import GMCM, THREE_DIM_CVINE_CONSTRUCTION, VineCopula, VineGMCMCopula
from PowerCurve_Class import PowerCurve
import csv


def train_using_first_2_or_3_years_and_by_seasons(full_this_wind_turbine: WT, years: int):
    """
    这是IET paper的training
    """
    full_this_wind_turbine_pro = copy.deepcopy(full_this_wind_turbine)
    if years == 2:
        full_this_wind_turbine_pro.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                               datetime.datetime(2008, 11, 30, 23, 50))
    elif years == 3:
        pass
    else:
        raise Exception
    for this_season in SeasonTemplate1.__members__:
        specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine_pro)
        specific_this_wind_turbine.do_truncate_by_season(this_season)
        # %% re-calculate outliers
        specific_this_wind_turbine.identify_outlier()
        # %% Plot to see different seasons by scatters
        # specific_this_wind_turbine.plot_wind_speed_to_active_power_output_scatter((0,))
        # specific_this_wind_turbine.plot_wind_speed_to_active_power_output_scatter(
        #     (5, 4, 3, 2, 1, 0),
        #     show_category_color=('indigo', 'r', 'b', 'aqua', 'k', 'g'),
        #     show_category_label=('CAT-V', 'CAT-IV', 'CAT-III', 'CAT-II', 'CAT-I', 'Normal'))
        # %% re-training
        specific_this_wind_turbine.fit_2d_conditional_probability_model_by_gmm(bin_step=0.5)
        specific_this_wind_turbine.fit_3d_cvine_gmcm_model()


def load_test_data_for_iet_paper(season_id: int) -> dict:
    # 奇葩转换
    if season_id <= 2:
        season_id = season_id + 1
    else:
        season_id = 0
    path_ = project_path_ + 'Data/Results/IET_paper/requested files with seasons corrected/'
    # %% 提取测试用的数据。给IET paper。这些数据来自意大利
    test_data = dict.fromkeys(('test_ws_actual', 'test_ws_mc_mean', 'test_ws_mc_5', 'test_ws_mc_95',
                               'test_wd_actual', 'test_wd_mc_mean', 'test_wd_mc_5', 'test_wd_mc_95',
                               'test_pout_actual', 'test_pout_actual_5', 'test_pout_actual_95',
                               'test_outlier_category'))
    test_data['test_ws_actual'] = loadmat(path_ + 'WS.mat')['wind_speed_da_mandare'][0, season_id].flatten()
    test_data['test_ws_mc_mean'] = loadmat(
        path_ + 'WS_forecasted_(Pout-WS model).mat')['predicted_values_simple_model'][0, season_id].flatten()
    test_data['test_ws_mc_5'] = loadmat(
        path_ + '5 percentile of WS for Pout-WS model.mat')['alpha_down_'][0, season_id].flatten()
    test_data['test_ws_mc_95'] = loadmat(
        path_ + '95 percentile of WS for Pout-WS model.mat')['alpha_up_'][0, season_id].flatten()

    test_data['test_wd_actual'] = loadmat(path_ + 'WD.mat')['wind_direction_da_mandare'][0, season_id].flatten()

    test_data['test_pout_actual'] = loadmat(path_ + 'Pout.mat')['potenza_da_passare'][0, season_id].flatten() * 3000
    test_data['test_pout_actual_5'] = loadmat(
        path_ + '5 percentile Pout.mat')['alpha_down_m'][0, season_id].flatten() * 3000
    test_data['test_pout_actual_95'] = loadmat(
        path_ + '95 percentile Pout.mat')['alpha_up_m'][0, season_id].flatten() * 3000

    test_data['test_outlier_category'] = loadmat(path_ + 'category_outliers.mat')['outlier'][0, season_id].flatten()

    return test_data


def for_iet_paper_wd_from_ws(full_this_wind_turbine: WT):
    full_this_wind_turbine.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                       datetime.datetime(2008, 11, 30, 23, 50))

    for this_season_idx, this_season in enumerate(SeasonTemplate1.__members__):
        specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine)
        specific_this_wind_turbine.do_truncate_by_season(this_season)

        path_ = ''.join((specific_this_wind_turbine.results_path,
                         'for_iet_paper_wd_from_ws/',
                         specific_this_wind_turbine.__str__() + '/'))
        try_to_find_path_otherwise_make_one(path_)

        flag = np.bitwise_and(specific_this_wind_turbine.outlier_category == 0,
                              specific_this_wind_turbine.outlier_category_detailed[
                                  'absolute wind direction'].values == 0)

        bivariate = Bivariate(specific_this_wind_turbine.measurements['wind speed'].values[flag],
                              specific_this_wind_turbine.measurements['absolute wind direction'].values[flag],
                              predictor_var_name='WS',
                              dependent_var_name='WD',
                              bin_step=0.5)
        wd_given_ws = bivariate.cal_mob_statistic(np.arange(0.05, 1., 0.05))
        savemat(path_ + this_season + '.mat', {'wd_given_ws': wd_given_ws})

        test_data = load_test_data_for_iet_paper(this_season_idx)

        def find_wd_series(have_ws_series):
            wd_series = []
            for i in have_ws_series:
                row = bivariate.find_mob_key_according_to_mob_or_mob_fitting_like_dict(
                    i, bivariate.mob
                )['nearest_not_none_bin_keys']
                wd_series.append(
                    [wd_given_ws[row, 1], wd_given_ws[row, -1]]
                )
            return np.array(wd_series)

        test_ws_mc_5_wd = find_wd_series(test_data.get('test_ws_mc_5'))
        test_ws_mc_95_wd = find_wd_series(test_data.get('test_ws_mc_95'))
        savemat(path_ + this_season + 'test_ws_mc_5_wd.mat', {'test_ws_mc_5_wd': test_ws_mc_5_wd})
        savemat(path_ + this_season + 'test_ws_mc_95_wd.mat', {'test_ws_mc_95_wd': test_ws_mc_95_wd})
        # Union set
        test_ws_mc_wd_union = np.full(test_ws_mc_5_wd.shape, np.nan)
        test_ws_mc_wd_union[:, 0] = np.min(np.stack(
            (test_ws_mc_5_wd[:, 0], test_ws_mc_95_wd[:, 0]), axis=1
        ), axis=1)
        test_ws_mc_wd_union[:, 1] = np.max(np.stack(
            (test_ws_mc_5_wd[:, 1], test_ws_mc_95_wd[:, 1]), axis=1
        ), axis=1)
        savemat(path_ + this_season + 'test_ws_mc_wd_union.mat', {'test_ws_mc_wd_union': test_ws_mc_wd_union})


def get_ws_wd_series_and_they_weights_for_grid_search_model(full_this_wind_turbine: WT, years=2):
    """
    IET paper. grid search.
    """
    full_this_wind_turbine_pro = copy.deepcopy(full_this_wind_turbine)
    if years != 2:
        raise
    full_this_wind_turbine_pro.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                           datetime.datetime(2008, 11, 30, 23, 50))
    for this_season_idx, this_season in enumerate(SeasonTemplate1.__members__):
        specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine_pro)
        specific_this_wind_turbine.do_truncate_by_season(this_season)
        path_ = ''.join((specific_this_wind_turbine.results_path,
                         '3d_cvine_gmcm_model/',
                         specific_this_wind_turbine.__str__() + '/IET_paper/'))
        # %% 提取测试用的数据。给IET paper
        test_data = load_test_data_for_iet_paper(this_season_idx)
        # 载入ws_percentile, wd_percentile数据
        ws_5 = test_data.get('test_ws_mc_5')
        ws_95 = test_data.get('test_ws_mc_95')

        wd_file_ = ''.join((project_path_,
                            'Data/Results/for_iet_paper_wd_from_ws/',
                            specific_this_wind_turbine.__str__() + '/',
                            this_season + 'test_ws_mc_wd_union.mat'))
        wd_5 = loadmat(wd_file_)['test_ws_mc_wd_union'][:, 0]
        wd_95 = loadmat(wd_file_)['test_ws_mc_wd_union'][:, 1]
        # %% grid combination
        ws_grid = np.linspace(ws_5, ws_95, 10, axis=1)
        wd_grid = np.linspace(wd_5, wd_95, 10, axis=1)
        ws_wd_grid_in_tuple = []
        for i in range(ws_grid.shape[1]):
            for j in range(wd_grid.shape[1]):
                ws_wd_grid_in_tuple.append(np.stack((ws_grid[:, i], wd_grid[:, j]), axis=1))
        ws_wd_grid_in_tuple = tuple(ws_wd_grid_in_tuple)
        save_pkl_file(path_ + 'ws_wd_grid_in_tuple.pkl', ws_wd_grid_in_tuple)


def test_on_year3_and_by_seasons(full_this_wind_turbine: WT, years: int, doing_day_analysis: bool = False,
                                 *, help_vincenzo: bool = False):
    """
    这是IET paper的test
    """
    full_this_wind_turbine_pro = copy.deepcopy(full_this_wind_turbine)
    # %%选择用前两年数据生成的模型还是全部三年数据生成的模型
    if years == 2:
        full_this_wind_turbine_pro.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                               datetime.datetime(2008, 11, 30, 23, 50))
    elif years == 3:
        pass
    else:
        raise Exception
    whole_year_test_buff = {}.fromkeys(('mean', '5', '95'))
    for key in whole_year_test_buff.keys():
        whole_year_test_buff[key] = []
    whole_year_model_buff = {}.fromkeys(('mfr', '2d_gmcm', 'cvine_gmcm'), {'mean': [], '5': [], '95': []})
    for key in whole_year_model_buff.keys():
        whole_year_model_buff[key] = {'mean': [], '5': [], '95': []}
    for this_season_idx, this_season in enumerate(SeasonTemplate1.__members__):
        specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine_pro)
        specific_this_wind_turbine.do_truncate_by_season(this_season)
        # %% 提取测试用的数据。给IET paper
        test_data = load_test_data_for_iet_paper(this_season_idx)
        # %% 提取boundary的数据，因为只考虑region_a
        boundary_path_ = ''.join((specific_this_wind_turbine.results_path,
                                  '3d_cvine_gmcm_model/' + specific_this_wind_turbine.__str__() + '/'))
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
        # %% Mfr
        pout_by_mfr = specific_this_wind_turbine.estimate_active_power_output_by_mfr_power_curve(
            test_data.get('test_ws_actual'))

        def cal_using_what_model(what_model_name):
            pout_by_what_model_mean = np.full(test_data.get('test_ws_actual').size, np.nan)
            pout_by_what_model_5 = np.full(test_data.get('test_ws_actual').size, np.nan)
            pout_by_what_model_95 = np.full(test_data.get('test_ws_actual').size, np.nan)
            if what_model_name == '2d_gmcm':
                pout_by_what_model = specific_this_wind_turbine.estimate_active_power_output_by_2d_gmcm_model(
                    np.reshape(test_data.get('test_ws_actual'), (-1, 1))[region_a_mask_and_time_mask])
            elif what_model_name == 'cvine_gmcm':
                pout_by_what_model = specific_this_wind_turbine.estimate_active_power_output_by_3d_cvine_gmcm_model(
                    np.stack((test_data.get('test_ws_actual'), test_data.get('test_wd_actual')),
                             axis=1)[region_a_mask_and_time_mask, :])
            elif what_model_name == 'gmm':
                pout_by_what_model = \
                    specific_this_wind_turbine.estimate_active_power_output_by_2d_conditional_probability_model_by_gmm(
                        test_data.get('test_ws_actual')[region_a_mask_and_time_mask], bin_step=0.5)
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

        # %% 2D gmcm
        _2d_gmcm_path = ''.join((specific_this_wind_turbine.results_path,
                                 '2d_gmcm_model/',
                                 specific_this_wind_turbine.__str__(),
                                 '/IET_paper/'))
        try_to_find_path_otherwise_make_one(_2d_gmcm_path)

        @load_exist_pkl_file_otherwise_run_and_save(_2d_gmcm_path + '_2d_gmcm_results.pkl')
        def cal_using_2d_gmcm_wrapper():
            return cal_using_what_model('2d_gmcm')

        _2d_gmcm_results = cal_using_2d_gmcm_wrapper

        # %% c-vine GMCM
        cvine_gmcm_path = ''.join((specific_this_wind_turbine.results_path,
                                   '3d_cvine_gmcm_model/',
                                   specific_this_wind_turbine.__str__(),
                                   '/IET_paper/'))
        try_to_find_path_otherwise_make_one(cvine_gmcm_path)

        @load_exist_pkl_file_otherwise_run_and_save(cvine_gmcm_path + 'cvine_gmcm_results.pkl')
        def cal_using_cvine_gmcm_wrapper():
            return cal_using_what_model('cvine_gmcm')

        cvine_gmcm_results = cal_using_cvine_gmcm_wrapper

        # %% gmm
        gmm_path = ''.join((specific_this_wind_turbine.results_path,
                            '2d_conditional_probability_by_gmm/',
                            specific_this_wind_turbine.__str__(), ' bin_step=0.5/',
                            'IET_paper/'))
        try_to_find_path_otherwise_make_one(gmm_path)

        @load_exist_pkl_file_otherwise_run_and_save(gmm_path + 'gmm_results.pkl')
        def cal_using_gmm_wrapper():
            return cal_using_what_model('gmm')

        gmm_results = cal_using_gmm_wrapper

        # %% grid search
        _3d_grid_search_results = load_pkl_file(cvine_gmcm_path + 'grid_search_results.pkl')
        _2d_grid_search_results = load_pkl_file(_2d_gmcm_path + 'grid_search_results.pkl')

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
            if what_model_name == 'mfr':
                what_results = pout_by_mfr
                what_mean = np.full(pout_by_mfr.shape, True)
                what_5 = what_95 = None
            elif what_model_name == 'gmm':
                what_results = gmm_results
                what_mean = 'pout_by_gmm_mean'
                what_5 = 'pout_by_gmm_5'
                what_95 = 'pout_by_gmm_95'
            elif what_model_name == 'cvine_gmcm':
                what_results = cvine_gmcm_results
                what_mean = 'pout_by_cvine_gmcm_mean'
                what_5 = 'pout_by_cvine_gmcm_5'
                what_95 = 'pout_by_cvine_gmcm_95'
            elif (what_model_name == '3d_grid_search') or (what_model_name == '2d_grid_search'):
                if all(time_mask_pure):
                    return what_model_errors
                if what_model_name == '3d_grid_search':
                    what_results = _3d_grid_search_results
                else:
                    what_results = _2d_grid_search_results
                what_mean = 'grid_results_mean_'
                what_5 = 'grid_results_5_'
                what_95 = 'grid_results_95_'
                time_mask_and_cat_0_mask_model = np.full(what_results[what_mean].shape, True)
            elif what_model_name == '2d_gmcm':
                what_results = _2d_gmcm_results
                what_mean = 'pout_by_what_model_mean'
                what_5 = 'pout_by_what_model_5'
                what_95 = 'pout_by_what_model_95'
            elif what_model_name == 'whole_year_mfr':
                what_results = whole_year_model_buff['mfr']
                what_mean = 'mean'
                what_5 = '5'
                what_95 = '95'
            elif what_model_name == 'whole_year_2d_gmcm':
                what_results = whole_year_model_buff['2d_gmcm']
                what_mean = 'mean'
                what_5 = '5'
                what_95 = '95'
            elif what_model_name == 'whole_year_cvine_gmcm':
                what_results = whole_year_model_buff['cvine_gmcm']
                what_mean = 'mean'
                what_5 = '5'
                what_95 = '95'
            else:
                raise

            # %% 有些模型只考虑region_a，所以这里对其它region进行修正
            if any((what_model_name == 'gmm', what_model_name == 'cvine_gmcm', what_model_errors == '2d_gmcm')):
                what_results[what_5][np.bitwise_or(region_1_mask, hard_cut_off_mask)] = 0.
                what_results[what_mean][np.bitwise_or(region_1_mask, hard_cut_off_mask)] = 0.
                what_results[what_95][np.bitwise_or(region_1_mask, hard_cut_off_mask)] = 0.

                what_results[what_5][hard_rated_mask] = 3000.
                what_results[what_mean][hard_rated_mask] = 3000.
                what_results[what_95][hard_rated_mask] = 3000.

            # whole year计算
            if any((what_model_name == 'mfr', what_model_name == '2d_gmcm', what_model_name == 'cvine_gmcm')):
                whole_year_model_buff[what_model_name]['mean'].extend(
                    what_results[what_mean][time_mask_and_cat_0_mask_model].flatten() / 3000)
                if what_model_name != 'mfr':
                    whole_year_model_buff[what_model_name]['5'].extend(
                        what_results[what_5][time_mask_and_cat_0_mask] / 3000)
                    whole_year_model_buff[what_model_name]['95'].extend(
                        what_results[what_95][time_mask_and_cat_0_mask] / 3000)

            # %% 正式计算errors的代码
            if 'whole_year_' not in what_model_name:
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
            else:
                what_model_errors['DeterministicError'] = DeterministicError(
                    target=np.array(whole_year_test_buff['mean']),
                    model_output=np.array(what_results[what_mean]))
                what_model_errors.update({
                    'mae': what_model_errors['DeterministicError'].cal_mean_absolute_error(),
                    'rmse': what_model_errors['DeterministicError'].cal_root_mean_square_error()})
                if 'mfr' not in what_model_name:
                    what_model_errors.update(ProbabilisticErrorIETPaperMethod(
                        target=np.stack((np.array(whole_year_test_buff['5']), np.array(whole_year_test_buff['95'])),
                                        axis=1),
                        model_output=np.stack((np.array(what_results[what_5]), np.array(what_results[what_95])),
                                              axis=1)
                    ).do_calculation())
                what_model_errors.update(EnergyBasedError(
                    target=np.array(whole_year_test_buff['mean']) * 3,
                    model_output=np.array(what_results[what_mean]) * 3,
                    time_step=1 / 6).do_calculation())
            return what_model_errors

        mfr_error = cal_what_model_errors('mfr')
        _2d_gmcm_error = cal_what_model_errors('2d_gmcm')
        cvine_gmcm_error = cal_what_model_errors('cvine_gmcm')

        _2d_grid_search_results_error = cal_what_model_errors('2d_grid_search')
        _3d_grid_search_results_error = cal_what_model_errors('3d_grid_search')
        gmm_error = cal_what_model_errors('gmm')

        # Form a table
        errors_table = []
        for this_error in (mfr_error, _2d_gmcm_error, cvine_gmcm_error,
                           _2d_grid_search_results_error, _3d_grid_search_results_error, gmm_error):
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
        # 全年计算
        whole_year_test_buff['mean'].extend(test_data.get('test_pout_actual')[time_mask_and_cat_0_mask] / 3000)
        whole_year_test_buff['5'].extend(test_data.get('test_pout_actual_5')[time_mask_and_cat_0_mask] / 3000)
        whole_year_test_buff['95'].extend(test_data.get('test_pout_actual_95')[time_mask_and_cat_0_mask] / 3000)

        """
        画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图画图
        """
        xx = np.linspace(day_start_idx, day_end_idx, day_end_idx - day_start_idx) / 144
        # 载入vincenzo数据
        path_vincenzo = project_path_ + 'Data/Results/IET_paper/2/'
        if this_season == 'winter':
            vincenzo_2d_mean = loadmat(path_vincenzo + 'figura a/forecast Pout WS model.mat')[
                                   'resulted_mean_P'][0, 7862:8007].flatten()
            vincenzo_2d_5 = loadmat(path_vincenzo + 'figura b/5 percentile Pout WS  model.mat')[
                'alpha_down_2'][0, 0].flatten()[time_mask_pure]
            vincenzo_2d_95 = loadmat(path_vincenzo + 'figura b/95 percentile Pout WS  model.mat')[
                'alpha_up_2'][0, 0].flatten()[time_mask_pure]

            vincenzo_3d_mean = loadmat(path_vincenzo + 'figura a/forecast Pout WS WD model.mat')[
                                   'expected_power'][0, 0].flatten()[7862:8007]
            vincenzo_3d_5 = loadmat(path_vincenzo + 'figura c/5 percentile Pout WS WD model.mat')[
                'alpha_down'][0, 0].flatten()[time_mask_pure]
            vincenzo_3d_95 = loadmat(path_vincenzo + 'figura c/95 percentile Pout WS WD model.mat')[
                'alpha_up'][0, 0].flatten()[time_mask_pure]

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
            # 画意大利的图
            if all((help_vincenzo, this_season == 'winter', doing_day_analysis)):
                ax = None
                ax = time_series(x=xx,
                                 y=test_data.get('test_pout_actual')[time_mask_pure] / 3000,
                                 label='Actl-M.',
                                 linestyle='-',
                                 color=[0, 1, 0])
                ax = time_series(x=xx,
                                 y=pout_by_mfr[time_mask_pure] / 3000,
                                 ax=ax,
                                 label='Mfr PC',
                                 linestyle='-.',
                                 color='k')
                ax = time_series(x=xx,
                                 y=vincenzo_2d_mean,
                                 ax=ax,
                                 label='2d_gmcm',
                                 linestyle=':',
                                 color='b',
                                 linewidth=1.5)
                ax = time_series(x=xx,
                                 y=vincenzo_3d_mean,
                                 ax=ax,
                                 label='cvine_gmcm',
                                 linestyle='--',
                                 color='b',
                                 save_file_=cvine_gmcm_path + 'vincenzo_figure_a',
                                 x_label='Time stamp (x-th day in the season)',
                                 y_label='Power output (p.u.)',
                                 x_lim=(xx[0] - 0.01, xx[-1] + 0.01),
                                 y_lim=(-0.01, 1.01))

        # %% 小图
        def plot_one_model_uncertainty_and_save_csv(model_name, y_linestyle: str, *,
                                                    save_csv_name: str = None):
            if model_name == '2d_gmcm':
                y1 = _2d_gmcm_results['pout_by_what_model_5'][time_mask_pure] / 3000
                y2 = _2d_gmcm_results['pout_by_what_model_95'][time_mask_pure] / 3000
                y_mean = _2d_gmcm_results['pout_by_what_model_mean'][time_mask_pure] / 3000
            elif model_name == 'cvine_gmcm':
                y1 = cvine_gmcm_results['pout_by_cvine_gmcm_5'][time_mask_pure] / 3000
                y2 = cvine_gmcm_results['pout_by_cvine_gmcm_95'][time_mask_pure] / 3000
                y_mean = cvine_gmcm_results['pout_by_cvine_gmcm_mean'][time_mask_pure] / 3000
            elif model_name == '2d_grid_search':
                y1 = _2d_grid_search_results['grid_results_5_'] / 3000
                y2 = _2d_grid_search_results['grid_results_95_'] / 3000
                y_mean = _2d_grid_search_results['grid_results_mean_'] / 3000
            elif model_name == '3d_grid_search':
                y1 = _3d_grid_search_results['grid_results_5_'] / 3000
                y2 = _3d_grid_search_results['grid_results_95_'] / 3000
                y_mean = _3d_grid_search_results['grid_results_mean_'] / 3000
            elif (model_name == 'vincenzo_2d'):
                if not help_vincenzo:
                    return
                y1 = vincenzo_2d_5
                y2 = vincenzo_2d_95
                y_mean = vincenzo_2d_mean
            elif (model_name == 'vincenzo_3d'):
                if not help_vincenzo:
                    return
                y1 = vincenzo_3d_5
                y2 = vincenzo_3d_95
                y_mean = vincenzo_3d_mean
            else:
                raise

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
            plot_day_comparison_of_model_means()
            plot_day_comparison_of_model_means(True)

            plot_one_model_uncertainty_and_save_csv('2d_gmcm', ':', save_csv_name='2d_gmcm')
            plot_one_model_uncertainty_and_save_csv('cvine_gmcm', '--', save_csv_name='cvine_gmcm')
            plot_one_model_uncertainty_and_save_csv('2d_grid_search', ':', save_csv_name='2d_grid_search')
            plot_one_model_uncertainty_and_save_csv('3d_grid_search', '--', save_csv_name='3d_grid_search')
            plot_one_model_uncertainty_and_save_csv('vincenzo_2d', ':')
            plot_one_model_uncertainty_and_save_csv('vincenzo_3d', '--')

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

    tt = 1
    # 计算全年的error


def plot_copula(full_this_wind_turbine: WT, years: int):
    """
    这是IET paper的test
    """
    full_this_wind_turbine_pro = copy.deepcopy(full_this_wind_turbine)
    # %%选择用前两年数据生成的模型还是全部三年数据生成的模型
    if years == 2:
        full_this_wind_turbine_pro.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                               datetime.datetime(2008, 11, 30, 23, 50))
    elif years == 3:
        pass
    else:
        raise Exception
    for this_season_idx, this_season in enumerate(SeasonTemplate1.__members__):
        if this_season != 'winter':
            continue
        specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine_pro)
        specific_this_wind_turbine.do_truncate_by_season(this_season)
        path_ = ''.join((specific_this_wind_turbine.results_path,
                         '3d_cvine_gmcm_model/',
                         specific_this_wind_turbine.__str__(), '/'))
        vine_gmcm_copula = VineGMCMCopula(construction=THREE_DIM_CVINE_CONSTRUCTION,
                                          gmcm_model_folder_for_construction_path_=path_ + 'a/',
                                          marginal_distribution_file_=path_ + 'a/marginal.pkl')
        for i, this_gmcm in enumerate(vine_gmcm_copula.pair_copula_instance_of_each_edge):
            this_gmcm.plot_simulated(n=5000,
                                     x_label='u{}|{}'.format(vine_gmcm_copula.construction[i][0],
                                                             vine_gmcm_copula.resolved_construction[
                                                                 'conditioning'][i]),
                                     y_label='u{}|{}'.format(vine_gmcm_copula.construction[i][1],
                                                             vine_gmcm_copula.resolved_construction['conditioning'][i]),
                                     title='c_{}|{} Copula density'.format(
                                         vine_gmcm_copula.resolved_construction['conditioned'][i],
                                         vine_gmcm_copula.resolved_construction['conditioning'][i]),
                                     save_file_=str(i))
        GMCM(gmcm_model_file_=path_ + 'a/GMCM_(2, 3).mat',
             marginal_distribution_file_=path_ + 'a/marginal_for_GMCM_(2, 3).pkl'
             ).plot_simulated(n=5000, x_label='u2', y_label='u3', title='c_23', save_file_='4')


def plot_empirical_as_eg(full_this_wind_turbine: WT, years: int = 2):
    full_this_wind_turbine_pro = copy.deepcopy(full_this_wind_turbine)
    # %%选择用前两年数据生成的模型还是全部三年数据生成的模型
    if years == 2:
        full_this_wind_turbine_pro.do_truncate(datetime.datetime(2007, 1, 1, 0, 0),
                                               datetime.datetime(2008, 11, 30, 23, 50))
    else:
        pass
    specific_this_wind_turbine = copy.deepcopy(full_this_wind_turbine_pro)
    specific_this_wind_turbine.do_truncate_by_season('winter')

    select_flag = np.all(np.stack((specific_this_wind_turbine.measurements['wind speed'].values < 12,
                                   specific_this_wind_turbine.measurements['wind speed'].values >= 11.5,
                                   specific_this_wind_turbine.outlier_category == 0,
                                   specific_this_wind_turbine.measurements['absolute wind direction'].values < 270,
                                   specific_this_wind_turbine.measurements['absolute wind direction'].values >= 240
                                   ), axis=1), axis=1)
    hist_pout = specific_this_wind_turbine.measurements['active power output'].values[select_flag]
    ax = hist(hist_pout / 3000, normed=True, label='Measurements')
    mfr = PowerCurveByMfr()(11.75)
    ax = vlines(mfr / 3000, ax=ax, label='Mfr PC',
                linestyles='-.', color='k')
    ax = vlines(np.percentile(hist_pout, 5) / 3000, ax=ax, label='5$^\mathrm{th}$ percentile', color='g')
    ax = vlines(np.percentile(hist_pout, 95) / 3000, ax=ax, label='95$^\mathrm{th}$ percentile',
                linestyles=':', color='b')
    ax = vlines(np.mean(hist_pout) / 3000, ax=ax, y_lim=(0, 17.6), label='Mean value',
                x_lim=(-0.01, 1.01), x_label='Power output (p.u.) (WS: 11.5~12 m/s, WD: 240~270 degrees)',
                y_label='Probability density',
                linestyles='-', color='r')


if __name__ == '__main__':
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
    for this_wind_turbine in tuple([wind_turbines[2]]):
        # plot_copula(this_wind_turbine, 2)
        # for_iet_paper_wd_from_ws(this_wind_turbine)
        # %% Train the model
        # train_using_first_2_or_3_years_and_by_seasons(this_wind_turbine, 2)
        # train_using_first_2_or_3_years_and_by_seasons(this_wind_turbine, 3)

        # %% grid search
        # get_ws_wd_series_and_they_weights_for_grid_search_model(this_wind_turbine)

        # %% Test the model
        # test_on_year3_and_by_seasons(this_wind_turbine, 2)
        test_on_year3_and_by_seasons(this_wind_turbine, 2, True, help_vincenzo=False)

        # %% Plot empirical as an example
        # plot_empirical_as_eg(this_wind_turbine)
