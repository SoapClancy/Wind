from PowerCurve_Class import PowerCurveByMfr
import pandas as pd
from prepare_datasets import load_raw_wt_from_txt_file_and_temperature_from_csv, \
    load_high_resol_for_averaging_effects_analysis
from BivariateAnalysis_Class import MethodOfBins
from Ploting.fast_plot_Func import *
from Wind_Class import Wind
from Ploting.adjust_Func import reassign_linestyles_recursively_in_ax, adjust_lim_label_ticks
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray, IntOneDimensionNdarray
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save, load_pkl_file, \
    load_exist_npy_file_otherwise_run_and_save
from project_utils import *
from pathlib import Path
from Data_Preprocessing.float_precision_control_Func import \
    covert_to_str_one_dimensional_ndarray
from Ploting.uncertainty_plot_Func import plot_from_uncertainty_like_dataframe
from ConvenientDataType import UncertaintyDataFrame, StrOneDimensionNdarray
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import truncnorm as scipy_truncnorm
from Data_Preprocessing import float_eps
from Data_Preprocessing.float_precision_control_Func import limit_ndarray_max_and_min
from Ploting.adjust_Func import adjust_lim_label_ticks
from UnivariateAnalysis_Class import UnivariateGaussianMixtureModel, GaussianMixture
from matplotlib import cm, colors
from scipy.stats import norm
from itertools import product
from functools import reduce
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

# %% Global variables
# choose manufacturer power curve
FIXED_MFR_PC = PowerCurveByMfr('1.12')
# load wind turbine
wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
THIS_WIND_TURBINE = wind_turbines[1]
outlier = load_pkl_file(THIS_WIND_TURBINE.default_results_saving_path['outlier'])['DataCategoryData obj']
THIS_WIND_TURBINE = THIS_WIND_TURBINE[~outlier('CAT-III')]
# choose wind speed bins and the wind speed std. in each bin
wind_speed = THIS_WIND_TURBINE['wind speed'].values
wind_speed_std = THIS_WIND_TURBINE['wind speed std.'].values
MOB = MethodOfBins(wind_speed, wind_speed_std, bin_step=0.5, first_bin_left_boundary=0)
RANGE_MASK = np.bitwise_and(MOB.array_of_bin_boundary[:, 1] >= 0,
                            MOB.array_of_bin_boundary[:, 1] <= 29.6)
WIND_SPEED_RANGE = MOB.cal_mob_statistic_eg_quantile(np.array([1.]))[RANGE_MASK, 0]
WIND_SPEED_STD_RANGE = MOB.cal_mob_statistic_eg_quantile('mean')[RANGE_MASK, 1]
WIND_SPEED_STD_UCT = MOB.cal_mob_statistic_eg_quantile(np.arange(0, 1.001, 0.001), behaviour='new')
SIMULATION_RESOLUTION = 10
SIMULATION_TRACES = 1_000_000
SIMULATION_RETURN_PERCENTILES = covert_to_str_one_dimensional_ndarray(np.arange(0, 100.001, 0.001), '0.001')

del wind_turbines, wind_speed, wind_speed_std


def plot_raw_wind_turbine_data(_ax=None):
    _ax = scatter(
        THIS_WIND_TURBINE['wind speed'].values,
        THIS_WIND_TURBINE['active power output'].values / 3000,
        **{'x_lim': (-0.05, 29.5),
           'y_lim': (-0.05, 1.05),
           'x_label': 'Wind Speed [m/s]',
           'y_label': 'Active Power Output [p.u.]'}
    )


def plot_mfr_pc(_ax, *, mfr_pc_obj=None, **kwargs):
    mfr_pc_obj = mfr_pc_obj or FIXED_MFR_PC

    # _ax = mfr_pc_obj.plot(np.concatenate((range(0, 26),
    #                                       [25 + float_eps * 100],
    #                                       range(26, 31))),
    #                       ax=_ax, zorder=300, mode='continuous')
    if mfr_pc_obj.air_density == '0.97':
        mfr_pc_obj_kwargs = MFR_KWARGS[0]
    elif mfr_pc_obj.air_density == '1.12':
        mfr_pc_obj_kwargs = MFR_KWARGS[1]
    elif mfr_pc_obj.air_density == '1.27':
        mfr_pc_obj_kwargs = MFR_KWARGS[2]
    else:
        raise
    _ax = mfr_pc_obj.plot(**mfr_pc_obj_kwargs,
                          ax=_ax, mode='discrete')
    return _ax


def uncertainty_plot(uncertainty_dataframe: UncertaintyDataFrame, _ax=None, *, mfr_pc_obj=None):
    _ax = plot_from_uncertainty_like_dataframe(
        WIND_SPEED_RANGE,
        uncertainty_dataframe,
        covert_to_str_one_dimensional_ndarray(
            np.arange(UncertaintyDataFrame.infer_percentile_boundaries_by_sigma(1.5)[0], 50, 5),
            '0.001'),
        ax=_ax,
    )
    # _ax = series(WIND_SPEED_RANGE, (uncertainty_dataframe(68).iloc[0] + uncertainty_dataframe(68).iloc[1]) / 2,
    #              ax=_ax, label='1' + r'$\sigma$' + ' %' + '\nrange mean', color='fuchsia', linestyle='--')
    return plot_mfr_pc(_ax, mfr_pc_obj=mfr_pc_obj)


def uncertainty_plot_25_75(uncertainty_dataframe: UncertaintyDataFrame, _ax=None, **kwargs):
    _ax = plot_from_uncertainty_like_dataframe(
        WIND_SPEED_RANGE,
        uncertainty_dataframe,
        covert_to_str_one_dimensional_ndarray(np.array([25.]), '0.001'),
        ax=_ax,
        **kwargs
    )
    _ax = series(WIND_SPEED_RANGE, (uncertainty_dataframe(50).iloc[0] + uncertainty_dataframe(50).iloc[1]) / 2,
                 ax=_ax, label='25th-75th mean')

    return plot_mfr_pc(_ax)


def uncertainty_plot_sigma(uncertainty_dataframe: UncertaintyDataFrame, _ax=None, **kwargs):
    uncertainty_dataframe.columns = WIND_SPEED_RANGE
    _ax = series(WIND_SPEED_RANGE, uncertainty_dataframe.iloc[-2].values,
                 color=(0, 1, 0), linestyle='--', ax=_ax, label='SIM mean')
    # cmap = cm.get_cmap('bone')
    # norm = colors.Normalize(vmin=0, vmax=int(lower_half_percentiles.size))
    _ax = plot_mfr_pc(_ax, y_lim=(-0.2, 1.2))
    _ax = uncertainty_dataframe.sigma_percentage_plot(ax=_ax, **{'x_label': 'Wind Speed [m/s]',
                                                                 'y_label': 'Active Power Output [p.u.]'})
    # _ax = _ax.fill_between(WIND_SPEED_RANGE,
    #                        uncertainty_dataframe.iloc[-2].values - uncertainty_dataframe.iloc[-1].values,
    #                        uncertainty_dataframe.iloc[-2].values + uncertainty_dataframe.iloc[-1].values,
    #                        facecolor='w',
    #                        edgecolor='k',
    #                        linewidth=0.25,
    #                        # alpha=alpha,
    #                        )
    # (WIND_SPEED_RANGE, uncertainty_dataframe.iloc[-2].values,
    #  color=(0, 1, 0), linestyle='--', ax=_ax, label='SIM mean')

    return _ax


# def plot_wind_speed_std():
#     ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_RANGE, marker='*', mec='r', ms=8,
#                 x_label='Wind Speed [m/s]', y_label='Wind Speed Std. [m/s]',
#                 x_lim=(-0.5, 30.5))
#     return ax

def plot_wind_speed_std():
    ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_RANGE, color='fuchsia', linestyle='--', label='Mean')
    ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(0).iloc[0].values[RANGE_MASK], ax=ax, linestyle='-',
                color='orange', label='Median')

    # ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=1).iloc[0].values[RANGE_MASK], ax=ax, linestyle='-',
    #             label='1' + r'$\sigma$' + ' %', color='grey')
    # ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=1).iloc[1].values[RANGE_MASK], ax=ax, linestyle='-',
    #             color='grey')
    #
    # ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=2).iloc[0].values[RANGE_MASK], ax=ax, linestyle='--',
    #             label='2' + r'$\sigma$' + ' %', color='fuchsia')
    # ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=2).iloc[1].values[RANGE_MASK], ax=ax, linestyle='--',
    #             color='fuchsia')
    #
    # ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=3).iloc[0].values[RANGE_MASK], ax=ax, linestyle='-.',
    #             label='3' + r'$\sigma$' + ' %', color='royalblue')
    # ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=3).iloc[1].values[RANGE_MASK], ax=ax, linestyle='-.',
    #             color='royalblue')
    # #
    # ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=4.5).iloc[0].values[RANGE_MASK], ax=ax, linestyle=':',
    #             label='4.5' + r'$\sigma$' + ' %', color='black')
    # ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=4.5).iloc[1].values[RANGE_MASK], ax=ax, linestyle=':',
    #             color='black', x_label='Wind Speed [m/s]', y_label='Wind Speed Std. [m/s]',
    #             x_lim=(-0.5, 30.5))
    #
    ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=1.5).iloc[0].values[RANGE_MASK], ax=ax, linestyle=':',
                label=r'6.68 to 93.32 percentiles', color='black')
    ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_UCT(by_sigma=1.5).iloc[1].values[RANGE_MASK], ax=ax, linestyle=':',
                color='black', x_label='Wind Speed [m/s]', y_label='Wind Speed Std. [m/s]',
                x_lim=WS_POUT_2D_PLOT_KWARGS['x_lim'])
    ax.set_ylim(-0.05, 5.9)
    plt.gca().legend(ncol=4, loc='upper center', prop={'size': 10})
    return ax


# %% Actual codes
def sasa_algorithm_to_cal_possible_pout_range():
    @load_exist_pkl_file_otherwise_run_and_save(
        Path(project_path_) / 'Data/Results/transient_study/Energies_Review_paper_2020/sasa_all_range_check.pkl')
    def get_results():
        # %% Initialise the possible power output range, type pd.DataFrame
        _return_percentiles = UncertaintyDataFrame(
            index=np.append(SIMULATION_RETURN_PERCENTILES, 'mean'),
            columns=range(len(WIND_SPEED_RANGE))
        )
        # %% Calculate all possible power output ranges for ws
        wind = Wind(WIND_SPEED_RANGE, WIND_SPEED_STD_RANGE)
        for i, this_ws in enumerate(WIND_SPEED_RANGE):
            pout = []
            if i == 0:
                _return_percentiles[0] = 0
                continue
            # Make scipy_truncnorm. Weired implementation. I do not like this
            distribution_low = float(wind.transient_distribution[i].low)
            distribution_high = float(wind.transient_distribution[i].high)
            distribution_mean = float(wind.transient_distribution[i].loc)
            distribution_std = float(wind.transient_distribution[i].scale)
            a = (distribution_low - distribution_mean) / distribution_std,
            b = (distribution_high - distribution_mean) / distribution_std
            distribution = scipy_truncnorm(a=a, b=b, loc=distribution_mean, scale=distribution_std)
            # Calculate highest_possible_ws and lowest_possible_ws
            lowest_possible_ws, highest_possible_ws = distribution.ppf(
                SIMULATION_RETURN_PERCENTILES[[1, -2]].astype(float) / 100
            )
            # The loop controlling WS_low, the choices are 0, 1, 2, ..., this_ws - 0.5
            for this_ws_lower_bound in np.arange(0, this_ws, 0.5):
                # Supposing the lower possible wind speed can happen 1, 2, ..., 9 minutes in the 10min window
                for how_long_minutes_this_ws_lower_bound_happen in range(1, 10):
                    how_long_minutes_this_ws_upper_bound_happen = 10 - how_long_minutes_this_ws_lower_bound_happen
                    this_ws_upper_bound = (this_ws * 10 -
                                           how_long_minutes_this_ws_lower_bound_happen *
                                           this_ws_lower_bound) / how_long_minutes_this_ws_upper_bound_happen
                    if (this_ws_upper_bound > highest_possible_ws) or (this_ws_upper_bound < lowest_possible_ws):
                        continue  # If WS_high > highest_possible_ws, the results will not be recorded
                    # calculate the possible average power output in the 10min average window
                    pout.append(
                        np.average(FIXED_MFR_PC([this_ws_lower_bound, this_ws_upper_bound]),
                                   weights=(how_long_minutes_this_ws_lower_bound_happen,
                                            how_long_minutes_this_ws_upper_bound_happen))
                    )
            # Write the minimum and maximum possible average power output to corresponding ws
            _return_percentiles[i] = np.concatenate(
                (np.percentile(pout, _return_percentiles.index.values[:-1].astype(float)),
                 np.mean(pout, keepdims=True))
            )

        return _return_percentiles

    results = get_results()  # type: UncertaintyDataFrame
    return uncertainty_plot(results)


def sasa_pmaps_to_cal_possible_pout_range():
    @load_exist_pkl_file_otherwise_run_and_save(
        Path(project_path_) / 'Data/Results/transient_study/Energies_Review_paper_2020/sasa_pmaps_all_range_check.pkl')
    def get_results():
        THIS_WIND_TURBINE.fit_2d_conditional_probability_model_by_gmm(
            bin_step=0.5
        )
        distributions = THIS_WIND_TURBINE.estimate_active_power_output_by_2d_conditional_probability_model_by_gmm(
            WIND_SPEED_RANGE,
            bin_step=0.5
        )
        _return_percentiles = UncertaintyDataFrame(
            index=np.append(SIMULATION_RETURN_PERCENTILES, 'mean'),
            columns=range(len(WIND_SPEED_RANGE)),
            data=np.nan
        )
        for i, this_distribution in enumerate(distributions):
            unconstrained_inverse_cdf = this_distribution.find_nearest_inverse_cdf(
                SIMULATION_RETURN_PERCENTILES.astype(np.float) / 100
            )
            constrained_inverse_cdf = limit_ndarray_max_and_min(
                unconstrained_inverse_cdf,
                min_value=0,
                max_value=FIXED_MFR_PC.rated_active_power_output
            )
            # Should remember that pd.DataFrame can only be assigned from np.ndarray, not its subclasses, e.g.,
            # IntFloatConstructedOneDimensionNdarray
            _return_percentiles.iloc[:-1, i] = np.array(constrained_inverse_cdf).astype(np.float)
            _return_percentiles.iloc[-1, i] = np.min(this_distribution.mean_, 0).astype(np.float)
        return _return_percentiles

    results = get_results()  # type: UncertaintyDataFrame
    mask = np.bitwise_or(THIS_WIND_TURBINE.outlier_category == 0,
                         THIS_WIND_TURBINE.outlier_category == 5)
    _ax = scatter(THIS_WIND_TURBINE.measurements['wind speed'].values[mask],
                  THIS_WIND_TURBINE.measurements['active power output'].values[mask],
                  color='b',
                  label='Recordings')
    _ax = uncertainty_plot_sigma(results, _ax=_ax, automatic_alpha_control=True)

    return _ax


def sasa_high_resol_check():
    # %% 不合理的方法！别说是我的contribution！
    data_whole = load_high_resol_for_averaging_effects_analysis()
    for key, data in data_whole.items():
        # Only consider the recordings containing cut-out effects
        # data = data[4800 - 1:6660 - 1]
        # aggregate by 10 seconds
        data_df = pd.DataFrame(data,
                               index=pd.date_range('1/1/1970', periods=data.shape[0], freq='S'))
        data_df = data_df.resample('10S').mean()
        data = data_df.values
        scatter(data[:, 0], data[:, 1], y_lim=(-0.5, 3125),
                x_label='0.1 Hz wind speed [m/s]', y_label='0.1 Hz power output [kW]',
                x_lim=(-0.5, 30.5), color='b',
                title=key)
        # overlapping_sliding_window
        ws = []
        ws_high_resol = []
        pout = []
        for window_start_index in range(0, data.shape[0] - 60, 1):
            high_resol_ws = data[window_start_index:window_start_index + 60, 0]
            ws_high_resol.append(high_resol_ws)
            high_resol_pout = data[window_start_index:window_start_index + 60, 1]

            ws.append(np.mean(high_resol_ws))
            pout.append(np.mean(high_resol_pout))
        ws, ws_high_resol, pout = np.array(ws), np.array(ws_high_resol), np.array(pout)
        # randomly choose the possible averaging window
        np.random.seed(10)
        candidate_index = np.random.randint(0, ws.shape[0], int(ws.shape[0] / 2))
        candidate_index = np.unique(candidate_index)
        ws, ws_high_resol, pout = ws[candidate_index], ws_high_resol[candidate_index], pout[candidate_index]
        scatter(ws, pout, y_lim=(-0.5, 3125),
                x_label='(Random consecutive window) \n10-min average wind speed [m/s]',
                y_label='(Random consecutive window) \n10-min average power output [kW]',
                x_lim=(-0.5, 30.5), color='b', title=key)
        # high/low WS in each WS bin (absolute)
        _mob = MethodOfBins(np.repeat(ws, 60),
                            ws_high_resol.flatten(),
                            bin_step=0.5)
        fake = _mob.cal_mob_statistic_eg_quantile(np.array([0., 1.]))
        # nmksztsf
        stupid_ax = series(fake[:, 0], fake[:, 1], color=(0, 1, 0), label='Absolute min 0.1 Hz WS \nin a 10-min WS bin')
        stupid_ax = series(fake[:, 0], fake[:, 2], color='b', linestyle='--',
                           label='Absolute max 0.1 Hz WS\nin a 10-min WS bin', ax=stupid_ax,
                           x_label='10-min average wind speed [m/s]',
                           y_label='0.1 Hz Wind speed [m/s]',
                           x_lim=(-0.5, 30.5), title=key)
        laji_ax = series(_mob.cal_mob_statistic_eg_quantile('mean')[:, 0],
                         FIXED_MFR_PC(_mob.cal_mob_statistic_eg_quantile('mean')[:, 1]),
                         color='k', label='Pout from average WS in the bin')
        laji_ax = series(_mob.cal_mob_statistic_eg_quantile('mean')[:, 0],
                         FIXED_MFR_PC(fake[:, 1]), color='b', linestyle='--',
                         label='Pout by absolute min 0.1 Hz WS \nin a 10-min WS bin',
                         ax=laji_ax)
        laji_ax = series(_mob.cal_mob_statistic_eg_quantile('mean')[:, 0],
                         FIXED_MFR_PC(fake[:, 2]), color=(0, 1, 0), linestyle='--',
                         label='Pout by absolute max 0.1 Hz WS \nin a 10-min WS bin',
                         ax=laji_ax,
                         x_label='10-min average wind speed [m/s]',
                         y_label='Calculated power output [kW]',
                         x_lim=(-0.5, 30.5), title=key
                         )
        laji_ax.legend(loc=5)
        # high/low WS in each WS bin (concurrent)
        bins = _mob.array_of_bin_boundary
        # s东西看每个bin的series的range的最大值
        stupid_unrealistic_bins = np.full((bins.shape[0], 3), np.nan)
        for i in range(bins.shape[0]):
            mask = np.bitwise_and(ws >= bins[i, 0], ws < bins[i, 1])
            if np.sum(mask) == 0:
                continue
            stupid_temp = ws_high_resol[mask]
            stupid_temp = stupid_temp[np.argmax(np.max(stupid_temp, axis=1) - np.min(stupid_temp, axis=1))]
            stupid_unrealistic_bins[i, 1] = bins[i, 1]
            stupid_unrealistic_bins[i, 0] = np.min(stupid_temp)
            stupid_unrealistic_bins[i, 2] = np.max(stupid_temp)

        stupid_ax = series(stupid_unrealistic_bins[:, 1], stupid_unrealistic_bins[:, 0],
                           color=(0, 1, 0), label='Concurrent min 0.1 Hz WS \nin a 10-min WS bin')
        stupid_ax = series(stupid_unrealistic_bins[:, 1], stupid_unrealistic_bins[:, 2],
                           color='b', linestyle='--',
                           label='Concurrent max 0.1 Hz WS\nin a 10-min WS bin', ax=stupid_ax,
                           x_label='10-min average wind speed [m/s]',
                           y_label='0.1 Hz Wind speed [m/s]',
                           x_lim=(-0.5, 30.5), title=key)
        laji_ax = series(_mob.cal_mob_statistic_eg_quantile('mean')[:, 0],
                         FIXED_MFR_PC(_mob.cal_mob_statistic_eg_quantile('mean')[:, 1]),
                         color='k', label='Pout from average WS in the bin')
        laji_ax = series(_mob.cal_mob_statistic_eg_quantile('mean')[:, 0],
                         FIXED_MFR_PC(stupid_unrealistic_bins[:, 0]), color='b', linestyle='--',
                         label='Pout by concurrent min 0.1 Hz WS \nin a 10-min WS bin',
                         ax=laji_ax)
        laji_ax = series(_mob.cal_mob_statistic_eg_quantile('mean')[:, 0],
                         FIXED_MFR_PC(stupid_unrealistic_bins[:, 2]), color=(0, 1, 0), linestyle='--',
                         label='Pout by concurrent max 0.1 Hz WS \nin a 10-min WS bin',
                         ax=laji_ax,
                         x_label='10-min average wind speed [m/s]',
                         y_label='Calculated power output [kW]',
                         x_lim=(-0.5, 30.5), title=key
                         )
        laji_ax.legend(loc=5)


def demonstration_possible_pout_range_in_wind_speed_bins_my_proposal_new(_this_prod, **kwargs):
    this_pc = PowerCurveByMfr(_this_prod[0])
    wind_speed_std_range = MOB.cal_mob_statistic_eg_quantile(_this_prod[1])[RANGE_MASK, 1]

    @load_exist_pkl_file_otherwise_run_and_save(
        Path(project_path_) / f'Data/Results/transient_study/Energies_Review_paper_2020/'
                              f'mine_rho_{_this_prod[0]}_std_{_this_prod[1]}.pkl')
    def get_results():
        # Initialise Wind instance
        _simulated_pout = []
        for this_ws, this_ws_std in tqdm(zip(WIND_SPEED_RANGE, wind_speed_std_range), total=len(WIND_SPEED_RANGE)):
            wind = Wind(this_ws, this_ws_std)
            high_resol_wind = wind.simulate_transient_wind_speed_time_series(
                resolution=SIMULATION_RESOLUTION,
                traces_number_for_each_recording=SIMULATION_TRACES,
                sigma_func=wind.learn_transition_by_looking_at_actual_high_resol()
            )
            template = UncertaintyDataFrame.init_from_template(
                columns_number=len(high_resol_wind),
                percentiles=np.arange(UncertaintyDataFrame.infer_percentile_boundaries_by_sigma(1.5)[0],
                                      UncertaintyDataFrame.infer_percentile_boundaries_by_sigma(1.5)[1] + 1,
                                      0.001)
            )
            this_simulated_pout = this_pc.cal_with_hysteresis_control_using_high_resol_wind(
                high_resol_wind,
                return_percentiles=template,
            )
            this_simulated_pout = this_simulated_pout.rename(columns={0: str(this_ws)})
            _simulated_pout.append(this_simulated_pout)
        _simulated_pout = reduce(lambda a, b: pd.merge(a, b, left_index=True, right_index=True), _simulated_pout)
        return _simulated_pout

    simulated_pout = get_results()  # type: UncertaintyDataFrame

    return uncertainty_plot(UncertaintyDataFrame(simulated_pout), mfr_pc_obj=this_pc, **kwargs)


def demonstration_possible_pout_range_in_wind_speed_bins_my_proposal_old():
    @load_exist_pkl_file_otherwise_run_and_save(
        Path(project_path_) / 'Data/Results/transient_study/Energies_Review_paper_2020/mine_old_all_range_check.pkl')
    def get_results():
        _return_percentiles = UncertaintyDataFrame(
            index=np.append(SIMULATION_RETURN_PERCENTILES, 'mean'),
            columns=range(len(WIND_SPEED_RANGE))
        )

        # Initialise Wind instance
        wind = Wind(WIND_SPEED_RANGE, WIND_SPEED_STD_RANGE)
        for i in range(wind.transient_distribution.batch_shape[0]):
            this_distribution = wind.transient_distribution[i]
            if i == 0:
                simulated_wind_speed = tf.fill((SIMULATION_TRACES,
                                                int(wind.original_resolution / SIMULATION_RESOLUTION)),
                                               0.).numpy()
            else:
                simulated_wind_speed = this_distribution.sample(
                    (SIMULATION_TRACES,
                     int(wind.original_resolution / SIMULATION_RESOLUTION))
                ).numpy()
            # calculate pout using linear interpolation
            pout = FIXED_MFR_PC(simulated_wind_speed).reshape(simulated_wind_speed.shape)
            pout = np.mean(pout, axis=1)
            _return_percentiles[i] = np.concatenate(
                (np.percentile(pout, _return_percentiles.index.values[:-1].astype(float)),
                 np.mean(pout, keepdims=True))
            )
            del simulated_wind_speed, pout  # Release memory to prevent OOM
            print(i)
        return _return_percentiles

    simulated_pout = get_results()  # type: UncertaintyDataFrame

    return uncertainty_plot(simulated_pout)


def demonstration_iec_standard():
    """
    check Annex M, IEC standard 61400-12-1.
    :return:
    """

    @load_exist_pkl_file_otherwise_run_and_save(
        Path(project_path_) / 'Data/Results/transient_study/Energies_Review_paper_2020/iec_simulation.pkl')
    def get_results():
        _return_percentiles = UncertaintyDataFrame(
            index=np.append(SIMULATION_RETURN_PERCENTILES, 'mean'),
            columns=range(len(WIND_SPEED_RANGE))
        )

        # _results = np.full((WIND_SPEED_RANGE.shape[0], 2), np.nan)
        # _results[:, 0] = WIND_SPEED_RANGE
        for i in range(WIND_SPEED_RANGE.shape[0]):
            # if i == 0:
            #     _results[i, 1] = 0
            #     continue
            # manually specify distribution parameter
            wind_distribution = GaussianMixture()
            wind_distribution.means_ = np.array(WIND_SPEED_RANGE[i]).reshape((1, 1))
            wind_distribution.covariances_ = pow(WIND_SPEED_STD_RANGE[i], 2).reshape((1, 1, 1))
            wind_distribution.weights_ = [1]
            # leverage UnivariateGaussianMixtureModel to do boundary-specified sampling
            wind_distribution = UnivariateGaussianMixtureModel(wind_distribution,
                                                               theoretic_min_value=0)
            samples = wind_distribution.sample(int(SIMULATION_TRACES * 600 / SIMULATION_RESOLUTION))
            # according to Equation (M.1)
            # p_sim_i = tfp.monte_carlo.expectation(
            #     f=FIXED_MFR_PC.__call__,
            #     samples=samples.astype(np.float32),
            #     log_prob=tfp.distributions.Normal(loc=WIND_SPEED_RANGE[i],
            #                                       scale=WIND_SPEED_STD_RANGE[i]).log_prob
            # )
            pout = FIXED_MFR_PC(samples)
            _return_percentiles[i] = np.concatenate(
                (np.percentile(pout, _return_percentiles.index.values[:-1].astype(float)),
                 np.mean(pout, keepdims=True))
            )
        return _return_percentiles

    results = get_results()
    return uncertainty_plot(results)
    # _ax = series(results[:, 0], results[:, 1], color=(0, 1, 0), linestyle='--', label='Mean')
    #
    # return plot_mfr_pc(_ax)


if __name__ == "__main__":
    # %% 无聊的各种排列组合
    air_density_list = ('0.97', '1.12', '1.27')
    ws_std_list = ([UncertaintyDataFrame.infer_percentile_boundaries_by_sigma(1.5)[0] / 100],
                   'mean',
                   [UncertaintyDataFrame.infer_percentile_boundaries_by_sigma(1.5)[1] / 100])
    prod = list(product(air_density_list, ws_std_list))
    for i, this_prod in enumerate(prod):
        # if i == 8:
        #     continue
        fig, ax_mine_new = plt.subplots(figsize=(5, 5 * 0.618), constrained_layout=True)

        ax_mine_new = demonstration_possible_pout_range_in_wind_speed_bins_my_proposal_new(this_prod,
                                                                                           _ax=ax_mine_new)
        ax_mine_new.legend(loc='lower center', prop={'size': 10})
        ax_mine_new = adjust_lim_label_ticks(ax_mine_new, x_lim=WS_POUT_2D_PLOT_KWARGS['x_lim'])

        left, bottom, width, height = [0.13, 0.56, 0.235, 0.415]
        ax_in = fig.add_axes([left, bottom, width, height])
        ax_in = demonstration_possible_pout_range_in_wind_speed_bins_my_proposal_new(this_prod, _ax=ax_in)
        ax_in.set_xlim(2.9, 8.1)
        ax_in.set_ylim(-0.005, 0.47)
        ax_in.set_xticklabels('')
        ax_in.set_yticklabels('')
        ax_in.set_xlabel('')
        ax_in.set_ylabel('')
        ax_in.grid(False)
        ax_in.get_legend().remove()
        ax_mine_new.indicate_inset_zoom(ax_in)

    #     # ax_mine_old = demonstration_possible_pout_range_in_wind_speed_bins_my_proposal_old()
    #     # ax_sasa = sasa_algorithm_to_cal_possible_pout_range()
    #     # ax_sasa_pmaps = sasa_pmaps_to_cal_possible_pout_range()
    #     # ax_iec_standard = demonstration_iec_standard()
    ax_std = plot_wind_speed_std()
    # ax_sasa_high_resol = sasa_high_resol_check()
