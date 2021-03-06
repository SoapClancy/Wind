from PowerCurve_Class import *
from File_Management.load_save_Func import load_pkl_file
from File_Management.path_and_file_management_Func import try_to_find_file, try_to_find_folder_path_otherwise_make_one
from project_utils import *
from ErrorEvaluation_Class import DeterministicError
import pandas as pd
from prepare_datasets import load_raw_wt_from_txt_file_and_temperature_from_csv, load_croatia_data, \
    load_dalry_wind_farm_met_mast
from WT_WF_Class import WT, WF
from Wind_Class import cal_air_density, celsius_to_kelvin, Wind
from pathlib import Path
from File_Management.load_save_Func import *
from ConvenientDataType import UncertaintyDataFrame, IntOneDimensionNdarray, IntFloatConstructedOneDimensionNdarray
from Ploting.adjust_Func import *
from File_Management.path_and_file_management_Func import remove_win10_max_path_limit
from Filtering.OutlierAnalyser_Class import DataCategoryNameMapper
from typing import Tuple, List
from Filtering.OutlierAnalyser_Class import DataCategoryData
from parse import parse
from ErrorEvaluation_Class import DeterministicError
from locale import setlocale, LC_ALL
import matplotlib.ticker as ticker
from UnivariateAnalysis_Class import UnivariateGaussianMixtureModel, GaussianMixture
from Filtering.sklearn_novelty_and_outlier_detection_Func import use_isolation_forest
from Filtering.sklearn_novelty_and_outlier_detection_Func import DBSCAN, LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from Filtering.dbscan_custom_func import dbscan_custom
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple
from Writting.utils import *
import tensorflow as tf

tf.keras.backend.set_floatx('float32')

setlocale(LC_ALL, "en_US")
remove_win10_max_path_limit()

########################################################################################################################
"""
This paper uses Dalry WF (especially its WT2 for individual WT analysis) and Zelengrad WF recordings
"""
# %% The Mfr-PC with the lowest and highest air densities
MFR_PC_LIMIT = (PowerCurveByMfr(air_density='0.97'),
                PowerCurveByMfr(air_density='1.12', color='black', linestyle=':'),
                PowerCurveByMfr(air_density='1.27', color='lime', linestyle='--'))

# %% Darly wind turbines
DARLY_WIND_TURBINES = load_raw_wt_from_txt_file_and_temperature_from_csv()
# This paper essentially only have 2D analysis
for i_outer in range(DARLY_WIND_TURBINES.__len__()):
    DARLY_WIND_TURBINES[i_outer].predictor_names = ('wind speed',)
del i_outer
# Darly WT2 is of the most interest
DARLY_WIND_TURBINE_1 = DARLY_WIND_TURBINES[0]  # type: WT
DARLY_WIND_TURBINE_2 = DARLY_WIND_TURBINES[1]  # type: WT
# WT outlier results
DARLY_WIND_TURBINES_OUTLIERS = []
for this_wind_turbine_outer in DARLY_WIND_TURBINES:
    DARLY_WIND_TURBINES_OUTLIERS.append(load_pkl_file(this_wind_turbine_outer.default_results_saving_path["outlier"]))

DARLY_WIND_FARM_RAW, _ = WF.init_from_wind_turbine_instances(DARLY_WIND_TURBINES, obj_name='Dalry raw')  # type: WF
BIN_WIDTH = 1.
DARLY_WIND_FARM_RAW_MOB_PC = PowerCurveByMethodOfBins(
    DARLY_WIND_FARM_RAW['wind speed'].values,
    DARLY_WIND_FARM_RAW['active power output'].values / DARLY_WIND_FARM_RAW.rated_active_power_output,
    bin_width=BIN_WIDTH
)

# %% Zelengrad WF
ZELENGRAD_WIND_FARM = load_croatia_data('Zelengrad')['Zelengrad']  # type: WF
ZELENGRAD_WIND_FARM.number_of_wind_turbine = 14

# In this paper, only year 2015 to year 2019 data will be used
ZELENGRAD_WIND_FARM = ZELENGRAD_WIND_FARM[np.bitwise_and(
    ZELENGRAD_WIND_FARM.index >= datetime.datetime(year=2015, month=1, day=1),
    ZELENGRAD_WIND_FARM.index < datetime.datetime(year=2019, month=1, day=1)
)]
ZELENGRAD_WIND_FARM_RAW_MOB_PC = PowerCurveByMethodOfBins(
    ZELENGRAD_WIND_FARM['wind speed'].values,
    ZELENGRAD_WIND_FARM['active power output'].values / ZELENGRAD_WIND_FARM.rated_active_power_output,
    bin_width=BIN_WIDTH
)
# ZELENGRAD_WIND_FARM = ZELENGRAD_WIND_FARM.resample(
#     '10T',
#     resampler_obj_func_source_code="agg(lambda x: np.mean(x.values))"
# )  # type: WF

# %% Vratarusa WF
VRATARUSA_WIND_FARM = load_croatia_data('Vratarusa')['Vratarusa']  # type: WF
VRATARUSA_WIND_FARM.number_of_wind_turbine = 14


# VRATARUSA_WIND_FARM.resample(
#     '10T',
#     resampler_obj_func_source_code="agg(lambda x: np.mean(x.values))"
# )  # type: WF
########################################################################################################################


def fit_plot_and_summary_all_mfr_pc_in_all_density(mode: str):
    assert ((mode == 'fit') or (mode == 'summary')), "'mode' may be either 'fit' or 'summary'"
    mfr_pc = PowerCurveByMfr.init_all_instances_in_docs(cut_in_ws=3)
    error_df = pd.DataFrame(index=[x.air_density for x in mfr_pc],
                            columns=('a',
                                     'd',
                                     'b',
                                     'c',
                                     'g',
                                     'rmse',
                                     'mae',
                                     'mape'))
    direct_cmp_df = pd.DataFrame(index=[x.air_density for x in mfr_pc],
                                 columns=[str(x) for x in range(4, 26)] + [f'mfr-{x}' for x in range(4, 26)])
    mfr_pc_to_fit = []
    for i, this_mfr_pc in enumerate(mfr_pc):
        # Fitting results' saving path
        save_to_file_path = project_path_ / f"Data/Results/PowerCurve/Mfr_fittings_{this_mfr_pc.air_density}.pkl"
        this_mfr_pc_to_fit = PowerCurveFittedBy5PLF(wind_speed_recording=this_mfr_pc.mfr_ws,
                                                    active_power_output_recording=this_mfr_pc.mfr_p,
                                                    interp_for_high_resol=False)
        # If there are any fitting results in the saving path, then they can be used as initials
        if try_to_find_file(save_to_file_path):
            this_mfr_pc_to_fit.update_params(*load_pkl_file(save_to_file_path)[-1]['variable'])  # The last is the best
            params_init_scheme = 'self'
        else:
            params_init_scheme = 'average'
        # To fit
        if mode == 'fit':
            # Fit using GA
            this_mfr_pc_to_fit.fit(ga_algorithm_param={'max_num_iteration': 10000,
                                                       'max_iteration_without_improv': 1000},
                                   params_init_scheme=params_init_scheme,
                                   run_n_times=360,
                                   wind_speed=np.arange(0, 25, 0.1),
                                   focal_error=0.005,
                                   save_to_file_path=save_to_file_path)
        # To summarise
        else:
            # Error
            wind_speed = np.arange(4, 26)
            error_obj = DeterministicError(target=this_mfr_pc(wind_speed),
                                           model_output=this_mfr_pc_to_fit(wind_speed))
            error_df.iloc[i, :5] = this_mfr_pc_to_fit.params
            error_df['rmse'].iloc[i] = error_obj.cal_root_mean_square_error()
            error_df['mae'].iloc[i] = error_obj.cal_mean_absolute_error()
            error_df['mape'].iloc[i] = error_obj.cal_mean_absolute_percentage_error()
            direct_cmp_df.iloc[i, :wind_speed.__len__()] = this_mfr_pc_to_fit(wind_speed)
            direct_cmp_df.iloc[i, wind_speed.__len__():] = this_mfr_pc(wind_speed)
            if i == mfr_pc.__len__() - 1:
                error_df.to_csv('12Mfr-PC summary.csv')
                direct_cmp_df.T.to_csv('direct_cmp_df.csv')

            mfr_pc_to_fit.append(this_mfr_pc_to_fit)

    if mode == 'summary':
        ax = None
        mfr_wind_speed = np.arange(4, 26)
        wind_speed = np.arange(0, 25.1, 0.1)
        upper_layer = 2
        lower_layer = 1
        # 0.97
        ax = series(x=wind_speed, y=mfr_pc_to_fit[0](wind_speed), color='k', linestyle=':', ax=ax,
                    label='5PLF (' + r'$\rho$' + f'={mfr_pc[0].air_density} kg/m' + '$^3$)', zorder=lower_layer)
        ax = scatter(x=mfr_wind_speed, y=mfr_pc[0](mfr_wind_speed), color='r', marker='+', s=32, ax=ax,
                     rasterized=False,
                     label='Mfr-PC (' + r'$\rho$' + f'={mfr_pc[0].air_density} kg/m' + '$^3$)', zorder=upper_layer)

        # 1.12
        ax = series(x=wind_speed, y=mfr_pc_to_fit[5](wind_speed), color='darkorange', linestyle='--', ax=ax,
                    label='5PLF (' + r'$\rho$' + f'={mfr_pc[5].air_density} kg/m' + '$^3$)', zorder=lower_layer)
        ax = scatter(x=mfr_wind_speed, y=mfr_pc[5](mfr_wind_speed), color='g', marker='2', s=45, ax=ax,
                     rasterized=False,
                     label='Mfr-PC (' + r'$\rho$' + f'={mfr_pc[5].air_density} kg/m' + '$^3$)', zorder=upper_layer)

        # 1.27
        ax = series(x=wind_speed, y=mfr_pc_to_fit[11](wind_speed), color='b', linestyle='-', ax=ax,
                    label='5PLF (' + r'$\rho$' + f'={mfr_pc[11].air_density} kg/m' + '$^3$)', zorder=lower_layer)
        ax = scatter(x=mfr_wind_speed, y=mfr_pc[11](mfr_wind_speed), marker='o', s=16, ax=ax,
                     rasterized=False,
                     facecolors='none', edgecolors='magenta',
                     label='Mfr-PC (' + r'$\rho$' + f'={mfr_pc[11].air_density} kg/m' + '$^3$)', zorder=upper_layer,
                     y_lim=(-0.05, 1.05),
                     x_lim=(-0.05, 26.5),
                     x_label='Wind Speed [m/s]', y_label='Active Power Output [p.u.]',
                     save_file_='12Mfr-PC_fittings',
                     save_format='svg')


def cat_6_demo():
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
    # Only analyse WT_1
    wind_turbine_1 = copy.deepcopy(wind_turbines[0])  # type: WT
    wind_turbine_1['active power output'] /= wind_turbine_1.rated_active_power_output
    # Only analyse day 8.5 to 9
    wind_turbine_1 = wind_turbine_1.iloc[int(8.5 * 144):int(9 * 144)]
    # air density based PC
    air_density = cal_air_density(celsius_to_kelvin(wind_turbine_1['environmental temperature'].values),
                                  wind_turbine_1['relative humidity'].values / 100,
                                  wind_turbine_1['barometric pressure'].values * 100)
    mfr_pc = np.array(PowerCurveByMfr.init_multiple_instances(air_density=air_density))
    # manually set the index of CAT-VI
    cat_6_index = [2, 3, 11, 21]
    # 画初始垃圾图
    cat_6_index_indicator = np.isin(range(wind_turbine_1.__len__()), cat_6_index)
    ax = scatter(wind_turbine_1['wind speed'].values[~cat_6_index_indicator],
                 wind_turbine_1['active power output'].values[~cat_6_index_indicator],
                 color='b', s=4, label='Normal',
                 rasterized=False)
    ax = scatter(
        wind_turbine_1['wind speed'].values[cat_6_index_indicator],
        wind_turbine_1['active power output'].values[cat_6_index_indicator],
        ax=ax,
        facecolors='none', edgecolors='r', marker='*',
        s=32,
        zorder=10,
        label='CAT-VI',
        rasterized=False
    )
    ax = mfr_pc[2].plot(ax=ax)
    # range label
    ax = series(np.arange(50, 51, 0.1), np.arange(50, 51, 0.1), ax=ax, color='black',
                label='4.5' + r'$\sigma$' + ' %' + '\nrange',
                **{'x_label': 'Wind Speed [m/s]',
                   'y_label': 'Active Power Output [p.u.]'})
    # simulate
    wind_speed_range = wind_turbine_1['wind speed'].values[cat_6_index_indicator]
    wind_speed_std_range = wind_turbine_1['wind speed std.'].values[cat_6_index_indicator]
    simulation_resolution = 10
    simulation_traces = 10_000_000

    # simulation_return_percentiles = covert_to_str_one_dimensional_ndarray(np.arange(0, 100.5, 0.5), '0.1')
    @load_exist_pkl_file_otherwise_run_and_save(
        Path(project_path_) / 'Data/Results/transient_study/TSE2020/demo.pkl')
    def get_results():
        mode = 'cross sectional'
        # Initialise Wind instance
        nonlocal cat_6_index
        simulated_pout_list = []
        raw_list = []
        for this_ws, this_ws_std, this_pc in zip(wind_speed_range, wind_speed_std_range, mfr_pc[cat_6_index]):
            wind = Wind(this_ws, this_ws_std)
            high_resol_wind = wind.simulate_transient_wind_speed_time_series(
                resolution=simulation_resolution,
                traces_number_for_each_recording=simulation_traces,
                mode=mode,
                sigma_func=wind.learn_transition_by_looking_at_actual_high_resol()
            )
            #
            template = UncertaintyDataFrame.init_from_template(
                columns_number=len(high_resol_wind),
                percentiles=[50.]
            )
            _simulated_pout = this_pc.cal_with_hysteresis_control_using_high_resol_wind(
                high_resol_wind,
                mode=mode,
                return_percentiles=template,
            )
            raw = this_pc.cal_with_hysteresis_control_using_high_resol_wind(high_resol_wind,
                                                                            mode=mode)
            template[:] = _simulated_pout.values
            simulated_pout_list.append(template)
            raw_list.append(raw)
        return simulated_pout_list, raw_list

    simulated_pout = get_results()[0]  # type: List[UncertaintyDataFrame,...]
    # 画error bar
    for i in range(cat_6_index.__len__()):
        ax = scatter(np.array([wind_speed_range[i]] * 2), simulated_pout[i](by_sigma=1).values, ax=ax,
                     label='1' + r'$\sigma$' + ' %' if i == 0 else None, rasterized=False,
                     color='grey', marker='+', s=100, zorder=12)
        ax = scatter(np.array([wind_speed_range[i]] * 2), simulated_pout[i](by_sigma=2).values, ax=ax,
                     label='2' + r'$\sigma$' + ' %' if i == 0 else None, rasterized=False,
                     color='fuchsia', marker='x', s=32, zorder=12)
        ax = scatter(np.array([wind_speed_range[i]] * 2), simulated_pout[i](by_sigma=3).values, ax=ax,
                     label='3' + r'$\sigma$' + ' %' if i == 0 else None, rasterized=False,
                     color='royalblue', marker='s', s=16, zorder=12)

        ax = scatter(wind_speed_range[i], simulated_pout[i].loc['mean'].values, ax=ax,
                     label='Mean' if i == 0 else None, rasterized=False,
                     color=(0, 1, 0), marker='1', s=64, zorder=12)

        ax = scatter(wind_speed_range[i], simulated_pout[i].loc['mean'].values, ax=ax,
                     label='Median' if i == 0 else None, rasterized=False,
                     color='orange', marker='2', s=64, zorder=12,
                     x_label='Wind Speed [m/s]',
                     y_label='Active Power Output [p.u.]')

        # ax.errorbar(wind_speed_range[i],
        #             simulated_pout.loc['mean'][i],
        #             yerr=np.array([[simulated_pout.loc['mean'][i] - simulated_pout(by_sigma=4.5).iloc[0, i]],
        #                            [simulated_pout(by_sigma=4.5).iloc[1, i] - simulated_pout.loc['mean'][i]]]),
        #             color='k', fmt='-', markersize=10)
    plt.gca().legend(ncol=1, loc='upper left', prop={'size': 10})

    return ax


def cat_6_demo_time_series():
    this_wind_turbine = copy.deepcopy(DARLY_WIND_TURBINES[0])  # type: WT
    this_wind_turbine['active power output'] /= this_wind_turbine.rated_active_power_output
    this_wind_turbine.rated_active_power_output = 1.
    mfr_pc = np.array(PowerCurveByMfr.init_multiple_instances(
        air_density=this_wind_turbine.update_air_density_to_last_column()))

    time_window = (datetime.datetime(2007, 1, 9, 12), datetime.datetime(2007, 1, 9, 20, 30))
    time_window_mask = np.bitwise_and(this_wind_turbine.index >= time_window[0],
                                      this_wind_turbine.index < time_window[1])
    index_mapper = [2, 3, 11, 21]

    # %% 画图代码！注释掉是为了调试后面的代码！不要删!
    # time_x = this_wind_turbine.index[time_window_mask]
    # ax = series(x=time_x, y=this_wind_turbine.loc[time_window_mask, 'wind speed'].values,
    # figure_size=(5, 3.3 * 0.618),
    #             x_axis_format='%H', x_label='Time of a Day [Hour]',
    #             marker='*', markersize=6, color='royalblue', linestyle='-',
    #             y_lim=(-1, 19), y_label='Wind Speed [m/s]', label='Wind speed')
    #
    # ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    # ax2.set_ylabel('Active Power Output [p.u.]', fontdict={'size': 10})  # we already handled the x-label with ax1
    # series(x=time_x, y=this_wind_turbine.loc[time_window_mask, 'active power output'].values, ax=ax2,
    #        x_axis_format='%H', marker='o', markersize=3, color='green', linestyle='--', label='Power output')
    # plt.grid(False)
    # # ask matplotlib for the plotted objects and their labels
    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax2.legend(lines + lines2, labels + labels2, loc=0, prop={'size': 10})

    # 无聊的compare
    sim_results = load_pkl_file(Path(project_path_) / 'Data/Results/transient_study/TSE2020/demo.pkl')

    # IF
    outlier_obj = load_pkl_file(this_wind_turbine.default_results_saving_path["outlier"])['DataCategoryData obj']
    consider_mask = outlier_obj(("CAT-IV.a", "CAT-IV.b", "others"))
    if_obj = use_isolation_forest(this_wind_turbine.loc[consider_mask, ['wind speed', 'active power output']].values,
                                  return_obj=True)
    if_score = -if_obj.score_samples(
        this_wind_turbine.loc[time_window_mask, ['wind speed', 'active power output']].values
    )
    print(if_score[index_mapper])

    # IEC
    iec_mean = []
    for i in range(index_mapper.__len__()):
        wind_distribution = GaussianMixture()
        wind_distribution.means_ = np.array(
            this_wind_turbine.loc[time_window_mask, 'wind speed'].values[index_mapper[i]]
        ).reshape((1, 1))
        wind_distribution.covariances_ = pow(
            this_wind_turbine.loc[time_window_mask, 'wind speed std.'].values[index_mapper[i]], 2
        ).reshape((1, 1, 1))
        wind_distribution.weights_ = [1]
        # leverage UnivariateGaussianMixtureModel to do boundary-specified sampling
        wind_distribution = UnivariateGaussianMixtureModel(wind_distribution,
                                                           theoretic_min_value=0)
        samples = wind_distribution.sample(int(10_000_000))
        pout = mfr_pc[time_window_mask][index_mapper[i]](samples)
        iec_mean.append(np.mean(pout))

    bins = np.arange(0, 1.005, 0.005)

    this_index = 0
    for this_index in range(index_mapper.__len__()):
        sim = sim_results[1][this_index].flatten()
        ax = hist(sim, density=True, bins=bins, figure_size=(3.6, 3.6 * 0.618),
                  label='Simulation', color='gold', edgecolor='gold',
                  x_label='Active Power Output [p.u.]', y_label='Probability Density', x_lim=(0.26, 1.0001))
        y_lim = ax.get_ylim()
        # actual
        ax = vlines(this_wind_turbine.loc[time_window_mask, 'active power output'].values[index_mapper[this_index]],
                    color='royalblue', label='Actual recording', linestyles='-',
                    ymin=y_lim[0], ymax=y_lim[-1], ax=ax, y_lim=y_lim)
        # mean and media
        ax = vlines(float(sim_results[0][this_index].loc['mean'].values), color='black', label='Simulation mean',
                    linestyles=':', ymin=y_lim[0], ymax=y_lim[-1], ax=ax, y_lim=y_lim)
        ax = vlines(sim_results[0][this_index](50).values[0, 0], color='red', label='Simulation median',
                    linestyles='--', ymin=y_lim[0], ymax=y_lim[-1], ax=ax, y_lim=y_lim)
        # 6.68% - 93.32%
        ax = vlines(np.percentile(sim_results[1][this_index].flatten(),
                                  UncertaintyDataFrame.infer_percentile_boundaries_by_sigma(1.5)[0]),
                    color='green', label='Simulation 6.68% - 93.32%', linestyles='-.',
                    ymin=y_lim[0], ymax=y_lim[-1], ax=ax, y_lim=y_lim)
        ax = vlines(np.percentile(sim_results[1][this_index].flatten(),
                                  UncertaintyDataFrame.infer_percentile_boundaries_by_sigma(1.5)[1]),
                    color='green', linestyles='-.',
                    ymin=y_lim[0], ymax=y_lim[-1], ax=ax, y_lim=y_lim)
        # IEC standard
        ax = vlines(iec_mean[this_index],
                    color='cyan', label='Expected value by [11, 42]', linestyles='-',
                    ymin=y_lim[0], ymax=y_lim[-1], ax=ax, y_lim=y_lim, legend_ncol=3)
        #
        if this_index == (index_mapper.__len__() - 1):
            ax.set_yticks([0, 5, 10])
            ax.set_yticklabels(['0', '5', '10'])
        adjust_legend_in_ax(ax, protocol='Outside center right', ncol=3)


def plot_raw_data_for_outlier_demo():
    """
    This function is to provide the very first figures in the paper, which shows that there seem to be outlier
    :return:
    """
    lof_and_dbscan_results = lof_and_dbscan_power_curve()
    for i, to_plot_obj in enumerate((DARLY_WIND_TURBINE_1, DARLY_WIND_FARM_RAW, ZELENGRAD_WIND_FARM)):
        ax = to_plot_obj.plot(plot_mfr=MFR_PC_LIMIT, mfr_mode='discrete', mfr_kwargs=MFR_KWARGS, plot_scatter_pc=True)
        ax = lof_and_dbscan_results[to_plot_obj.obj_name + '_DBSCAN'].plot(
            ax=ax,
            label='DBSCAN PC', linestyle='--', color='dimgray', plot_recording=False)
        ax = lof_and_dbscan_results[to_plot_obj.obj_name + '_LOF'].plot(
            ax=ax,
            label='LOF PC', linestyle='-.', color='darkorange', plot_recording=False)
        adjust_legend_in_ax(ax, loc='upper left')
        if to_plot_obj is ZELENGRAD_WIND_FARM:
            adjust_lim_label_ticks(ax=ax, x_lim=(-0.5, 33.5))


def lof_and_dbscan_power_curve():
    @load_exist_pkl_file_otherwise_run_and_save(project_path_ / r"Data\Results\Filtering\TSE2020_dbscan_lof\pcs.pkl")
    # @update_exist_pkl_file_otherwise_run_and_save(
    #     project_path_ / r"Data\Results\Filtering\TSE2020_dbscan_lof\pcs.pkl")
    def func(existing_file=None):
        power_curves = existing_file or {}
        # 两个WT，两个WF
        for this_obj in (DARLY_WIND_TURBINE_1, DARLY_WIND_TURBINE_2, DARLY_WIND_FARM_RAW, ZELENGRAD_WIND_FARM):
            this_ws_pout_data = copy.deepcopy(this_obj)
            this_ws_pout_data['active power output'] /= this_obj.rated_active_power_output
            this_ws_pout_data = this_ws_pout_data[['wind speed', 'active power output']].values
            this_ws_pout_data = this_ws_pout_data[~np.any(np.isnan(this_ws_pout_data), axis=1)]
            min_max_scaler = MinMaxScaler()
            min_max_scaler.fit(this_ws_pout_data)
            this_ws_pout_data = min_max_scaler.transform(this_ws_pout_data)
            # %% DBSCAN
            if this_obj.obj_name != "Zelengrad WF":
                dbscan_clustering = DBSCAN(eps=0.025, min_samples=5).fit(this_ws_pout_data)
                dbscan_clustering_results = dbscan_clustering.labels_
            else:
                # %% 应该要用ELKI的
                # dbscan_clustering = DBSCAN(eps=0.0025, min_samples=5).fit(this_ws_pout_data)
                # dbscan_clustering_results = dbscan_clustering.labels_
                elki_results_path = project_path_ / r"Data\Results\Filtering\TSE2020_dbscan_lof\ELKI\cluster_0.txt"
                elki_results = pd.read_table(elki_results_path, skiprows=8, sep=r'\s+|=', header=None)
                elki_results = elki_results.values[:, 1]
                dbscan_clustering_results = np.full(this_ws_pout_data.shape[0], fill_value=-1)
                dbscan_clustering_results[(elki_results.astype(int) - 1).tolist()] = 0
            #
            # %% LOF
            lof_clustering_results = LocalOutlierFactor(n_neighbors=300).fit_predict(this_ws_pout_data)

            this_ws_pout_data = min_max_scaler.inverse_transform(this_ws_pout_data)
            power_curves.setdefault(
                this_obj.obj_name + '_DBSCAN',
                PowerCurveByMethodOfBins(
                    wind_speed_recording=this_ws_pout_data[dbscan_clustering_results == 0, 0],
                    active_power_output_recording=this_ws_pout_data[dbscan_clustering_results == 0, 1],
                    bin_width=BIN_WIDTH
                )
            )
            power_curves.setdefault(
                this_obj.obj_name + '_LOF',
                PowerCurveByMethodOfBins(
                    wind_speed_recording=this_ws_pout_data[lof_clustering_results != -1, 0],
                    active_power_output_recording=this_ws_pout_data[lof_clustering_results != -1, 1],
                    bin_width=BIN_WIDTH
                )
            )
            """DEBUG"""
            # ax = None
            # for this_label in np.unique(dbscan_clustering.labels_):
            #     ax = scatter(*this_ws_pout_data[this_label == dbscan_clustering.labels_].T, ax=ax,
            #                  label=str(this_label), legend_ncol=2)
            # ax = scatter(*this_ws_pout_data[lof_clustering_results == -1].T, label='-1')
            # scatter(*this_ws_pout_data[lof_clustering_results != -1].T, label='0', ax=ax)
        return power_curves

    return func()


def individual_wind_turbine_outliers_outlier_detector(power_curve_model_task: str,
                                                      wt_index: tuple = (),
                                                      write_to_a_docx: bool = False):
    assert power_curve_model_task in ('fit', 'load')
    document = docx_document_template_to_collect_figures() if write_to_a_docx else None

    for i, this_wt in enumerate(load_raw_wt_from_txt_file_and_temperature_from_csv()):
        if i not in wt_index:
            continue
        this_wt.predictor_names = ('wind speed',)
        # this_wt = this_wt.loc[:datetime.datetime(2007, 1, 10)]
        # %% Detect outlier
        print(f"{power_curve_model_task} {this_wt} outliers")
        obj = this_wt.outlier_detector(
            prior_sim_knowledge_path=project_path_ / r"Data\Results\Filtering\Darly WTs prior_simulation_dict\sim.pkl"
        )
        # this_wt[obj("normal")].plot(plot_scatter_pc=True)
        # this_wt.outlier_plot()
        # this_wt.outlier_plot(plot_individual=True)
        # this_wt.outlier_report()

        # %% Fit PC on the filtered data
        fitted_pc_obj = this_wt.select_data_and_get_power_curve_model(power_curve_model_task)
        # print(this_wt)
        # print(fitted_pc_obj)

        outlier = load_pkl_file(this_wt.default_results_saving_path["outlier"])['DataCategoryData obj']
        if not write_to_a_docx:
            if power_curve_model_task == 'load':
                ax = this_wt[outlier('normal')].plot(plot_mfr=MFR_PC_LIMIT, mfr_mode='discrete', mfr_kwargs=MFR_KWARGS,
                                                     plot_scatter_pc=True)
                fitted_pc_obj.plot(plot_recording=False, ax=ax, label="6P-PC")

            this_wt.outlier_plot()
            # this_wt.outlier_plot(plot_individual=True)
            this_wt.outlier_report()

            # %% Calculate error
            ws = np.arange(0, 28.5 + BIN_WIDTH, BIN_WIDTH)
            error_obj = DeterministicError(target=fitted_pc_obj.corresponding_mob_pc_obj(ws),
                                           model_output=fitted_pc_obj(ws))
        # print(f"MAE = {error_obj.cal_mean_absolute_error()}")
        # print(f"RMSE = {error_obj.cal_root_mean_square_error()}")

        if write_to_a_docx:
            ax_1_raw = this_wt.plot(plot_mfr=MFR_PC_LIMIT, mfr_mode='discrete', mfr_kwargs=MFR_KWARGS,
                                    plot_scatter_pc=True, save_to_buffer=True)
            ax_2_classified = this_wt.outlier_plot(save_to_buffer=True)
            # ax_3_normal = this_wt[outlier('normal')].plot(plot_mfr=MFR_PC_LIMIT, mfr_mode='discrete',
            #                                               mfr_kwargs=MFR_KWARGS, plot_scatter_pc=True,
            #                                               save_to_buffer=True)
            ax = this_wt[outlier('normal')].plot(plot_mfr=MFR_PC_LIMIT, mfr_mode='discrete', mfr_kwargs=MFR_KWARGS,
                                                 plot_scatter_pc=True)
            ax_3_normal = fitted_pc_obj.plot(plot_recording=False, ax=ax, label="6P-PC", save_to_buffer=True)

            ax_4_report, ax_5_report = this_wt.outlier_report(save_to_buffer=True)
            document.add_heading(this_wt.__str__().replace("Darly", "Dalry"), level=1)
            to_plot_list_buffer_list = [ax_1_raw, ax_2_classified, ax_3_normal, ax_4_report, ax_5_report]
            p = document.add_paragraph()
            for j, this_to_plot in enumerate(to_plot_list_buffer_list):
                if (j % 2 == 0) and (j != 0):
                    p = document.add_paragraph()
                p.add_run().add_picture(this_to_plot, width=Cm(7))
            document.add_page_break()
    if write_to_a_docx:
        document.save(f'.\\individual_wind_turbine_outliers_outlier_detector.docx')


def wind_turbine_level_outlier_results_demo():
    all_wind_turbine_results = []
    for i, this_wt in enumerate(load_raw_wt_from_txt_file_and_temperature_from_csv()):
        this_wt.plot(plot_mfr=MFR_PC_LIMIT, plot_scatter_pc=True, title=f"WT{i + 1}")
        this_wt.outlier_plot(title=f"WT{i + 1}")
        all_wind_turbine_results.append(this_wt.outlier_detector().report())
    all_wind_turbine_results = pd.concat([x['number'] for x in all_wind_turbine_results], axis=1)
    all_wind_turbine_results = all_wind_turbine_results.sum(axis=1)
    all_wind_turbine_results = all_wind_turbine_results / sum(all_wind_turbine_results) * 100
    all_wind_turbine_results = all_wind_turbine_results.rename({
        'missing': 'Missing',
        'normal': 'Normal'
    })
    ax = bar(all_wind_turbine_results.index,
             all_wind_turbine_results.values,
             y_label="Recording Percentage [%]",
             autolabel_format="{:.2f}",
             x_ticks_rotation=45, y_lim=(-1, 85))
    return ax


def darly_wind_farm_operating_regime() -> Tuple[WF, pd.DataFrame, DataCategoryData]:
    # %% Select by wind turbines outliers
    # In the WF analyse, only 'CAT-I.a', 'CAT-I.b', 'CAT-II', 'CAT-IV.a' and 'others' WT recordings are used
    # wind_turbines_with_selected_recordings = []
    wind_turbine_instances_data_category = []
    for i in range(DARLY_WIND_TURBINES_OUTLIERS.__len__()):
        # %% rename to adapt to WF-level analysis, and the rules below are used:
        this_outlier = DARLY_WIND_TURBINES_OUTLIERS[i]['DataCategoryData obj']
        this_outlier.rename(
            mapper={'CAT-III': 'shutdown',
                    'CAT-I': 'curtailed',
                    'normal': 'operating',
                    'CAT-II': 'nan', 'CAT-IV': 'nan', 'missing': 'nan'},
            new_name_mapper=DataCategoryNameMapper(
                [["shutdown", "shutdown", 3, "originally be CAT-III"],
                 ["curtailed", "curtailed", 2, "originally be CAT-I"],
                 ["operating", "operating", 1, "originally be normal"],
                 ["nan", "nan", -1, "originally be either CAT-II or CAT-IV"]],
                columns=['long name', 'abbreviation', 'code', 'description']
            )
        )
        wind_turbine_instances_data_category.append(this_outlier)

    darly_wind_farm, darly_wind_farm_total_curtailment_amount = WF.init_from_wind_turbine_instances(
        DARLY_WIND_TURBINES,
        obj_name='Dalry',
        wind_turbine_instances_data_category=wind_turbine_instances_data_category
    )
    del i, this_outlier

    # %% Infer WF data category by looking at individual WT outlier information (Sequence[DataCategoryData])
    operating_regime = WF.infer_operating_regime_from_wind_turbine_instances_data_category(
        wind_turbine_instances_data_category
    )

    # %% Plot
    # DARLY_WIND_FARM_RAW.plot(plot_mfr=MFR_PC_LIMIT)
    # darly_wind_farm.plot(plot_mfr=MFR_PC_LIMIT)
    # darly_wind_farm[operating_regime('S1')].plot(plot_mfr=MFR_PC_LIMIT, plot_scatter_pc=True)
    # darly_wind_farm.plot(operating_regime=operating_regime)

    # %% report
    # operating_regime.report(
    #     sorted_kwargs={'key': lambda x: "0" + x[1:] if x[1:].__len__() < 2 else x[1:]},
    #     report_pd_to_csv_file_path=darly_wind_farm.default_results_saving_path['operating regime']
    # )
    return darly_wind_farm, darly_wind_farm_total_curtailment_amount, operating_regime


def fit_or_analyse_darly_wind_farm_power_curve_model_with_known_wind_turbines(task: str):
    assert task in ('time series check', '2D plot check', 'fit')

    darly_wind_farm, darly_wind_farm_total_curtailment_amount, operating_regime = darly_wind_farm_operating_regime()
    # p.u. is necessary, which can reduce the fitting burden
    darly_wind_farm.loc[:, 'active power output'] /= darly_wind_farm.rated_active_power_output
    darly_wind_farm_total_curtailment_amount /= darly_wind_farm.rated_active_power_output
    darly_wind_farm.rated_active_power_output = 1.  # per-unitized
    # %% fully operating regime fitting
    fitting_file_path = darly_wind_farm.default_results_saving_path['fully operating regime power curve']
    fully_operating_8_param_pc = PowerCurveFittedBy8PLF(
        darly_wind_farm.loc[operating_regime('S1'), 'wind speed'],
        darly_wind_farm.loc[operating_regime('S1'), 'active power output'],
        bin_width=BIN_WIDTH,
        interp_for_high_resol=False
    )

    # If there are any fitting results in the saving path, then they can be used as initials
    if try_to_find_file(fitting_file_path):
        current_best = load_pkl_file(fitting_file_path)[-1]['variable']
        ################################################################################################################
        current_best[:2] = [1, 0]
        fully_operating_8_param_pc.params_constraints = {'a': [1 - float_eps, 1 + float_eps],
                                                         'd': [-float_eps, float_eps]}
        ################################################################################################################
        fully_operating_8_param_pc.update_params(*current_best)  # The last the best
        params_init_scheme = 'self'
    else:
        params_init_scheme = 'average'
    if task == 'fit':
        # Fit using GA
        fully_operating_8_param_pc.fit(ga_algorithm_param={'max_num_iteration': 10000,
                                                           'max_iteration_without_improv': 1000},
                                       params_init_scheme=params_init_scheme,
                                       run_n_times=500,
                                       wind_speed=np.arange(0, 29.5, 0.1),
                                       focal_error=0.0025,
                                       save_to_file_path=fitting_file_path)
    else:
        # %% Plot GA fitting convergence and GA results
        # print(fully_operating_8_param_pc)
        # series(np.array(list(map(lambda x: x['function'], load_pkl_file(fitting_file_path)))),
        #        title='GA Fitting Convergence')
        # fully_operating_8_param_pc.plot(plot_recording=True)

        # %% Fully operating to partly operating linear scaling plus bias analysis
        darly_wind_farm_equivalent_power_curve_model = EquivalentWindFarmPowerCurve.init_from_8p_pc_obj(
            fully_operating_8_param_pc,
            total_wind_turbine_number=6,
            wind_speed_recording=darly_wind_farm['wind speed'].values,
            active_power_output_recording=darly_wind_farm['active power output'].values,
            index=darly_wind_farm.index
        )
        print(f"darly_wind_farm_equivalent_power_curve_model = {darly_wind_farm_equivalent_power_curve_model}")
        if task == '2D plot check':
            darly_wind_farm_equivalent_power_curve_model.assess_fit_2d_scatters(
                operating_regime=operating_regime,
                total_curtailment_amount=darly_wind_farm_total_curtailment_amount,
                original_scatters_pc=DARLY_WIND_FARM_RAW_MOB_PC
            )

        else:
            darly_wind_farm_equivalent_power_curve_model.assess_fit_time_series(
                operating_regime=operating_regime,
                total_curtailment_amount=darly_wind_farm_total_curtailment_amount.values,
                original_scatters_pc=DARLY_WIND_FARM_RAW_MOB_PC
            )


def fit_or_analyse_darly_wind_farm_power_curve_model_without_known_wind_turbines(task: str):
    assert task in ('time series check', '2D plot check', 'fit')

    # Make sure the global variable DARLY_WIND_FARM_RAW will not be changed accidentally,
    # Also, change the obj_name attribute, since they are now treated as single-sensor measurements
    darly_wind_farm_single = copy.deepcopy(DARLY_WIND_FARM_RAW)
    darly_wind_farm_single.obj_name = 'Darly single'

    # %% Do the outlier analyse
    outlier = darly_wind_farm_single.outlier_detector('30T', )
    # Set CAT-III outliers to be nan
    darly_wind_farm_single[outlier(('CAT-III', 'missing'))] = np.nan

    # results = darly_wind_farm_single.operating_regime_detector_initial_guess()
    # %% Try to find the operating regime
    if task == 'fit':
        darly_wind_farm_single.operating_regime_detector('fit')
    # %% Check results
    else:
        wf_pc_obj, operating_regime = darly_wind_farm_single.operating_regime_detector('load')
        wf_pc_obj.bin_width = BIN_WIDTH
        if task == '2D plot check':
            darly_wind_farm_single.plot(operating_regime=operating_regime, not_show_color_bar=True)
            wf_pc_obj.assess_fit_2d_scatters(
                operating_regime=operating_regime,
                original_scatters_pc=DARLY_WIND_FARM_RAW_MOB_PC
            )
        else:
            # lof_and_dbscan_results = lof_and_dbscan_power_curve()
            wf_pc_obj.assess_fit_time_series(
                operating_regime=operating_regime,
                original_scatters_pc=DARLY_WIND_FARM_RAW_MOB_PC
                # original_scatters_pc=lof_and_dbscan_results['Dalry raw_DBSCAN']
                # original_scatters_pc=lof_and_dbscan_results['Dalry raw_LOF']
            )


def fit_or_analyse_zelengrad_wind_farm_power_curve_model(task: str):
    assert task in ('time series check', '2D plot check', 'fit')

    # Make sure the global variable DARLY_WIND_FARM_RAW will not be changed accidentally,
    # Also, change the obj_name attribute, since they are now treated as single-sensor measurements
    zelengrad_wind_farm = copy.deepcopy(ZELENGRAD_WIND_FARM)
    zelengrad_wind_farm.obj_name = 'Zelengrad single'
    zelengrad_wind_farm['active power output'] /= zelengrad_wind_farm.rated_active_power_output
    zelengrad_wind_farm.rated_active_power_output = 1

    # %% Do the outlier analyse
    outlier = zelengrad_wind_farm.outlier_detector(
        '6T',
        constant_error={'wind speed': 0.001},
        extra_boundary_rules=[
            {'wind speed': (0, 2), 'active power output': (0.02, 1)},
            {'wind speed': (3.064, 3.065), 'active power output': (0.0075, 1)},
            {'wind speed': (6.1, 6.2), 'active power output': (0.28, 1)},
            {'wind speed': (5.0, 5.1), 'active power output': (0.375, 1)},
            {'wind speed': (10.3, 10.4), 'active power output': (0.001, 0.04)}
        ]
    )
    # Set CAT-III outliers to be nan
    zelengrad_wind_farm[outlier(('CAT-III', 'missing'))] = np.nan

    # results = darly_wind_farm_single.operating_regime_detector_initial_guess()
    # %% Try to find the operating regime
    if task == 'fit':
        zelengrad_wind_farm.operating_regime_detector('fit')
    else:
        wf_pc_obj, operating_regime = zelengrad_wind_farm.operating_regime_detector('load')
        wf_pc_obj.bin_width = BIN_WIDTH
        if task == '2D plot check':
            zelengrad_wind_farm[operating_regime('S1')].plot()
            zelengrad_wind_farm.plot(operating_regime=operating_regime, not_show_color_bar=False)
            wf_pc_obj.assess_fit_2d_scatters(
                operating_regime=operating_regime,
                original_scatters_pc=ZELENGRAD_WIND_FARM_RAW_MOB_PC
            )
            operating_regime.report()

        else:
            # lof_and_dbscan_results = lof_and_dbscan_power_curve()
            wf_pc_obj.assess_fit_time_series(
                operating_regime=operating_regime,
                original_scatters_pc=ZELENGRAD_WIND_FARM_RAW_MOB_PC,
                # original_scatters_pc=lof_and_dbscan_results['Zelengrad WF_DBSCAN']
                # original_scatters_pc=lof_and_dbscan_results['Zelengrad WF_LOF']
            )


def fit_or_analyse_vratarusa_wind_farm_power_curve_model(task: str):
    assert task in ('time series check', '2D plot check', 'fit')

    # Make sure the global variable DARLY_WIND_FARM_RAW will not be changed accidentally,
    # Also, change the obj_name attribute, since they are now treated as single-sensor measurements
    vratarusa_wind_farm = copy.deepcopy(VRATARUSA_WIND_FARM)
    vratarusa_wind_farm.obj_name = 'Vratarusa single'

    # %% Do the outlier analyse
    outlier = vratarusa_wind_farm.outlier_detector('60T', constant_error={'wind speed': 0.01})
    # Set CAT-III outliers to be nan
    vratarusa_wind_farm[outlier(('CAT-III', 'missing'))] = np.nan

    # results = darly_wind_farm_single.operating_regime_detector_initial_guess()
    # %% Try to find the operating regime
    if task == 'fit':
        vratarusa_wind_farm.operating_regime_detector('fit')


def fit_or_analyse_zelengrad_10min_power_curve_model(task: str):
    assert task in ('time series check', '2D plot check', 'fit')

    # Make sure the global variable DARLY_WIND_FARM_RAW will not be changed accidentally,
    # Also, change the obj_name attribute, since they are now treated as single-sensor measurements
    # Aggregate to 10 min
    zelengrad_wind_farm_10min = ZELENGRAD_WIND_FARM.resample(
        '10T',
        resampler_obj_func_source_code="agg(lambda x: np.mean(x.values))"
    )  # type: WF
    zelengrad_wind_farm_10min.obj_name = 'Zelengrad single 10min-resol'
    zelengrad_wind_farm_10min['active power output'] /= zelengrad_wind_farm_10min.rated_active_power_output
    zelengrad_wind_farm_10min.rated_active_power_output = 1
    zelengrad_wind_farm_10min.plot()


def fit_or_analysis_wind_turbine_power_curve_model(wind_turbine_obj: WT, task: str):
    assert task in ('time series check', '2D plot check', 'fit')

    if task == 'fit':
        wind_turbine_obj.select_data_and_get_power_curve_model(task, bin_width=BIN_WIDTH)
    else:
        pc_obj = wind_turbine_obj.select_data_and_get_power_curve_model('load', bin_width=BIN_WIDTH)

        wind_turbine_obj = copy.deepcopy(wind_turbine_obj)

        outlier = load_pkl_file(wind_turbine_obj.default_results_saving_path["outlier"])['DataCategoryData obj']
        selected_data_mask = outlier(("others", "CAT-I.a", "CAT-IV.a"))
        wind_turbine_obj[:] = wind_turbine_obj[selected_data_mask]
        ax = wind_turbine_obj.plot(plot_mfr=MFR_PC_LIMIT, mfr_mode='discrete', mfr_kwargs=MFR_KWARGS,
                                   plot_scatter_pc=True)
        ax = pc_obj.plot(plot_recording=False, ax=ax, label="6PL PC")
        print(pc_obj)

        wind_turbine_obj.plot()


def individual_wind_turbine_outliers_outlier_detector_sasa_method(wind_turbine_obj: WT):
    if try_to_find_file(wind_turbine_obj.default_results_saving_path['outlier']):
        raise FileExistsError
    exec(r"from Energies_Review_paper_2020 import sasa_combine_upper_and_lower", globals())
    sasa_combine_upper_and_lower = globals()['sasa_combine_upper_and_lower']  # type: Callable
    sasa_combine_uct = sasa_combine_upper_and_lower(wind_turbine_obj=wind_turbine_obj)
    wind_turbine_obj.outlier_detector(sasa_combine_uct=sasa_combine_uct)
    wind_turbine_obj.outlier_plot()


def low_wind_speed_cat_i_a_outlier_number(wind_turbine_obj: WT):
    """
    This function is to check weired occurs of CAT-I.a outliers below 20 m/s
    Its results show the phenomenon is because of a bug in original definition.
    This function can be ignored for now.
    :param wind_turbine_obj:
    :return:
    """
    outlier = load_pkl_file(wind_turbine_obj.default_results_saving_path["outlier"])['DataCategoryData obj']
    mask = np.bitwise_and(outlier("CAT-I.a"), wind_turbine_obj['wind speed'].values < 20)
    # %% Time series plot
    masked_index = wind_turbine_obj.index.values[mask]
    masked_index_view = np.reshape(masked_index, (-1, 1))

    expanding_length = np.timedelta64(600, "m")
    to_plot_window_indices_list = []
    window_left_index = masked_index[0]
    for i in range(1, masked_index.__len__()):
        if (masked_index[i] - masked_index[i - 1]) == np.timedelta64(10, "m"):
            window_right_index = masked_index[i]
        else:
            to_plot_window_indices_list.append((window_left_index - expanding_length,
                                                masked_index[i - 1] + expanding_length))
            window_left_index = masked_index[i]
            window_right_index = masked_index[min(i + 1, masked_index.__len__() - 1)]
        if i == (masked_index.__len__() - 1):
            if window_left_index == window_right_index:
                to_plot_window_indices_list.append((window_left_index - expanding_length,
                                                    window_left_index + expanding_length))
            else:
                to_plot_window_indices_list.append((window_left_index - expanding_length,
                                                    masked_index[i] + expanding_length))
    # 画图
    for this_window_index in to_plot_window_indices_list:
        this_window_mask = np.bitwise_and(wind_turbine_obj.index >= this_window_index[0],
                                          wind_turbine_obj.index <= this_window_index[1])
        _, ax = wind_turbine_obj.twin_time_series_plot(time_window_mask=this_window_mask,
                                                       wind_speed_y_lim=(-0.05, 30.55),
                                                       x_axis_format="%m-%d %H:%M",
                                                       x_label="Date and Time")
        ax = vlines(this_window_index[0] + expanding_length, ax=ax, color='r')
        ax = vlines(this_window_index[1] - expanding_length, ax=ax, color='r')


def migrate_from_old_wind_turbine_outlier_detection():
    """
    老的定义太复杂，所以在最终交稿的时候，把定义全部更新了。
    :return:
    """
    confirm = input("确认更新WT-Level的outlier结果的定义？[Y/n]")
    assert confirm == 'Y'
    for i, this_wt in enumerate(load_raw_wt_from_txt_file_and_temperature_from_csv()):
        this_outlier = load_pkl_file(this_wt.default_results_saving_path['outlier'])
        # Refactoring this_outlier['raw_ndarray_data']
        old_cat_i_mask = this_outlier['raw_ndarray_data'] == 'CAT-I'
        old_cat_ii_mask = this_outlier['raw_ndarray_data'] == 'CAT-II'
        old_cat_iii_mask = this_outlier['raw_ndarray_data'] == 'CAT-III'
        this_outlier['raw_ndarray_data'][old_cat_i_mask] = 'CAT-III'
        this_outlier['raw_ndarray_data'][old_cat_ii_mask] = 'CAT-I'
        this_outlier['raw_ndarray_data'][old_cat_iii_mask] = 'CAT-II'
        # Refactoring this_outlier['DataCategoryData obj'].abbreviation
        this_outlier['DataCategoryData obj'].abbreviation = OneDimensionNdarray(this_outlier['raw_ndarray_data'])
        # this_wt.outlier_plot(this_outlier['DataCategoryData obj'])
        save_pkl_file(this_wt.default_results_saving_path['outlier'], this_outlier)


def dalry_wind_farm_with_weather():
    darly_wind_farm, darly_wind_farm_total_curtailment_amount, operating_regime = darly_wind_farm_operating_regime()
    wind_direction = load_dalry_wind_farm_met_mast()
    DARLY_WIND_TURBINE_1.update_air_density_to_last_column()
    # %% Weather
    weather = darly_wind_farm[["wind speed"]].pd_view()
    weather = pd.merge(left=weather, right=DARLY_WIND_TURBINE_1[['air density']].pd_view(),
                       left_index=True, right_index=True, how="left")
    weather = pd.merge(left=weather, right=wind_direction[["wind direction"]],
                       left_index=True, right_index=True, how="left")
    wd = np.radians(weather["wind direction"].values)
    weather.drop("wind direction", axis=1, inplace=True)
    weather.loc[:, "wind direction cos"] = np.cos(wd)
    weather.loc[:, "wind direction sin"] = np.sin(wd)
    weather = weather.resample("60T").mean()
    weather.to_csv("./Dalry Hourly Weather.csv")

    # %% Pout modelling
    darly_wind_farm = darly_wind_farm.pd_view()
    darly_wind_farm.loc[~operating_regime("S1"), "active power output"] = np.nan  # type: pd.DataFrame
    # %% Save
    darly_wind_farm = darly_wind_farm.resample("60T").mean()
    darly_wind_farm = pd.merge(left=darly_wind_farm, right=weather.loc[:, weather.columns[1:]],
                               left_index=True, right_index=True, how="left")
    darly_wind_farm.to_csv("./Dalry Hourly.csv")


if __name__ == '__main__':
    pass
    # %% LOF and DBSCAN
    # cc = lof_and_dbscan_power_curve()
    #
    # %% Data exploratory
    # plot_raw_data_for_outlier_demo()
    #
    # %% WT-level outlier detector and plot
    # cat_6_demo()
    # cat_6_demo_time_series()
    # migrate_from_old_wind_turbine_outlier_detection()
    # individual_wind_turbine_outliers_outlier_detector('load',
    #                                                   wt_index=tuple(range(6)),
    #                                                   write_to_a_docx=True)
    # wind_turbine_level_outlier_results_demo()

    # %% WF-level operating regime analyser and plot
    # darly_wind_farm_operating_regime()
    # dalry_wind_farm_with_weather()

    # %% WF-level PC model study (with known wind turbines)
    # fit_or_analyse_darly_wind_farm_power_curve_model_with_known_wind_turbines(task='time series check')

    # %% WF-level PC model study (WITHOUT known wind turbines)
    fit_or_analyse_darly_wind_farm_power_curve_model_without_known_wind_turbines(task='fit')
    # fit_or_analyse_darly_wind_farm_power_curve_model_without_known_wind_turbines(task='time series check')

    # fit_or_analyse_zelengrad_wind_farm_power_curve_model(task='fit')
    # fit_or_analyse_zelengrad_wind_farm_power_curve_model(task='2D plot check')
    # fit_or_analyse_zelengrad_wind_farm_power_curve_model(task='time series check')

    # %% Fit wind turbine power curve model
    # fit_or_analysis_wind_turbine_power_curve_model(DARLY_WIND_TURBINE_2, '2D plot check')

    # %% Sasa WT outlier method
    # individual_wind_turbine_outliers_outlier_detector_sasa_method(DARLY_WIND_TURBINE_2)

    # DARLY_WIND_TURBINE_2.plot()
    # %% Test or debug codes, please ignore:
    # fit_plot_and_summary_all_mfr_pc_in_all_density('fit')
    # fit_plot_and_summary_all_mfr_pc_in_all_density('summary')
    # ZELENGRAD_WIND_FARM.outlier_detector()
    # ZELENGRAD_WIND_FARM.plot(plot_mfr=MFR_PC_LIMIT, mfr_mode='continuous', plot_scatter_pc=True)
    # DARLY_WIND_TURBINE_2[
    #     load_pkl_file(DARLY_WIND_TURBINE_2.default_results_saving_path["outlier"])['DataCategoryData obj'](
    #         ('CAT-IV.a', 'others'))
    # ].plot(plot_scatter_pc=True)

    # %% Rubbish
