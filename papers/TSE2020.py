from Ploting.fast_plot_Func import *
from PowerCurve_Class import PowerCurveByMfr, PowerCurveFittedBy5PLF
from File_Management.load_save_Func import load_pkl_file
from File_Management.path_and_file_management_Func import try_to_find_file, try_to_find_folder_path_otherwise_make_one
from project_utils import project_path_, WS_POUT_SCATTER_ALPHA, WS_POUT_2D_PLOT_KWARGS, WS_POUT_SCATTER_SIZE
from ErrorEvaluation_Class import DeterministicError
import pandas as pd
from prepare_datasets import load_raw_wt_from_txt_file_and_temperature_from_csv, load_croatia_data
from WT_WF_Class import WT, WF
from Wind_Class import cal_air_density, celsius_to_kelvin, Wind
from pathlib import Path
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save
from Data_Preprocessing.float_precision_control_Func import \
    covert_to_str_one_dimensional_ndarray
from ConvenientDataType import UncertaintyDataFrame
from Ploting.adjust_Func import *
from File_Management.path_and_file_management_Func import remove_win10_max_path_limit
from Filtering.OutlierAnalyser_Class import DataCategoryNameMapper
from typing import Tuple

remove_win10_max_path_limit()

########################################################################################################################
"""
This paper uses Dalry WF (especially its WT2 for individual WT analysis) and Zelengrad WF recordings
"""
# %% The Mfr-PC with the lowest and highest air densities
MFR_PC_LIMIT = (PowerCurveByMfr(air_density='0.97'), PowerCurveByMfr(air_density='1.27',
                                                                     color='lime',
                                                                     linestyle='--'))
# %% Darly wind turbines
DARLY_WIND_TURBINES = load_raw_wt_from_txt_file_and_temperature_from_csv()
# This paper essentially only have 2D analysis
for i_outer in range(DARLY_WIND_TURBINES.__len__()):
    DARLY_WIND_TURBINES[i_outer].predictor_names = ('wind speed',)
del i_outer
# Darly WT2 is of the most interest
DARLY_WIND_TURBINE_2 = DARLY_WIND_TURBINES[1]  # type: WT
# WT outlier results
DARLY_WIND_TURBINES_OUTLIERS = []
for this_wind_turbine_outer in DARLY_WIND_TURBINES:
    DARLY_WIND_TURBINES_OUTLIERS.append(load_pkl_file(this_wind_turbine_outer.default_results_saving_path["outlier"]))

DARLY_WIND_FARM_RAW, _ = WF.init_from_wind_turbine_instances(DARLY_WIND_TURBINES, obj_name='Dalry raw')  # type: WF


# %% Zelengrad WF
# ZELENGRAD_WIND_FARM = load_croatia_data('Zelengrad')['Zelengrad']  # type: WF
# # Aggregate to 10 min
# ZELENGRAD_WIND_FARM = ZELENGRAD_WIND_FARM.resample(
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
    wind_turbine_1 = wind_turbines[0]  # type: WT
    wind_turbine_1['active power output'] /= 3000
    # Only analyse day 8.5 to 9
    wind_turbine_1 = wind_turbine_1.iloc[int(8.5 * 144):int(9 * 144)]
    # air density based PC
    air_density = cal_air_density(celsius_to_kelvin(wind_turbine_1['environmental temperature'].values),
                                  wind_turbine_1['relative humidity'].values / 100,
                                  wind_turbine_1['barometric pressure'].values * 100)
    mfr_pc = PowerCurveByMfr.init_multiple_instances(air_density=air_density)
    # manually set the index of CAT-VI
    cat_6_index = [2, 3, 11, 21]
    # 画初始垃圾图
    cat_6_index_indicator = np.isin(range(wind_turbine_1.__len__()), cat_6_index)
    # ax = scatter(wind_turbine_1['wind speed'].values[~cat_6_index_indicator],
    #              wind_turbine_1['active power output'].values[~cat_6_index_indicator],
    #              color='b', s=4, label='Normal',
    #              rasterized=False)
    # ax = scatter(
    #     wind_turbine_1['wind speed'].values[cat_6_index_indicator],
    #     wind_turbine_1['active power output'].values[cat_6_index_indicator],
    #     ax=ax,
    #     facecolors='none', edgecolors='r', marker='*',
    #     s=32,
    #     zorder=10,
    #     label='CAT-VI',
    #     rasterized=False
    # )
    # ax = mfr_pc[2].plot(ax=ax)
    # # range label
    # ax = series(np.arange(50, 51, 0.1), np.arange(50, 51, 0.1), ax=ax, color='black',
    #             label='4.5' + r'$\sigma$' + ' %' + '\nrange',
    #             **{'x_label': 'Wind Speed [m/s]',
    #                'y_label': 'Active Power Output [p.u.]'})
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
        wind = Wind(wind_speed_range, wind_speed_std_range)
        high_resol_wind = wind.simulate_transient_wind_speed_time_series(
            resolution=simulation_resolution,
            traces_number_for_each_recording=simulation_traces,
            mode=mode
        )
        # TODO 加入初始值设定
        _simulated_pout = mfr_pc[2].cal_with_hysteresis_control_using_high_resol_wind(
            high_resol_wind,
            return_percentiles=None,
            mode=mode
        )
        return _simulated_pout

    simulated_pout = get_results()  # type: ndarray
    simulated_pout = UncertaintyDataFrame.init_from_2d_ndarray(simulated_pout.T)  # type: UncertaintyDataFrame
    preserved_data_pct = 99.99966
    # simulated_pout_uct = np.percentile(simulated_pout, [(100 - preserved_data_pct) / 2,
    #                                                     100 - (100 - preserved_data_pct) / 2], axis=1).T
    # simulated_pout_mean = np.mean(simulated_pout, axis=1)
    # 画error bar
    for i in range(cat_6_index.__len__()):
        ax = scatter(np.array([wind_speed_range[i]] * 2), simulated_pout(by_sigma=1).iloc[:, i].values, ax=ax,
                     label='1' + r'$\sigma$' + ' %' if i == 0 else None, rasterized=False,
                     color='grey', marker='+', s=100, zorder=12)
        ax = scatter(np.array([wind_speed_range[i]] * 2), simulated_pout(by_sigma=2).iloc[:, i].values, ax=ax,
                     label='2' + r'$\sigma$' + ' %' if i == 0 else None, rasterized=False,
                     color='fuchsia', marker='x', s=32, zorder=12)
        ax = scatter(np.array([wind_speed_range[i]] * 2), simulated_pout(by_sigma=3).iloc[:, i].values, ax=ax,
                     label='3' + r'$\sigma$' + ' %' if i == 0 else None, rasterized=False,
                     color='royalblue', marker='s', s=16, zorder=12)

        ax = scatter(wind_speed_range[i], simulated_pout.loc['mean'][i], ax=ax,
                     label='Mean' if i == 0 else None, rasterized=False,
                     color=(0, 1, 0), marker='1', s=64, zorder=12)

        ax = scatter(wind_speed_range[i], simulated_pout(0).iloc[0, i], ax=ax,
                     label='Median' if i == 0 else None, rasterized=False,
                     color='orange', marker='2', s=64, zorder=12,
                     x_label='Wind Speed [m/s]',
                     y_label='Active Power Output [p.u.]')

        ax.errorbar(wind_speed_range[i],
                    simulated_pout.loc['mean'][i],
                    yerr=np.array([[simulated_pout.loc['mean'][i] - simulated_pout(by_sigma=4.5).iloc[0, i]],
                                   [simulated_pout(by_sigma=4.5).iloc[1, i] - simulated_pout.loc['mean'][i]]]),
                    color='k', fmt='-', markersize=10)
    plt.gca().legend(ncol=1, loc='upper left', prop={'size': 10})

    return ax


def plot_raw_data_for_outlier_demo():
    """
    This function is to provide the very first figures in the paper, which shows that there seem to be outlier
    :return:
    """

    for to_plot_obj in (DARLY_WIND_TURBINE_2, DARLY_WIND_FARM_RAW, ZELENGRAD_WIND_FARM):
        exec("to_plot_obj.plot(plot_mfr=MFR_PC_LIMIT, plot_scatter_pc=True)")


def individual_wind_turbine_outliers_outlier_detector():
    for i, this_wt in enumerate(load_raw_wt_from_txt_file_and_temperature_from_csv()):
        this_wt.predictor_names = ('wind speed',)
        this_wt.outlier_detector()
        # this_wt.outlier_plot(outlier)
        # this_wt.outlier_report(outlier)


def wind_turbine_level_outlier_results_demo():
    for i, _ in enumerate(load_raw_wt_from_txt_file_and_temperature_from_csv()):
        # if i != 1:
        #     continue
        _.outlier_plot()
        # _.outlier_plot(plot_individual=True)
        # _.outlier_report()
    # DARLY_WIND_TURBINE_2.outlier_plot()
    # DARLY_WIND_TURBINE_2.outlier_plot(plot_individual=True)
    # DARLY_WIND_TURBINE_2.outlier_report()


def darly_wind_farm_operating_regime():
    # %% Select by wind turbines outliers
    # In the WF analyse, only 'CAT-I.a', 'CAT-I.b', 'CAT-II', 'CAT-IV.a' and 'others' WT recordings are used
    # wind_turbines_with_selected_recordings = []
    wind_turbine_instances_data_category = []
    for i in range(DARLY_WIND_TURBINES_OUTLIERS.__len__()):
        # %% rename to adapt to WF-level analysis, and the rules below are used:
        # 'CAT-I.a', 'CAT-I.b' are 'shutdown'
        # 'CAT-II' are 'curtailed'
        # 'CAT-IV.a' and 'others' are 'operating'
        this_outlier = DARLY_WIND_TURBINES_OUTLIERS[i]['DataCategoryData obj']
        this_outlier.rename(
            mapper={'CAT-I.a': 'shutdown', 'CAT-I.b': 'shutdown',
                    'CAT-II': 'curtailed',
                    'CAT-IV.a': 'operating', 'others': 'operating',
                    'CAT-III': 'nan', 'CAT-IV.b': 'nan', 'missing': 'nan'},
            new_name_mapper=DataCategoryNameMapper(
                [["shutdown", "shutdown", 3, "originally be either CAT-I.a or CAT-I.b"],
                 ["curtailed", "curtailed", 2, "originally be CAT-II"],
                 ["operating", "operating", 1, "originally be either CAT-IV.a or others"],
                 ["nan", "nan", -1, "originally be either CAT-III, CAT-IV.b or missing"]],
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
    # darly_wind_farm.plot(operating_regime=operating_regime, plot_mfr=MFR_PC_LIMIT)

    # %% report
    operating_regime.report(
        sorted_kwargs={'key': lambda x: "0" + x[1:] if x[1:].__len__() < 2 else x[1:]},
        report_pd_to_csv_file_path=darly_wind_farm.default_results_saving_path['operating regime']
    )
    return darly_wind_farm, darly_wind_farm_total_curtailment_amount, operating_regime


if __name__ == '__main__':
    # fit_plot_and_summary_all_mfr_pc_in_all_density('fit')
    # fit_plot_and_summary_all_mfr_pc_in_all_density('summary')
    # cat_6_demo()
    # plot_raw_data_for_outlier_demo()

    # individual_wind_turbine_outliers_outlier_detector()
    # wind_turbine_level_outlier_results_demo()

    darly_wind_farm_operating_regime()
