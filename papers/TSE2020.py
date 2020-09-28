from PowerCurve_Class import *
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
from ConvenientDataType import UncertaintyDataFrame, IntOneDimensionNdarray, IntFloatConstructedOneDimensionNdarray
from Ploting.adjust_Func import *
from File_Management.path_and_file_management_Func import remove_win10_max_path_limit
from Filtering.OutlierAnalyser_Class import DataCategoryNameMapper
from typing import Tuple
from Filtering.OutlierAnalyser_Class import DataCategoryData
from parse import parse
from ErrorEvaluation_Class import DeterministicError

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
BIN_WIDTH = 1.
DARLY_WIND_FARM_RAW_MOB_PC = PowerCurveByMethodOfBins(
    DARLY_WIND_FARM_RAW['wind speed'].values,
    DARLY_WIND_FARM_RAW['active power output'].values / DARLY_WIND_FARM_RAW.rated_active_power_output,
    bin_width=1
)


# %% Zelengrad WF
# ZELENGRAD_WIND_FARM = load_croatia_data('Zelengrad')['Zelengrad']  # type: WF
# ZELENGRAD_WIND_FARM.number_of_wind_turbine = 14
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
        #
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


def darly_wind_farm_operating_regime() -> Tuple[WF, pd.DataFrame, DataCategoryData]:
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
                                       run_n_times=5000,
                                       wind_speed=np.arange(0, 29.5, 0.1),
                                       focal_error=0.0025,
                                       save_to_file_path=fitting_file_path)
    else:
        print(fully_operating_8_param_pc)
        # %% Plot GA fitting convergence and GA results
        series(np.array(list(map(lambda x: x['function'], load_pkl_file(fitting_file_path)))),
               title='GA Fitting Convergence')
        fully_operating_8_param_pc.plot(plot_recording=True)

        # %% Fully operating to partly operating linear scaling plus bias analysis
        darly_wind_farm_equivalent_power_curve_model = \
            EquivalentWindFarmPowerCurve.init_from_power_curve_fitted_by_8plf_obj(fully_operating_8_param_pc,
                                                                                  total_wind_turbine_number=6)
        if task == '2D plot check':
            # %% Define a function to obtain the corresponding PowerCurveByMethodOfBins obj
            def obtain_corresponding_power_curve_by_method_of_bins_obj(index_in_darly_wind_farm: ndarray):
                this_operating_regime_mob_pc_inner = PowerCurveByMethodOfBins(
                    *darly_wind_farm.loc[index_in_darly_wind_farm, ['wind speed', 'active power output']].values.T,
                    bin_width=BIN_WIDTH,
                )
                ws_inner = np.array(
                    [x['this_bin_boundary'][1]
                     for x in this_operating_regime_mob_pc_inner.corresponding_mob_obj.mob.values()
                     if not x['this_bin_is_empty']]
                )
                power_output_inner = np.array(
                    [np.mean(x['dependent_var_in_this_bin'])
                     for x in this_operating_regime_mob_pc_inner.corresponding_mob_obj.mob.values()
                     if not x['this_bin_is_empty']]
                )
                return this_operating_regime_mob_pc_inner, ws_inner, power_output_inner

            ax = None
            error_df = pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    [['Mfr PC 0.97', 'Mfr PC 1.27', 'Raw', 'My model'], ['RMSE', 'MAE']],
                    names=['Model used', 'error']
                ),
                index=[x for x in operating_regime.name_mapper['abbreviation'] if x != 'others']
            )
            for i, (this_operating_regime_long_name, this_operating_regime_abbreviation) in enumerate(zip(
                    operating_regime.name_mapper['long name'],
                    operating_regime.name_mapper['abbreviation']
            )):
                if this_operating_regime_abbreviation == 'others':
                    continue
                parse_obj = parse("({}, {}, {})", this_operating_regime_long_name)
                operating_wind_turbine_number = int(parse_obj[0])
                curtailed_wind_turbine_number = int(parse_obj[1])
                actual_recording = []
                output_dict = {x: [] for x in error_df.columns.get_level_values('Model used').unique()}

                # %% Define a function to obtain power outputs for error analysis and also for plot
                def obtain_power_output_for_error_and_plot(total_curtailment_amount_inner):
                    power_output_inner = darly_wind_farm_equivalent_power_curve_model(
                        ws,
                        operating_wind_turbine_number=IntOneDimensionNdarray([operating_wind_turbine_number]),
                        total_curtailment_amount=total_curtailment_amount_inner
                    )
                    power_output_plot_inner = darly_wind_farm_equivalent_power_curve_model(
                        ws_plot,
                        operating_wind_turbine_number=IntOneDimensionNdarray([operating_wind_turbine_number]),
                        total_curtailment_amount=total_curtailment_amount_inner
                    )
                    output_dict['My model'].extend(power_output_inner)
                    output_dict['Mfr PC 0.97'].extend(MFR_PC_LIMIT[0](ws))
                    output_dict['Mfr PC 1.27'].extend(MFR_PC_LIMIT[-1](ws))
                    output_dict['Raw'].extend(DARLY_WIND_FARM_RAW_MOB_PC(ws))
                    actual_recording.extend(bin_pout)
                    return power_output_plot_inner

                # If there are any WT curtailment, note that the amount of curtailment is uncertain can be be any value!
                # So, the pain here is to do the more detailed check again, for all unique curtailment,
                # despite that the curtailed_wind_turbine_number is the same!
                if curtailed_wind_turbine_number > 0:
                    curtailment_in_this_operating_regime = darly_wind_farm_total_curtailment_amount[
                        operating_regime(this_operating_regime_abbreviation)
                    ]
                    # round to 4 decimal places to reduce noise effects
                    curtailment_in_this_operating_regime_round = curtailment_in_this_operating_regime.round(4)
                    # Do a for-loop for considering all different curtailment amount separately
                    for _, curtailment in enumerate(np.unique(curtailment_in_this_operating_regime_round)):
                        index = curtailment_in_this_operating_regime_round[
                            curtailment == curtailment_in_this_operating_regime_round].index
                        this_operating_regime_mob_pc, ws, bin_pout = \
                            obtain_corresponding_power_curve_by_method_of_bins_obj(index)
                        ws_plot = np.arange(np.min(ws), np.max(ws) + 0.01, 0.01)
                        # Wind speed ndarray obj for calculation error
                        power_output_plot = obtain_power_output_for_error_and_plot(curtailment)

                        if (this_operating_regime_long_name == '(5, 1, 0)') and (_ == 0):
                            label = 'LS-M. PC\nplus actl-C.'
                            label_2 = 'Scatters PC'
                        else:
                            label = None
                            label_2 = None
                        ax = series(ws_plot, power_output_plot, ax=ax, color='red', linestyle='-', label=label)
                        ax = series(ws, bin_pout, ax=ax, color='black', linestyle='--', label=label_2)

                # Much easy and straight forward if there are no WT curtailment
                else:
                    this_operating_regime_mob_pc, ws, bin_pout = obtain_corresponding_power_curve_by_method_of_bins_obj(
                        operating_regime(this_operating_regime_abbreviation)
                    )
                    total_curtailment_amount = 0
                    ws_plot = np.arange(0, 30, 0.01)
                    # Wind speed ndarray obj for calculation error
                    power_output_plot = obtain_power_output_for_error_and_plot(total_curtailment_amount)

                    if this_operating_regime_abbreviation == 'S1':
                        ax = fully_operating_8_param_pc.plot(ws=ws_plot, ax=ax, plot_recording=False, color='blue',
                                                             linestyle='-', label='M. PC')
                    else:
                        label = 'LS-M. PC' if this_operating_regime_long_name == '(5, 0, 1)' else None
                        ax = series(ws_plot, power_output_plot, ax=ax, color='green', linestyle='-', label=label)
                    ax = series(ws, bin_pout, ax=ax, color='black', linestyle='--')

                # %% Calculate error
                for key in error_df.columns.get_level_values('Model used').unique():
                    error_obj = DeterministicError(target=np.array(actual_recording).flatten(),
                                                   model_output=np.array(output_dict[key]).flatten())
                    error_df.loc[this_operating_regime_abbreviation,
                                 (key, 'RMSE')] = error_obj.cal_root_mean_square_error()
                    error_df.loc[this_operating_regime_abbreviation,
                                 (key, 'MAE')] = error_obj.cal_mean_absolute_error()
            ax = scatter(*darly_wind_farm.loc[~operating_regime('others'), ['wind speed',
                                                                            'active power output']].values.T,
                         ax=ax, color='silver', alpha=0.75, zorder=-1, label="Actl-M.")
            # Adjust the order of the legend
            ax = adjust_legend_order_in_ax(ax, new_order_of_labels=('Actl-M.',
                                                                    'Scatters PC',
                                                                    'M. PC',
                                                                    'LS-M. PC',
                                                                    'LS-M. PC\nplus actl-C.'))
            error_df.to_csv(fitting_file_path.parent / 'errors.csv')
            for this_error in error_df.columns.get_level_values('error').unique():
                ax_error = series(error_df.index, error_df[('Mfr PC 0.97', this_error)].values,
                                  label=MFR_PC_LIMIT[0].label, figure_size=(5, 5 * (0.618 ** 2)),
                                  marker='^', markersize=6, x_label='WF Operating Regime', y_label=f'{this_error}')
                ax_error = series(error_df.index, error_df[('Raw', this_error)].values, label='Raw', color='fuchsia',
                                  marker='*', markersize=8, ax=ax_error, legend_loc='upper center',
                                  legend_ncol=4, x_label='WF Operating Regime', y_label=f'{this_error}')
                ax_error = series(error_df.index, error_df[('Mfr PC 1.27', this_error)].values,
                                  label=MFR_PC_LIMIT[-1].label,
                                  marker='v', markersize=6, ax=ax_error, legend_loc='upper center',
                                  legend_ncol=4, x_label='WF Operating Regime', y_label=f'{this_error}')
                ax_error = series(error_df.index, error_df[('My model', this_error)].values, label='Model',
                                  ax=ax_error, marker='.', markersize=8, x_ticks_rotation=45, color='green',
                                  y_lim=(-0.04, 0.91 if this_error == 'MAE' else 1.11), legend_loc='upper left',
                                  legend_ncol=2)
            return error_df
        else:
            for i, (this_operating_regime_long_name, this_operating_regime_abbreviation) in enumerate(zip(
                    operating_regime.name_mapper['long name'],
                    operating_regime.name_mapper['abbreviation']
            )):
                if this_operating_regime_abbreviation == 'others':
                    continue


def fit_or_analyse_darly_wind_farm_power_curve_model_without_known_wind_turbines(task: str, *,
                                                                                 ga_max_num_iteration=None):
    assert task in ('time series check', '2D plot check', 'fit')

    # Make sure the global variable DARLY_WIND_FARM_RAW will not be changed accidentally,
    # Also, change the obj_name attribute, since they are now treated as single-sensor measurements
    darly_wind_farm_single = copy.deepcopy(DARLY_WIND_FARM_RAW)
    darly_wind_farm_single.obj_name = 'Darly single'

    # %% Do the outlier analyse
    outlier = darly_wind_farm_single.outlier_detector('30T', )
    # Set CAT-III outliers to be nan
    darly_wind_farm_single[outlier(('CAT-III', 'missing'))] = np.nan

    results = darly_wind_farm_single.operating_regime_detector_initial_guess()
    # %% Try to find the operating regime
    if task == 'fit':
        darly_wind_farm_single.operating_regime_detector(initial_guess_results=results,
                                                         ga_max_num_iteration=ga_max_num_iteration)
    # %% Check results
    else:
        darly_wind_farm_single.operating_regime_detector(initial_guess_results=results)
        if task == '2D plot check':
            pass
        else:
            pass


if __name__ == '__main__':
    # %% WT-level outlier detector and plot
    # individual_wind_turbine_outliers_outlier_detector()
    # wind_turbine_level_outlier_results_demo()

    # %% WF-level operating regime analyser and plot
    # darly_wind_farm_operating_regime()

    # %% WF-level PC model study (with known wind turbines)
    # fit_or_analyse_darly_wind_farm_power_curve_model_with_known_wind_turbines(task='2D plot check')

    # %% WF-level PC model study (WITHOUT known wind turbines)
    fit_or_analyse_darly_wind_farm_power_curve_model_without_known_wind_turbines(task='fit', ga_max_num_iteration=1600)

    # %% Test or debug codes, please ignore:
    # fit_plot_and_summary_all_mfr_pc_in_all_density('fit')
    # fit_plot_and_summary_all_mfr_pc_in_all_density('summary')
    # cat_6_demo()
    # plot_raw_data_for_outlier_demo()
    # ZELENGRAD_WIND_FARM.outlier_detector()
