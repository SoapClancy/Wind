from Ploting.fast_plot_Func import *
from PowerCurve_Class import PowerCurveByMfr, PowerCurveFittedBy5PLF
from File_Management.load_save_Func import load_pkl_file
from File_Management.path_and_file_management_Func import try_to_find_file
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
from Ploting.uncertainty_plot_Func import plot_from_uncertainty_like_dataframe


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
    ax = series(np.arange(50, 51, 0.1), np.arange(50, 51, 0.1), ax=ax, color='g', marker='+', label='SIM range',
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

    simulated_pout = get_results()  # type: UncertaintyDataFrame
    preserved_data_pct = 99.99966
    simulated_pout_uct = np.percentile(simulated_pout, [(100 - preserved_data_pct) / 2,
                                                        100 - (100 - preserved_data_pct) / 2], axis=1).T
    simulated_pout_mean = np.mean(simulated_pout, axis=1)
    # 画error bar
    for i in range(cat_6_index.__len__()):
        ax.errorbar(wind_speed_range[i],
                    simulated_pout_mean[i],
                    yerr=np.array([[simulated_pout_mean[i] - simulated_pout_uct[i, 0]],
                                   [simulated_pout_uct[i, 1] - simulated_pout_mean[i]]]),
                    color='g', fmt='-+', markersize=10)

    return ax


def plot_raw_data_for_outlier_demo():
    """
    This function is to provide the very first figures in the paper, which shows that there seem to be outlier
    :return:
    """
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
    # Plot on-shore WT2
    wind_turbine_2 = wind_turbines[1]  # type: WT
    wind_turbine_2.plot(color='royalblue')
    # Plot on-shore WF, which is instanced by WT1 - WT6
    darly_wind_farm = WF.init_from_wind_turbine_instances(wind_turbines, obj_name='Dalry')
    darly_wind_farm.plot(color='royalblue')
    # Plot Zelengrad WF


if __name__ == '__main__':
    # fit_plot_and_summary_all_mfr_pc_in_all_density('fit')
    # fit_plot_and_summary_all_mfr_pc_in_all_density('summary')
    # tt = cat_6_demo()
    plot_raw_data_for_outlier_demo()
