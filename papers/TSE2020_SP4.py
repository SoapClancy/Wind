from Ploting.fast_plot_Func import *
from PowerCurve_Class import PowerCurveByMfr, PowerCurveFittedBy5PLF
from File_Management.load_save_Func import load_pkl_file
from File_Management.path_and_file_management_Func import try_to_find_file
from project_utils import project_path_
from ErrorEvaluation_Class import DeterministicError
import pandas as pd


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
    mfr_pc_to_fit = []
    for i, this_mfr_pc in enumerate(mfr_pc):
        if i < 7:
            continue
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
                                   run_n_times=100,
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
            if i == mfr_pc.__len__() - 1:
                error_df.to_csv('12Mfr-PC summary.csv')
            mfr_pc_to_fit.append(this_mfr_pc_to_fit)
            # Plot
            # if i % 4 == 0:
            #     ax = None
            # ax = this_mfr_pc.plot(ws=range(4, 26), mode='discrete', ax=ax)
            # ax = this_mfr_pc_to_fit.plot(plot_recordings_and_mob=False,
            #                              ax=ax,
            #                              save_file_=this_mfr_pc.air_density,
            #                              title=this_mfr_pc.air_density)
            # ax.get_legend().remove()

    if mode == 'summary':
        ax = None
        mfr_wind_speed = np.arange(4, 26)
        wind_speed = np.arange(0, 25.1, 0.1)
        upper_layer = 2
        lower_layer = 1
        # 0.97
        ax = series(x=wind_speed, y=mfr_pc_to_fit[0](wind_speed), color='k', linestyle=':', ax=ax,
                    label='5PLF (' + r'$\rho$' + f'={mfr_pc[0].air_density} kg/m'+'$^3$)', zorder=lower_layer)
        ax = scatter(x=mfr_wind_speed, y=mfr_pc[0](mfr_wind_speed), color='r', marker='+', s=32, ax=ax,
                     rasterized=False,
                     label='Mfr-PC (' + r'$\rho$' + f'={mfr_pc[0].air_density} kg/m'+'$^3$)', zorder=upper_layer)

        # 1.12
        ax = series(x=wind_speed, y=mfr_pc_to_fit[5](wind_speed), color='darkorange', linestyle='--', ax=ax,
                    label='5PLF (' + r'$\rho$' + f'={mfr_pc[5].air_density} kg/m'+'$^3$)', zorder=lower_layer)
        ax = scatter(x=mfr_wind_speed, y=mfr_pc[5](mfr_wind_speed), color='g', marker='2', s=45, ax=ax,
                     rasterized=False,
                     label='Mfr-PC (' + r'$\rho$' + f'={mfr_pc[5].air_density} kg/m'+'$^3$)', zorder=upper_layer)

        # # 1.15
        # ax = series(x=wind_speed, y=mfr_pc_to_fit[6](wind_speed), color='g', linestyle='-.', ax=ax,
        #             label='5PLF (' + r'$\rho$' + f'={mfr_pc[6].air_density})', zorder=lower_layer)
        # ax = scatter(x=mfr_wind_speed, y=mfr_pc[6](mfr_wind_speed), color='r', marker='+', s=32, ax=ax,
        #              label='Mfr-PC (' + r'$\rho$' + f'={mfr_pc[6].air_density})', zorder=upper_layer)

        # 1.27
        ax = series(x=wind_speed, y=mfr_pc_to_fit[11](wind_speed), color='b', linestyle='-', ax=ax,
                    label='5PLF (' + r'$\rho$' + f'={mfr_pc[11].air_density} kg/m'+'$^3$)', zorder=lower_layer)
        ax = scatter(x=mfr_wind_speed, y=mfr_pc[11](mfr_wind_speed), marker='o', s=16, ax=ax,
                     rasterized=False,
                     facecolors='none', edgecolors='magenta',
                     label='Mfr-PC (' + r'$\rho$' + f'={mfr_pc[11].air_density} kg/m'+'$^3$)', zorder=upper_layer,
                     y_lim=(-0.05, 1.05),
                     x_lim=(-0.05, 26.5),
                     x_label='Wind speed [m/s]', y_label='Active power output [p.u.]',
                     save_file_='12Mfr-PC_fittings',
                     save_format='svg')


if __name__ == '__main__':
    fit_plot_and_summary_all_mfr_pc_in_all_density('fit')
    # fit_plot_and_summary_all_mfr_pc_in_all_density('summary')
