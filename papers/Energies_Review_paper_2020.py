from PowerCurve_Class import PowerCurveByMfr
import pandas as pd
from initialise_wt_or_wf_Func import load_raw_wt_from_txt_file_and_temperature_from_csv
from BivariateAnalysis_Class import MethodOfBins
from Ploting.fast_plot_Func import *
from Wind_Class import Wind
from Ploting.adjust_Func import reassign_linestyles_recursively_in_ax, adjust_lim_label_ticks
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray, IntOneDimensionNdarray
from File_Management.load_save_Func import load_exist_pkl_file_otherwise_run_and_save, load_pkl_file, \
    load_exist_npy_file_otherwise_run_and_save
from project_path_Var import project_path_
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

# %% Global variables
# choose manufacturer power curve
FIXED_MFR_PC = PowerCurveByMfr('1.225')
# load wind turbine
wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
THIS_WIND_TURBINE = wind_turbines[0]
# choose wind speed bins and the wind speed std. in each bin
wind_speed = THIS_WIND_TURBINE.measurements['wind speed'].values
wind_speed_std = THIS_WIND_TURBINE.measurements['wind speed std.'].values
mob = MethodOfBins(wind_speed, wind_speed_std, bin_step=0.5)
range_mask = np.bitwise_and(mob.array_of_bin_boundary[:, 1] >= 0,
                            mob.array_of_bin_boundary[:, 1] <= 30)
WIND_SPEED_RANGE = mob.cal_mob_statistic(np.array([1.]))[range_mask, 0]
WIND_SPEED_STD_RANGE = mob.cal_mob_statistic(np.array([1.]))[range_mask, 1]
SIMULATION_RESOLUTION = 10
SIMULATION_TRACES = 10_000_000
SIMULATION_RETURN_PERCENTILES = covert_to_str_one_dimensional_ndarray(np.arange(0, 100.5, 0.5), '0.1')
del wind_turbines, wind_speed, wind_speed_std, mob, range_mask


def plot_mfr_pc(_ax):
    _ax = FIXED_MFR_PC.plot_power_curve(np.concatenate((range(0, 26),
                                                        [25 + float_eps * 100],
                                                        range(26, 31))),
                                        ax=_ax, zorder=300, mode='discrete')
    _ax.legend(loc='upper left')
    _ax = adjust_lim_label_ticks(_ax, x_lim=(-0.5, 30.5))
    return _ax


def uncertainty_plot(uncertainty_dataframe: UncertaintyDataFrame):
    _ax = plot_from_uncertainty_like_dataframe(
        WIND_SPEED_RANGE,
        uncertainty_dataframe,
        covert_to_str_one_dimensional_ndarray(np.arange(0, 50.5, 0.5), '0.1'),
    )

    return plot_mfr_pc(_ax)


def plot_wind_speed_std():
    ax = series(WIND_SPEED_RANGE, WIND_SPEED_STD_RANGE, marker='*', mec='r', ms=8,
                x_label='Wind speed [m/s]', y_label='Wind speed standard deviation [m/s]',
                x_lim=(-0.5, 30.5))
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
    return uncertainty_plot(results)


def demonstration_possible_pout_range_in_wind_speed_bins_my_proposal_new():
    @load_exist_pkl_file_otherwise_run_and_save(
        Path(project_path_) / 'Data/Results/transient_study/Energies_Review_paper_2020/mine_all_range_check.pkl')
    def get_results():
        mode = 'cross sectional'
        # Initialise Wind instance
        wind = Wind(WIND_SPEED_RANGE, WIND_SPEED_STD_RANGE)
        high_resol_wind = wind.simulate_transient_wind_speed_time_series(
            resolution=SIMULATION_RESOLUTION,
            traces_number_for_each_recording=SIMULATION_TRACES,
            mode=mode
        )
        _simulated_pout = FIXED_MFR_PC.cal_with_hysteresis_control_using_high_resol_wind(
            high_resol_wind,
            return_percentiles=SIMULATION_RETURN_PERCENTILES,
            mode=mode
        )
        return _simulated_pout

    simulated_pout = get_results()  # type: UncertaintyDataFrame

    return uncertainty_plot(simulated_pout)


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

    @load_exist_npy_file_otherwise_run_and_save(
        Path(project_path_) / 'Data/Results/transient_study/Energies_Review_paper_2020/iec_simulation.npy')
    def get_results():

        _results = np.full((WIND_SPEED_RANGE.shape[0], 2), np.nan)
        _results[:, 0] = WIND_SPEED_RANGE
        for i in range(WIND_SPEED_RANGE.shape[0]):
            if i == 0:
                _results[i, 1] = 0
                continue
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
            p_sim_i = tfp.monte_carlo.expectation(
                f=FIXED_MFR_PC.__call__,
                samples=samples.astype(np.float32),
                log_prob=tfp.distributions.Normal(loc=WIND_SPEED_RANGE[i],
                                                  scale=WIND_SPEED_STD_RANGE[i]).log_prob
            )
            _results[i, 1] = float(p_sim_i)
        return _results

    results = get_results()

    _ax = series(results[:, 0], results[:, 1], color=(0, 1, 0), linestyle='--', label='Mean')

    return plot_mfr_pc(_ax)


if __name__ == "__main__":
    ax_mine_new = demonstration_possible_pout_range_in_wind_speed_bins_my_proposal_new()
    # ax_mine_old = demonstration_possible_pout_range_in_wind_speed_bins_my_proposal_old()
    # ax_sasa = sasa_algorithm_to_cal_possible_pout_range()
    # ax_sasa_pmaps = sasa_pmaps_to_cal_possible_pout_range()
    # ax_iec_standard = demonstration_iec_standard()
    # ax_std = plot_wind_speed_std()
