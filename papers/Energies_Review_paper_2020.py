from PowerCurve_Class import PowerCurveByMfr
import pandas as pd
from initialise_wt_or_wf_Func import load_raw_wt_from_txt_file_and_temperature_from_csv
from BivariateAnalysis_Class import MethodOfBins
from Ploting.fast_plot_Func import *
from Wind_Class import Wind
from Ploting.adjust_Func import reassign_linestyles_recursively_in_ax, adjust_lim_label_ticks
from NdarraySubclassing import IntFloatConstructedOneDimensionNdarray
# import os
#
# os.environ["TF_XLA_FLAGS"] = "tf_xla_auto_jit=2"


def sasa_algorithm_to_cal_possible_pout_range(ws: Union[ndarray, float, int, range],
                                              highest_possible_ws: float = 30.) -> pd.DataFrame:
    # choose manufacturer power curve
    fixed_mfr_pc = PowerCurveByMfr('1.225')
    ws = IntFloatConstructedOneDimensionNdarray(ws)
    # %% Initialise the possible power output range, type pd.DataFrame
    possible_pout_range = pd.DataFrame(
        {
            'wind speed [m/s]': ws,
            'mfr pc used air density [kg/m^3]': np.full(ws.shape, np.nan),
            'possible power output lower bound [kW]': np.full(ws.shape, np.nan),
            'possible power output upper bound [kW]': np.full(ws.shape, np.nan),
        }
    )
    # %% Calculate all possible power output ranges for ws
    for i, this_ws in enumerate(ws):
        possible_pout_range['mfr pc used air density [kg/m^3]'] = fixed_mfr_pc.air_density
        possible_average_pout_in_this_10min = []
        # The loop controlling WS_low, the choices are 1, 2, ..., this_ws - 1
        for this_ws_lower_bound in range(1, int(this_ws)):
            # Supposing the lower possible wind speed can happen 1, 2, ..., 9 minutes in the 10min window
            for how_long_minutes_this_ws_lower_bound_happen in range(1, 10):
                how_long_minutes_this_ws_upper_bound_happen = 10 - how_long_minutes_this_ws_lower_bound_happen
                this_ws_upper_bound = (this_ws * 10 -
                                       how_long_minutes_this_ws_lower_bound_happen *
                                       this_ws_lower_bound) / how_long_minutes_this_ws_upper_bound_happen
                if this_ws_upper_bound > highest_possible_ws:
                    continue  # If WS_high > highest_possible_ws, the results will not be recorded
                # calculate the possible average power output in the 10min average window
                possible_average_pout_in_this_10min.append(
                    np.average(fixed_mfr_pc([this_ws_lower_bound, this_ws_upper_bound]),
                               weights=(how_long_minutes_this_ws_lower_bound_happen,
                                        how_long_minutes_this_ws_upper_bound_happen))
                )
                if (len(possible_average_pout_in_this_10min) == 25) and (this_ws == 5):
                    tt = 1
        # Write the minimum and maximum possible average power output to corresponding ws
        possible_pout_range.iloc[i, -2:] = [min(possible_average_pout_in_this_10min),
                                            max(possible_average_pout_in_this_10min)]
    ax = series(possible_pout_range.iloc[:, 0], possible_pout_range.iloc[:, -2], label='min')
    ax = scatter(possible_pout_range.iloc[:, 0], fixed_mfr_pc(possible_pout_range.iloc[:, 0].values), ax=ax,
                 label='Mfr-PC', color='r', marker='+', s=32)
    ax = series(possible_pout_range.iloc[:, 0], possible_pout_range.iloc[:, -1], ax=ax,
                x_label='WS [m/s]', y_label='Pout [kW]', label='max')
    ax = reassign_linestyles_recursively_in_ax(ax)

    return possible_pout_range


def demonstration_possible_pout_range_in_wind_speed_bins():
    # choose manufacturer power curve
    fixed_mfr_pc = PowerCurveByMfr('1.225')
    # load wind turbine
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
    this_wind_turbine = wind_turbines[0]
    # choose wind speed bins and the wind speed std. in each bin
    wind_speed = this_wind_turbine.measurements['wind speed'].values
    wind_speed_std = this_wind_turbine.measurements['wind speed std.'].values
    mob = MethodOfBins(wind_speed, wind_speed_std, bin_step=0.5)
    range_mask = np.bitwise_and(mob.array_of_bin_boundary[:, 1] >= 0,
                                mob.array_of_bin_boundary[:, 1] <= 30)
    wind_speed_range = mob.cal_mob_statistic(np.array([1.]))[range_mask, 0]
    wind_speed_std_range = mob.cal_mob_statistic(np.array([1.]))[range_mask, 1]
    # Initialise Wind instance
    wind = Wind(wind_speed_range, wind_speed_std_range)
    high_resol_wind = wind.simulate_transient_wind_speed_time_series(resolution=10,
                                                                     traces_number_for_each_recording=1_000_000,
                                                                     mode='cross sectional')
    simulated_pout = fixed_mfr_pc.cal_with_hysteresis_control_using_high_resol_wind(high_resol_wind)
    tt=1


if __name__ == "__main__":
    demonstration_possible_pout_range_in_wind_speed_bins()
