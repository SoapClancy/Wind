import numpy as np
from numpy import ndarray
import pandas as pd
from Ploting.fast_plot_Func import *
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray
from initialise_wt_or_wf_Func import load_raw_wt_from_txt_file_and_temperature_from_csv
from PowerCurve_Class import PowerCurveByMfr


def celsius_to_kelvin(celsius):
    """
    This function do the transformation from Celsius degree(s) to Kelvin degree(s)
    """
    return celsius + 273.15


def cal_air_density(absolute_temperature_in_kelvin: Union[int, float, ndarray],
                    relative_humidity_from_0_to_1: Union[int, float, ndarray],
                    barometric_pressure_in_pascal: Union[int, float, ndarray]) -> ndarray:
    """
    This function is identical to Equation (F.1) on page 156 of IEC 61400-12-1:2017
    :param absolute_temperature_in_kelvin: is the absolute temperature [K];
    :param relative_humidity_from_0_to_1: is the relative humidity (range 0 to 1);
    :param barometric_pressure_in_pascal: barometric pressure [Pa];
    :return:
    """
    # Variables
    T_10min = IntFloatConstructedOneDimensionNdarray(absolute_temperature_in_kelvin)
    phi = IntFloatConstructedOneDimensionNdarray(relative_humidity_from_0_to_1)
    B_10min = IntFloatConstructedOneDimensionNdarray(barometric_pressure_in_pascal)
    # Constants
    R_0 = 287.05  # [J/kgK]
    R_W = 461.5  # [J/kgK]
    P_W = 0.0000205 * np.exp(0.0631846 * T_10min)  # [Pa]
    rho_10min = (1 / T_10min) * (B_10min / R_0 - phi * P_W * (1 / R_0 - 1 / R_W))  # [kg/m^3]
    return np.array(rho_10min)


def cal_turbulence_intensity(wind_speed: Union[int, float, ndarray],
                             wind_speed_standard_deviation: Union[int, float, ndarray]) -> ndarray:
    """
    This function is to calculate the turbulence intensity, please refer to Annex M of IEC 61400-12-1:2017
    :param wind_speed: [m/s]
    :param wind_speed_standard_deviation: [m/s]
    :return:
    """
    wind_speed = IntFloatConstructedOneDimensionNdarray(wind_speed)
    wind_speed_standard_deviation = IntFloatConstructedOneDimensionNdarray(wind_speed_standard_deviation)
    return np.array(wind_speed_standard_deviation / wind_speed)


if __name__ == '__main__':
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
    this_wind_turbine = wind_turbines[0]
    rho = cal_air_density(
        celsius_to_kelvin(this_wind_turbine.measurements['environmental temperature'].values),
        this_wind_turbine.measurements['relative humidity'].values / 100,
        this_wind_turbine.measurements['barometric pressure'].values * 100
    )
    mfr_pc_instances = PowerCurveByMfr.init_multiple_instances(rho)
    for i, this_mfr_pc in enumerate(mfr_pc_instances):
        this_mfr_pc.sasa_algorithm_to_cal_possible_pout_range(range(2, 26))
        # if i not in (1245, 10929):
        #     continue
        # this_mfr_pc.simulation_based_algorithm_to_cal_possible_pout_range(
        #     this_wind_turbine.measurements['wind speed'].values[i],
        #     this_wind_turbine.measurements['wind speed std.'].values[i],
        #     resolution=10,
        #     traces=100_000
        # )



