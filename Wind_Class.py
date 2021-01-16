from ConvenientDataType import IntFloatConstructedOneDimensionNdarray
from Ploting.fast_plot_Func import *
import tensorflow as tf
import tensorflow_probability as tfp
from PowerCurve_Class import PowerCurveByMfr
from prepare_datasets import load_raw_wt_from_txt_file_and_temperature_from_csv
from typing import Iterator, Union, Callable
from BivariateAnalysis_Class import MethodOfBins
from pathlib import Path
from File_Management.path_and_file_management_Func import list_all_specific_format_files_in_a_folder_path
import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression
from tensorflow_probability.python.util.seed_stream import SeedStream
from tensorflow_probability.python.internal import dtype_util
import getpass


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


class Wind:
    __slots__ = ('wind_speed', 'wind_speed_std', 'original_resolution')

    def __init__(self,
                 wind_speed: Union[int, float, ndarray],
                 wind_speed_std: Union[int, float, ndarray],
                 original_resolution=600):
        """
        :param wind_speed:
        :param wind_speed_std:
        :param original_resolution original resolution of the recording, default is 600 [seconds], as suggested by IEC
        Standard 61400-12-1
        """
        self.wind_speed = IntFloatConstructedOneDimensionNdarray(wind_speed).astype(np.float32)
        self.wind_speed_std = IntFloatConstructedOneDimensionNdarray(wind_speed_std).astype(np.float32)
        self.original_resolution = original_resolution

    def simulate_transient_wind_speed_time_series(self,
                                                  resolution: int,
                                                  traces_number_for_each_recording: int, *,
                                                  mode: str = 'cross sectional',
                                                  sigma_func: Callable):
        """
        To simulate the transient wind speed in original_resolution [seconds]
        :param resolution: the resolution of simulated transient time series [seconds]

        :param traces_number_for_each_recording: how many times series needed

        :param mode: can be either "time series" or "cross sectional"
        Both will use MCMC, as wind speed time series must be auto-correlated. The difference are as following:

        "time series":For index i, simulate samples according to wind_speed[i] and wind_speed_std[i]. And a further
        careful step is to use the last samples in the i-th simulation as the initial points for (i+1)-th simulation.

        "cross sectional": wind_speed[i] and wind_speed[i+1], wind_speed_std[i] and wind_speed_std[i+1], are not
        correlated. This is the common situation when doing Pout-WS scatter plot analysis. In this case, the last
        samples in the i-th simulation will be not initial points for (i+1)-th simulation.

        :param sigma_func: A function that can specify the proposal distribution sigma

        :return: Iterator, __next__ return is ndarray
         axis 0 is the number of traces
         axis 1 is the transient time step
        """
        original_resolution = self.original_resolution

        class HighResolWindGenerator:
            __slots__ = ('i', 'previous_trace')
            recording_distributions = self.transient_distribution
            recording_length = self.wind_speed.shape[0]

            def __init__(self):
                self.i = 0
                self.previous_trace = None

            def __len__(self):
                return self.recording_length

            def __iter__(self):
                self.i = 0  # reset，以便以后继续能iter
                self.previous_trace = None
                return self

            def __next__(self) -> ndarray:
                try:
                    if self.i == self.recording_length:
                        raise IndexError
                    this_recording_distribution = self.recording_distributions[self.i]
                    # %% determine the initial value for MCMC
                    if (self.i == 0) or (mode == 'cross sectional'):
                        current_state = tf.fill(traces_number_for_each_recording, this_recording_distribution.loc)
                    elif mode == 'time series':
                        current_state = self.previous_trace[:, -1]
                    else:
                        raise ValueError("'mode' should be either 'cross sectional' or 'time series'")
                    num_of_results = int(original_resolution / resolution)

                    # define proposal func
                    def custom_random_walk_normal_fn():
                        def _fn(state_parts, seed):
                            with tf.name_scope('custom_random_walk_normal_fn'):
                                scales = [sigma_func(x) for x in state_parts]
                                if len(scales) == 1:
                                    scales *= len(state_parts)
                                if len(state_parts) != len(scales):
                                    raise ValueError('`scale` must broadcast with `state_parts`.')
                                next_state_parts = [
                                    tf.random.normal(  # pylint: disable=g-complex-comprehension
                                        mean=state_part,
                                        stddev=scale_part,
                                        shape=tf.shape(state_part),
                                        dtype=dtype_util.base_dtype(state_part.dtype),
                                        seed=1
                                    )
                                    for scale_part, state_part in zip(scales, state_parts)
                                ]

                                return next_state_parts

                        return _fn

                    # define sample func
                    def sample_func():
                        # zero burn-in, as we want to be as near as current_state as possible
                        _this_trace = tfp.mcmc.sample_chain(num_results=num_of_results,
                                                            num_burnin_steps=0,
                                                            current_state=current_state,
                                                            kernel=tfp.mcmc.RandomWalkMetropolis(
                                                                this_recording_distribution.log_prob,
                                                                # new_state_fn=custom_random_walk_normal_fn()
                                                            ),
                                                            # kernel=tfp.mcmc.HamiltonianMonteCarlo(
                                                            #     this_recording_distribution.log_prob,
                                                            #     num_leapfrog_steps=2,
                                                            #     step_size=0.5),
                                                            trace_fn=None,
                                                            seed=1)
                        return _this_trace

                    # sample_func_faster = tf.function(sample_func, autograph=False, experimental_compile=True)
                    # this_trace = sample_func_faster().numpy().T

                    this_trace = sample_func().numpy().T

                    self.previous_trace = this_trace if mode == 'time series' else None
                    # For debug
                    # X, Y = this_trace[:50000, :-1].flatten(), this_trace[:50000, 1:].flatten()
                    # # scatter(X, Y)
                    # mob = MethodOfBins(X, Y, bin_step=0.1)
                    # # hist(mob.mob[int(mob.mob.__len__() / 2)]['dependent_var_in_this_bin'])
                    # temp = mob.cal_mob_statistic_eg_quantile(behaviour='new', statistic=None)
                    # series(temp.pd_view.loc['std.'])
                    self.i += 1
                except IndexError:
                    raise StopIteration
                return this_trace

        return HighResolWindGenerator()

    @property
    def transient_distribution(self):
        distribution = tfp.distributions.TruncatedNormal(loc=self.wind_speed,
                                                         scale=self.wind_speed_std,
                                                         low=0.,
                                                         high=70.)
        return distribution

    @staticmethod
    def learn_transition_by_looking_at_actual_high_resol() -> Callable:
        """
        This function is to look at the actual WS measurements from North Harris
        :return:
        """

        data_folder_path = Path(f"C:\\Users\\{getpass.getuser()}\\OneDrive\\PhD\\"
                                f"01-PhDProject\\Database\\Wind_and_NetworkData\\CD-ROM\\North harris")
        actual_high_resol_wind_speed = []
        for this_csv_path in list_all_specific_format_files_in_a_folder_path(data_folder_path, "CSV", ""):
            # Read, and find out how many years in this recording
            this_file_pd = pd.read_csv(this_csv_path, index_col=False)[['YEAR', 'DAY', 'HH', 'MM', 'SS', 'V50']]
            unique_year, unique_year_index, unique_year_counts = np.unique(
                this_file_pd['YEAR'].values, return_index=True, return_counts=True
            )
            # Simply calculate the time shift in seconds
            seconds = ((this_file_pd['DAY'].values - 1) * 24 * 3600 +  # The first day starts from 1
                       this_file_pd['HH'].values * 3600 +
                       this_file_pd['MM'].values * 60 +
                       this_file_pd['SS'].values)
            pd_timedelta = pd.to_timedelta(seconds, 'second')
            # Find starting of the year
            pd_date_time_index_meta = []
            for i in range(unique_year.__len__()):
                slice_i = slice(unique_year_index[i], unique_year_index[i] + unique_year_counts[i])
                pd_date_time_index = pd.to_datetime(
                    datetime.datetime(year=this_file_pd['YEAR'].iloc[unique_year_index[i]], month=1, day=1)
                )
                pd_date_time_index += pd_timedelta[slice_i]
                pd_date_time_index_meta.append(pd_date_time_index)
            # Convert back to get the pd.DateTimeIndex obj
            pd_date_time_index = pd.DatetimeIndex(np.concatenate(pd_date_time_index_meta))
            # Aggregate to 10 seconds
            this_reading_usable = this_file_pd['V50']
            this_reading_usable.index = pd_date_time_index
            this_reading_usable = this_reading_usable.resample("3S").mean()
            actual_high_resol_wind_speed.append(this_reading_usable)
        # Analysis through pure Numpy
        actual_high_resol_wind_speed = np.concatenate(actual_high_resol_wind_speed)
        # Method of bins
        mob = MethodOfBins(actual_high_resol_wind_speed[:-1],
                           actual_high_resol_wind_speed[1:], bin_step=0.1)
        sigma = mob.cal_mob_statistic_eg_quantile(behaviour="new", statistic=None)
        # Fit
        sigma_to_fit_x = sigma.columns.values
        sigma_to_fit_y = sigma.loc['std.'].values
        only_consider_smaller_than_33_index = sigma_to_fit_x < 33
        reg = LinearRegression().fit(sigma_to_fit_x[only_consider_smaller_than_33_index].reshape(-1, 1),
                                     sigma_to_fit_y[only_consider_smaller_than_33_index].reshape(-1, 1))

        # return func
        def func(x):
            x = np.array(IntFloatConstructedOneDimensionNdarray(x)).reshape(-1, 1)
            return reg.predict(x).flatten()

        return func


if __name__ == '__main__':
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
    this_wind_turbine = wind_turbines[0]
    rho = cal_air_density(
        celsius_to_kelvin(this_wind_turbine['environmental temperature'].values),
        this_wind_turbine['relative humidity'].values / 100,
        this_wind_turbine['barometric pressure'].values * 100
    )
    _func = Wind.learn_transition_by_looking_at_actual_high_resol()
    ws = np.arange(0, 30, 0.1)
    y = _func(ws)
    series(ws, y)
