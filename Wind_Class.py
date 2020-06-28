from NdarraySubclassing import IntFloatConstructedOneDimensionNdarray
from Ploting.fast_plot_Func import *
import tensorflow_probability as tfp
from PowerCurve_Class import PowerCurveByMfr
import pandas as pd


class Wind:
    __slots__ = ('wind_speed', 'wind_speed_std')

    def __init__(self,
                 wind_speed: Union[int, float, ndarray],
                 wind_speed_std: Union[int, float, ndarray]):
        self.wind_speed = IntFloatConstructedOneDimensionNdarray(wind_speed)
        self.wind_speed_std = IntFloatConstructedOneDimensionNdarray(wind_speed_std)

    def simulate_transient_wind_speed_time_series(self,
                                                  resolution: int,
                                                  traces: int,
                                                  original_resolution=600) -> ndarray:
        """
        To simulate the transient wind speed in original_resolution [seconds]
        :param resolution: the resolution of simulated transient time series [seconds]
        :param traces: how many times series needed
        :param original_resolution original resolution of the recording, default is 600 [seconds], as suggested by IEC
        :return: ndarray,
         axis 0 is the number of traces,
         axis 1 is the transient time step
         axis 2 is the No. (i.e., the index, or corresponding position) of wind speed or wind_speed_std
        """
        return self.transient_distribution.sample((traces, int(original_resolution / resolution))).numpy()

    @property
    def transient_distribution(self):
        distribution = tfp.distributions.TruncatedNormal(loc=self.wind_speed,
                                                         scale=self.wind_speed_std,
                                                         low=0,
                                                         high=np.inf)
        return distribution

    def show_possible_resulted_power_output_range(self,
                                                  mfr_pc: PowerCurveByMfr,
                                                  wind_information:pd.DataFrame):
        pass


if __name__ == "__main__":
    wind = Wind(24.9, 4.7)
    sample = wind.simulate_transient_wind_speed_time_series(10, 1000)
