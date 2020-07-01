from NdarraySubclassing import IntFloatConstructedOneDimensionNdarray
from Ploting.fast_plot_Func import *
import tensorflow as tf
import tensorflow_probability as tfp
from PowerCurve_Class import PowerCurveByMfr
import pandas as pd
from initialise_wt_or_wf_Func import load_raw_wt_from_txt_file_and_temperature_from_csv
from BivariateAnalysis_Class import MethodOfBins


class Wind:
    __slots__ = ('wind_speed', 'wind_speed_std')

    def __init__(self,
                 wind_speed: Union[int, float, ndarray],
                 wind_speed_std: Union[int, float, ndarray]):
        self.wind_speed = IntFloatConstructedOneDimensionNdarray(wind_speed).astype(np.float32)
        self.wind_speed_std = IntFloatConstructedOneDimensionNdarray(wind_speed_std).astype(np.float32)

    def simulate_transient_wind_speed_time_series(self,
                                                  resolution: int,
                                                  traces_number_for_each_recording: int,
                                                  original_resolution=600, *,
                                                  mode: str) -> ndarray:
        """
        To simulate the transient wind speed in original_resolution [seconds]
        :param resolution: the resolution of simulated transient time series [seconds]

        :param traces_number_for_each_recording: how many times series needed

        :param original_resolution original resolution of the recording, default is 600 [seconds], as suggested by IEC
        Standard 61400-12-1

        :param mode: Optional, which can be either "time series" or "cross sectional"
        Both will use MCMC, as wind speed time series must be auto-correlated. The difference are as following:

        "time series":For index i, simulate samples according to wind_speed[i] and wind_speed_std[i]. And a further
        careful step is to use the last samples in the i-th simulation as the initial points for (i+1)-th simulation.

        "cross sectional": wind_speed[i] and wind_speed[i+1], wind_speed_std[i] and wind_speed_std[i+1], are not
        correlated. This is the common situation when doing Pout-WS scatter plot analysis. In this case, the last
        samples in the i-th simulation will be not initial points for (i+1)-th simulation.

        :return: ndarray,
         axis 0 is the No. (i.e., the index, or corresponding position) of wind speed or wind_speed_std
         axis 1 is the number of traces
         axis 2 is the transient time step
        """
        traces = []
        for i in range(self.wind_speed.shape[0]):
            # # TODO DEBUG
            # if i not in (22, 40):
            #     continue
            this_recording_distribution = self.transient_distribution[i]
            # %% determine the initial value for MCMC
            if (i == 0) or mode == 'cross sectional':
                current_state = tf.fill(traces_number_for_each_recording, this_recording_distribution.loc)
            elif mode == 'time series':
                current_state = traces[-1][-1, :]
            else:
                raise ValueError("'mode' should be either 'cross sectional' or 'time series'")
            num_of_results = int(original_resolution / resolution)

            def sample_func():
                # zero burn-in, as we want to be as near as current_state as possible
                _this_trace = tfp.mcmc.sample_chain(num_results=num_of_results,
                                                    num_burnin_steps=0,
                                                    current_state=current_state,
                                                    kernel=tfp.mcmc.HamiltonianMonteCarlo(
                                                        this_recording_distribution.log_prob,
                                                        num_leapfrog_steps=2,
                                                        step_size=0.5),
                                                    trace_fn=None)
                return _this_trace

            sample_func_faster = tf.function(sample_func, autograph=False, experimental_compile=True)
            this_trace = sample_func_faster().numpy().T
            traces.append(this_trace)

        return np.array(traces)

    @property
    def transient_distribution(self):
        distribution = tfp.distributions.TruncatedNormal(loc=self.wind_speed,
                                                         scale=self.wind_speed_std,
                                                         low=0,
                                                         high=np.inf)
        return distribution
