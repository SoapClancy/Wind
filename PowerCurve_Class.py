from BivariateAnalysis_Class import MethodOfBins
from abc import ABCMeta, abstractmethod
from functools import wraps
from numpy import ndarray
from scipy.interpolate import interp1d
import numpy as np
from typing import Union, Tuple, Iterator, Iterable
from Ploting.fast_plot_Func import *
from project_path_Var import project_path_
from scipy.io import loadmat
import pandas as pd
from Data_Preprocessing import float_eps
import warnings
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray, UncertaintyDataFrame, StrOneDimensionNdarray
from typing import Iterable
# from Wind_Class import Wind
from Ploting.adjust_Func import reassign_linestyles_recursively_in_ax, adjust_lim_label_ticks
from numba import jit, vectorize, float32, guvectorize, int32, boolean, int8, bool_
import tensorflow as tf
import numba as nb
import copy
import time


class PowerCurve(metaclass=ABCMeta):
    cut_in_wind_speed = 4  # [m/s]
    cut_out_wind_speed = 25  # [m/s]
    restart_wind_speed = 20  # [m/s]

    rated_active_power_output = 3000  # [kW]
    linestyle = ''  # type:str
    color = ''  # type:str
    label = ''  # type:str

    __slots__ = ('region_12_boundary', 'region_23_boundary', 'region_34_boundary', 'region_45_boundary',
                 'power_curve_look_up_table')

    @abstractmethod
    def __call__(self, ws: Union[Iterable, float, int]):
        pass

    def cal_region_boundary(self, tolerance_factor: float = 0.005) -> Tuple[float, float, float, float]:
        # region的边界可能的范围
        tolerance = self.rated_active_power_output * tolerance_factor
        region_12_boundary_est = np.arange(2, 6, 0.01)
        region_23_boundary_est = np.arange(15, 21, 0.01)
        region_34_boundary_est = np.arange(22, 28, 0.01)
        region_45_boundary_est = np.arange(22, 28, 0.01)
        # 计算region的边界，注意经验power curve，如PowerCurveByMethodOfBins可能因为在指定区间内
        # 不是非单调增或非单调减函数而导致误差
        region_12_boundary = region_12_boundary_est[np.where(
            (self(region_12_boundary_est) - tolerance) > 0)[0][0]]
        region_23_boundary = region_23_boundary_est[np.where(
            (self(region_23_boundary_est) - (
                    self.rated_active_power_output - tolerance)) > 0)[0][0]]
        try:
            region_34_boundary = region_34_boundary_est[np.where(
                (self(region_34_boundary_est) - (
                        self.rated_active_power_output - tolerance)) < 0)[0][0]]
        except IndexError:
            region_34_boundary = np.nan
        try:
            region_45_boundary = region_45_boundary_est[np.where(
                (self(region_45_boundary_est) - tolerance) < 0)[0][0]]
        except IndexError:
            region_45_boundary = np.nan
        return region_12_boundary, region_23_boundary, region_34_boundary, region_45_boundary

    @staticmethod
    def cal_region_boundary_mask(boundary_like: ndarray, wind_speed_data: ndarray,
                                 normal_data_mask: ndarray = True) -> Tuple[ndarray, ...]:
        model_a_mask = np.bitwise_and(wind_speed_data >= boundary_like[0], wind_speed_data < boundary_like[1])
        model_b_mask = np.bitwise_and(wind_speed_data >= boundary_like[2], wind_speed_data < boundary_like[3])
        region_1_mask = wind_speed_data < boundary_like[0]
        region_rated_mask = np.bitwise_and(wind_speed_data >= boundary_like[1], wind_speed_data < boundary_like[2])
        region_5_mask = wind_speed_data >= boundary_like[3]
        hard_rated_mask = np.bitwise_and(wind_speed_data >= boundary_like[1],
                                         wind_speed_data < 25)
        hard_cut_off_mask = wind_speed_data >= 25
        return (np.bitwise_and(region_1_mask, normal_data_mask),
                np.bitwise_and(model_a_mask, normal_data_mask),
                np.bitwise_and(region_rated_mask, normal_data_mask),
                np.bitwise_and(model_b_mask, normal_data_mask),
                np.bitwise_and(region_5_mask, normal_data_mask),
                np.bitwise_and(hard_rated_mask, normal_data_mask),
                np.bitwise_and(hard_cut_off_mask, normal_data_mask))

    def __plot_region_boundary(self, ax=None):
        ax = vlines(self.region_12_boundary, ax)
        ax = vlines(self.region_23_boundary, ax)
        ax = vlines(self.region_34_boundary, ax)
        return vlines(self.region_45_boundary, ax)

    def plot_power_curve(self, ws: ndarray = None, plot_region_boundary: bool = False, ax=None,
                         mode='continuous', **kwargs):
        ax = self.__plot_region_boundary(ax) if plot_region_boundary else ax
        ws = ws if ws is not None else np.arange(0, 28.5, 0.01)
        active_power_output = self(ws)
        if mode == 'continuous':
            return series(ws, active_power_output, ax,
                          linestyle=self.linestyle, color=self.color, label=self.label,
                          y_lim=(-self.rated_active_power_output * 0.013, self.rated_active_power_output * 1.013),
                          x_lim=(0, 28.6),
                          x_label='Wind speed [m/s]', y_label='Active power output [kW]',
                          **kwargs)
        elif mode == 'discrete':
            return scatter(ws, active_power_output, ax,
                           color='r', marker='+', s=32, label=self.label,
                           y_lim=(-self.rated_active_power_output * 0.013, self.rated_active_power_output * 1.013),
                           x_lim=(0, 28.6),
                           x_label='Wind speed [m/s]', y_label='Active power output [kW]',
                           **kwargs)
        else:
            raise ValueError("'mode' should be either 'continuous' or 'discrete'")


class PowerCurveByMfr(PowerCurve):
    linestyle = '-.'
    color = (0.64, 0.08, 0.18)
    label = 'Mfr-PC'

    __slots__ = ('mfr_ws', 'mfr_p', 'air_density')

    def __init__(self, air_density: Union[int, float, str] = None):
        if air_density is None:
            warnings.warn("'air_density' unspecified. Set air_density to 1.15."
                          "But this may lead to unrealistic manufacturer power curve."
                          "User should always calculate the air density value at first!",
                          UserWarning)
            air_density = '1.15'
        if type(air_density) not in (int, float, str):
            raise TypeError("'air_density' of PowerCurveByMfr instance initialisation function should be "
                            "type int or type float or type str")
        if not isinstance(air_density, str):
            air_density = self.__infer_str_air_density(air_density)
        mfr_pc_metadata = self.make_mfr_pc_metadata()
        self.mfr_ws = np.concatenate(([-np.inf],
                                      [mfr_pc_metadata.index.values[0] - float_eps * 2],
                                      mfr_pc_metadata.index.values,
                                      [25 + float_eps * 10],
                                      [np.inf]))
        self.mfr_p = np.concatenate(([0],
                                     [0],
                                     mfr_pc_metadata[air_density].values,
                                     [0],
                                     [0]))
        self.air_density = air_density

    @classmethod
    def init_multiple_instances(cls, air_density: ndarray) -> tuple:
        """
        Initialise multiple PowerCurveByMfr instances，
        suitable for long period of analysis, as the air density will change
        :param air_density: ndarray, and dtype should be int or float
        :return: Tuple[PowerCurveByMfr, ...] remember, the same instances are copies 同一内存地址！
        if the air density strings are the same, the instances share the same memory
        """
        if not isinstance(air_density, ndarray):
            raise TypeError("To initialise multiple PowerCurveByMfr instances at the same time, "
                            "'air_density' must be type ndarray")
        if air_density.dtype not in (int, float):
            raise TypeError("the elements of 'air_density' must be type int or type float")
        # %% to get the unique strings indicating air density
        air_density_str_tuple = cls.__infer_str_air_density(air_density)
        air_density_str_unique, air_density_str_unique_inverse = np.unique(air_density_str_tuple, return_inverse=True)
        # %% initialise the ndarray storing the PowerCurveByMfr objs
        multiple_instances = np.empty(air_density.shape, dtype=object)
        # %% initialise the instances, but, remember, the same instances are copies,
        # i.e., if the air density strings are the same, the instances share the same memory
        for i, this_air_density_str in enumerate(air_density_str_unique):
            this_instance = cls(this_air_density_str.__str__())
            multiple_instances[air_density_str_unique_inverse == i] = this_instance
        return tuple(multiple_instances)  # type Tuple[PowerCurveByMfr, ...]

    def __str__(self):
        return ' rho='.join((self.label, self.air_density))

    def __call__(self, ws: Union[Iterable, float, int]):
        ws = IntFloatConstructedOneDimensionNdarray(ws)
        return interp1d(self.mfr_ws, self.mfr_p)(ws)

    def cal_with_hysteresis_control_using_high_resol_wind(
            self,
            high_resol_wind,
            mode: str,
            return_percentiles=None
    ) -> Union[ndarray, pd.DataFrame]:
        """
        TO calculate the Pout considering hysteresis and restart wind speed

        :param high_resol_wind: type Iterator
        should be the return of Wind (in Wind_Class.py) instance method "simulate_transient_wind_speed_time_series"
        axis 0 is the number of traces
        axis 1 is the transient time step

        :param mode: can be either "time series" or "cross sectional"
        refer to Wind (in Wind_Class.py) instance method "simulate_transient_wind_speed_time_series"

        :param return_percentiles: type StrOneDimensionNdarray
        If specified, then return to a pd.DataFrame instances, whose index is determined by return_percentiles

        :return: simulated distribution of Pout
        for ndarray:
        axis 0 is the No. (i.e., the index, or corresponding position) of wind speed or wind_speed_std
        axis 1 is the number of traces (re-average)
        for pd.DataFrame:
        index is return_percentiles,
        column is the No. (i.e., the index, or corresponding position) of wind speed or wind_speed_std
        """
        if return_percentiles is not None:
            return_percentiles = StrOneDimensionNdarray(return_percentiles)
            return_percentiles = UncertaintyDataFrame(
                index=np.append(return_percentiles, 'mean'),
                columns=range(len(high_resol_wind))
            )
        # Numba type should be determined so it can compile. Dynamic is not allowed
        numba_var_mfr_ws, numba_var_mfr_p = self.mfr_ws, self.mfr_p
        numba_var_cut_out_wind_speed = self.cut_out_wind_speed
        numba_var_restart_wind_speed = self.restart_wind_speed

        @guvectorize([(float32[:], boolean[:], float32[:], boolean[:])],
                     '(n), (n)->(n), (n)', nopython=True, target='parallel')
        def perform_calculation(x, control_signal_input, y, control_signal_output):
            """
            Impressive! Very high performance http://numba.pydata.org/numba-doc/latest/user/vectorize.html#guvectorize
            I have also tried jit(nopython=True, fastmath=True, parallel=True), but it is not as fast as this
            """
            y[:] = np.interp(x, numba_var_mfr_ws, numba_var_mfr_p)
            # Need to look at each WS inside a trace.
            # Calculate hysteresis signal, 'False' means no re-control,
            # 'True' means stay shut-down until below restart wind speed
            for this_wind_speed_index, this_wind_speed in enumerate(x):
                if this_wind_speed > numba_var_cut_out_wind_speed:
                    control_signal_input[this_wind_speed_index:] = True
                elif this_wind_speed < numba_var_restart_wind_speed:
                    control_signal_input[this_wind_speed_index:] = False
            # apply control
            y[control_signal_input] = 0.
            control_signal_output[:] = control_signal_input[:]

        results = []
        # hysteresis_control_signal = None
        hysteresis_control_signal = None
        for this_high_resol_wind_index, this_high_resol_wind in enumerate(high_resol_wind):
            # Initialise the hysteresis_control_signal
            if (this_high_resol_wind_index == 0) or (mode == 'cross sectional'):
                hysteresis_control_signal = np.full(this_high_resol_wind.shape, False)
            elif mode == 'time series':
                # Only use the last control signals as init
                hysteresis_control_signal = np.repeat(hysteresis_control_signal[:, [-1]],
                                                      hysteresis_control_signal.shape[-1],
                                                      axis=1)
            else:
                raise ValueError("'mode' should be either 'cross sectional' or 'time series'")

            this_high_resol_pout_results, hysteresis_control_signal = perform_calculation(
                this_high_resol_wind,
                hysteresis_control_signal
            )
            this_high_resol_pout_results = np.mean(this_high_resol_pout_results, axis=1)
            if return_percentiles is not None:
                # The reason not using @guvectorize decorating np.percentile is it will actually be slower...
                return_percentiles[this_high_resol_wind_index] = np.concatenate(
                    (np.percentile(this_high_resol_pout_results,
                                   return_percentiles.index.values[:-1].astype(float)),
                     np.mean(this_high_resol_pout_results, keepdims=True))
                )
            else:
                results.append(this_high_resol_pout_results)
        if return_percentiles is not None:
            # Must save as pd.DataFrame for stability. As UncertaintyDataFrame may change frequently
            return pd.DataFrame(return_percentiles)
        else:
            return np.array(results)

    @staticmethod
    def __infer_str_air_density(air_density: Union[int, float, ndarray]) -> Union[str, Tuple[str, ...]]:
        candidates_as_columns = PowerCurveByMfr.make_mfr_pc_metadata().columns
        candidates = candidates_as_columns.values.astype(float)
        if not isinstance(air_density, ndarray):
            index = np.argmin(np.abs(candidates - air_density))
            return candidates_as_columns[index]
        else:
            str_tuple = []
            for this_air_density in air_density:
                index = np.argmin(np.abs(candidates - this_air_density))
                str_tuple.append(candidates_as_columns[index])
            return tuple(str_tuple)

    @staticmethod
    def make_mfr_pc_metadata() -> pd.DataFrame:
        """
        生成初始化PowerCurveByMfr必须要有的元数据，包括mfr_ws和mfr_p。
        数据会以pd.DataFrame的形式储存，行索引是WS [m/s]，int类型
        列名是air density [kg/m^3], str类型，
        元素是Pout [kW]，float类型

        :return: pd.DataFrame
        """
        metadata = pd.DataFrame(
            index=range(4, 26),
            data={
                '0.97': np.array(
                    [
                        53, 142, 271, 451, 691,
                        995, 1341, 1686, 2010, 2310,
                        2588, 2815, 2943, 2988, 2998,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1': np.array(
                    [
                        56, 148, 281, 466, 714,
                        1028, 1385, 1740, 2074, 2382,
                        2662, 2868, 2965, 2994, 2999,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.03': np.array(
                    [
                        59, 153, 290, 482, 737,
                        1061, 1428, 1794, 2137, 2455,
                        2730, 2909, 2979, 2997, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.06': np.array(
                    [
                        61, 159, 300, 497, 760,
                        1093, 1471, 1849, 2201, 2525,
                        2790, 2939, 2988, 2998, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.09': np.array(
                    [
                        64, 165, 310, 512, 783,
                        1126, 1515, 1903, 2265, 2593,
                        2841, 2960, 2993, 2999, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.12': np.array(
                    [
                        67, 170, 319, 528, 806,
                        1159, 1558, 1956, 2329, 2658,
                        2883, 2975, 2996, 3000, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.15': np.array(
                    [
                        70, 176, 329, 543, 829,
                        1191, 1602, 2010, 2392, 2717,
                        2915, 2984, 2998, 3000, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.18': np.array(
                    [
                        72, 181, 339, 558, 852,
                        1224, 1645, 2064, 2454, 2771,
                        2940, 2990, 2999, 3000, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.21': np.array(
                    [
                        75, 187, 348, 574, 875,
                        1257, 1688, 2118, 2514, 2817,
                        2958, 2994, 2999, 3000, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.225': np.array(
                    [
                        77, 190, 353, 581, 886,
                        1273, 1710, 2145, 2544, 2837,
                        2965, 2995, 3000, 3000, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.24': np.array(
                    [
                        78, 193, 358, 589, 898,
                        1289, 1732, 2172, 2573, 2856,
                        2971, 2996, 3000, 3000, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
                '1.27': np.array(
                    [
                        81, 198, 368, 604, 921,
                        1322, 1775, 2226, 2628, 2889,
                        2981, 2998, 3000, 3000, 3000,
                        3000, 3000, 3000, 3000, 3000, 3000, 3000,
                    ], dtype=float),
            })
        return metadata


class PowerCurveByMethodOfBins(PowerCurve):
    __slots__ = ('wind_speed_recording', 'active_power_output_recording')

    def __init__(self, wind_speed_recording: ndarray, active_power_output_recording: ndarray, linestyle=':',
                 color='#00ffff', label='Scatters PC', *, cal_region_boundary: bool = True):
        super().__init__(linestyle, color, label)
        self.wind_speed_recording = wind_speed_recording
        self.active_power_output_recording = active_power_output_recording
        self.power_curve_look_up_table = self.__cal_power_curve_look_up_table()
        # 只有在把outlier都去除了的条件下去计算region boundary才有意义，不然根本连rated都达不到，算法会有问题
        if cal_region_boundary:
            self.region_12_boundary, self.region_23_boundary, self.region_34_boundary, self.region_45_boundary = \
                self.cal_region_boundary()

    def __cal_power_curve_look_up_table(self) -> ndarray:
        """
        利用线性插值得到高精度(bin=0.05m/s)的power curve的查找表
        """
        power_curve_look_up_table_hi_resol = np.arange(0, 30, 0.05).reshape(-1, 1)
        power_curve_look_up_table_hi_resol = np.stack(np.broadcast(power_curve_look_up_table_hi_resol, np.nan), axis=0)
        power_curve_look_up_table = MethodOfBins(self.wind_speed_recording, self.active_power_output_recording,
                                                 bin_step=0.5,
                                                 first_bin_left_boundary=0,
                                                 last_bin_left_boundary=28.5).cal_mob_statistic()
        power_curve_look_up_table_hi_resol[:, 1] = interp1d(
            np.concatenate((np.array([-100]), power_curve_look_up_table[:, 0], np.array([100]))),
            np.concatenate((np.array([0]), power_curve_look_up_table[:, 1], np.array([0]))))(
            power_curve_look_up_table_hi_resol[:, 0])
        return power_curve_look_up_table_hi_resol

    def __call__(self, ws: Union[Iterable, float, int]) -> ndarray:
        if not isinstance(ws, ndarray):
            ws = np.array([ws * 1.0])
        ws = ws.flatten()
        return interp1d(np.concatenate((np.array([-100]), self.power_curve_look_up_table[:, 0], np.array([100]))),
                        np.concatenate((np.array([0]), self.power_curve_look_up_table[:, 1], np.array([0]))))(ws)

    def plot_power_curve(self, ws: ndarray = None, plot_region_boundary: bool = True, plot_recording: bool = True):
        ax = scatter(self.wind_speed_recording, self.active_power_output_recording,
                     alpha=0.1) if plot_recording else None
        return super().plot_power_curve(ws, plot_region_boundary, ax)


if __name__ == '__main__':
    cc = PowerCurveByMfr.init_multiple_instances(np.array([1.1, 1.2, 1.1]))
