from BivariateAnalysis_Class import MethodOfBins
from abc import ABCMeta, abstractmethod
from functools import wraps
from numpy import ndarray
from scipy.interpolate import interp1d
import numpy as np
from typing import Union, Tuple, Iterator, Iterable, Callable
from Ploting.fast_plot_Func import *
from project_utils import project_path_
from scipy.io import loadmat
import pandas as pd
from Data_Preprocessing import float_eps
import warnings
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray, UncertaintyDataFrame, StrOneDimensionNdarray, \
    OneDimensionNdarray
from typing import Iterable
from Ploting.adjust_Func import reassign_linestyles_recursively_in_ax, adjust_lim_label_ticks
from numba import jit, vectorize, float32, guvectorize, int32, boolean, int8
import tensorflow as tf
import numba as nb
import copy
import time
from collections import OrderedDict, ChainMap
from geneticalgorithm import geneticalgorithm as ga
import itertools
from inspect import Parameter, Signature
import inspect
from pathlib import Path
from File_Management.load_save_Func import save_pkl_file


class PowerCurve(metaclass=ABCMeta):
    cut_in_wind_speed = 4.  # [m/s]
    cut_out_wind_speed = 25.  # [m/s]
    restart_wind_speed = 20.  # [m/s]

    rated_active_power_output = 3000  # [kW]
    linestyle = ''  # type:str
    color = ''  # type:str
    label = ''  # type:str

    __slots__ = ('region_12_boundary', 'region_23_boundary', 'region_34_boundary', 'region_45_boundary')

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

    def plot(self, ws: ndarray = None, plot_region_boundary: bool = False, ax=None,
             mode='continuous', **kwargs):
        ax = self.__plot_region_boundary(ax) if plot_region_boundary else ax
        ws = ws if ws is not None else np.arange(0, 29.5, 0.01)
        active_power_output = self(ws)
        if mode == 'continuous':
            return series(ws, active_power_output, ax,
                          linestyle=self.linestyle, color=self.color, label=self.label,
                          y_lim=(-0.05, 1.05),
                          x_lim=(-0.05, 29.5),
                          x_label='Wind Speed [m/s]', y_label='Active Power Output [p.u.]',
                          **kwargs)
        elif mode == 'discrete':
            return scatter(ws, active_power_output, ax,
                           color='r', marker='+', s=32, label=self.label,
                           y_lim=(-0.05, 1.05),
                           x_lim=(-0.05, 29.5),
                           x_label='Wind Speed [m/s]', y_label='Active Power Output [p.u.]',
                           **kwargs)
        else:
            raise ValueError("'mode' should be either 'continuous' or 'discrete'")


class PowerCurveByMfr(PowerCurve):
    linestyle = '-.'
    color = (0.64, 0.08, 0.18)
    label = 'Mfr-PC'

    __slots__ = ('mfr_ws', 'mfr_p', 'air_density')

    def __init__(self,
                 air_density: Union[int, float, str] = None,
                 *, cut_in_ws: Union[int, float] = None):
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
                                      [cut_in_ws or mfr_pc_metadata.index.values[0] - float_eps * 2],
                                      mfr_pc_metadata.index.values,
                                      [25 + float_eps * 10],
                                      [np.inf]))
        self.mfr_p = np.concatenate(([0],
                                     [0],
                                     mfr_pc_metadata[air_density].values,
                                     [0],
                                     [0])) / self.rated_active_power_output
        self.air_density = air_density

    @classmethod
    def init_multiple_instances(cls, air_density: ndarray, **kwargs) -> tuple:
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
            this_instance = cls(this_air_density_str.__str__(), **kwargs)
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

    @classmethod
    def air_density_in_docs(cls):
        return '0.97', '1', '1.03', '1.06', '1.09', '1.12', '1.15', '1.18', '1.21', '1.225', '1.24', '1.27'

    @classmethod
    def init_all_instances_in_docs(cls, **kwargs):
        air_density_in_docs_float = np.array([float(x) for x in cls.air_density_in_docs()])
        return cls.init_multiple_instances(air_density_in_docs_float, **kwargs)


class PowerCurveByMethodOfBins(PowerCurve):
    linestyle = ':'
    color = 'red'
    label = 'MOB PC'

    __slots__ = ('wind_speed_recording', 'active_power_output_recording', 'power_curve_look_up_table')

    def __init__(self,
                 wind_speed_recording: ndarray,
                 active_power_output_recording: ndarray,
                 *, interp_for_high_resol: bool = True,
                 cal_region_boundary: bool):
        self.wind_speed_recording = wind_speed_recording
        self.active_power_output_recording = active_power_output_recording
        if interp_for_high_resol:
            self.power_curve_look_up_table = self.__cal_power_curve_look_up_table()
        else:
            self.power_curve_look_up_table = np.stack((self.wind_speed_recording,
                                                       self.active_power_output_recording), axis=1)
        # 只有在把outlier都去除了的条件下去计算region boundary才有意义，不然根本连rated都达不到，算法会有问题
        if cal_region_boundary:
            self.region_12_boundary, self.region_23_boundary, self.region_34_boundary, self.region_45_boundary = \
                self.cal_region_boundary()

    def __cal_power_curve_look_up_table(self) -> ndarray:
        """
        利用线性插值得到高精度(bin=0.05m/s)的power curve的查找表
        """
        core_ws = np.arange(0, 30, 0.05)
        power_curve_look_up_table_hi_resol = np.full((core_ws.size, 2), np.nan)
        power_curve_look_up_table_hi_resol[:, 0] = core_ws
        del core_ws
        power_curve_look_up_table = MethodOfBins(self.wind_speed_recording, self.active_power_output_recording,
                                                 bin_step=0.5,
                                                 first_bin_left_boundary=0,
                                                 last_bin_left_boundary=29.5).cal_mob_statistic_eg_quantile()
        power_curve_look_up_table_hi_resol[:, 1] = interp1d(
            np.concatenate((np.array([-100]), power_curve_look_up_table[:, 0], np.array([100]))),
            np.concatenate((np.array([0]), power_curve_look_up_table[:, 1], np.array([0]))))(
            power_curve_look_up_table_hi_resol[:, 0])
        return power_curve_look_up_table_hi_resol

    def __call__(self, ws: IntFloatConstructedOneDimensionNdarray) -> ndarray:
        ws = IntFloatConstructedOneDimensionNdarray(ws)
        return interp1d(np.concatenate((np.array([-100]), self.power_curve_look_up_table[:, 0], np.array([100]))),
                        np.concatenate((np.array([0]), self.power_curve_look_up_table[:, 1], np.array([0]))))(ws)

    def plot(self,
             ws: ndarray = None,
             plot_region_boundary: bool = False,
             ax=None,
             plot_recording: bool = True,
             mode='continuous',
             **kwargs):
        ax = scatter(self.wind_speed_recording, self.active_power_output_recording, ax=ax, color='royalblue',
                     alpha=0.5) if plot_recording else ax
        return super().plot(ws, plot_region_boundary, ax, mode=mode, **kwargs)


class PowerCurveFittedBy8PLF(PowerCurveByMethodOfBins):
    linestyle = '--'
    color = 'darkorange'
    label = '8PLF PC'
    ordered_params = ('a', 'd', 'b_1', 'c_1', 'g_1', 'b_2', 'c_2', 'g_2')

    __slots__ = ('a', 'd', 'b_1', 'c_1', 'g_1', 'b_2', 'c_2', 'g_2')

    @classmethod
    def init_from_power_curve_by_method_of_bins(cls,
                                                power_curve_by_method_of_bins_instance: PowerCurveByMethodOfBins):
        return cls(wind_speed_recording=power_curve_by_method_of_bins_instance.wind_speed_recording,
                   active_power_output_recording=power_curve_by_method_of_bins_instance.active_power_output_recording)

    def __init__(self,
                 wind_speed_recording: ndarray = None,
                 active_power_output_recording: ndarray = None,
                 *, interp_for_high_resol: bool = True,
                 **kwargs):
        super(PowerCurveFittedBy8PLF, self).__init__(wind_speed_recording=wind_speed_recording,
                                                     active_power_output_recording=active_power_output_recording,
                                                     interp_for_high_resol=interp_for_high_resol,
                                                     cal_region_boundary=False)
        self.update_params(**kwargs)

    @classmethod
    def call_static_func(cls) -> Callable:
        source = f"def static_func(x, {', '.join(cls.ordered_params)}):\n" \
                 f"    left_side = d + (a - d) / np.power(1 + np.power(x / c_1, b_1), g_1)\n" \
                 f"    right_side = d + (a - d) / np.power(1 + np.power(x / c_2, b_2), g_2)\n" \
                 f"    output = np.min(np.stack((left_side, right_side), axis=0), axis=0)\n" \
                 f"    return np.array(output)"
        exec(source, globals())

        return globals()['static_func']

    def __call__(self, ws: Union[IntFloatConstructedOneDimensionNdarray, ndarray]) -> ndarray:
        ws = IntFloatConstructedOneDimensionNdarray(ws)
        return self.call_static_func()(ws, *self.params)

    def _loss_func(self, wind_speed, *, focal_error=None) -> Callable:
        if wind_speed is None:
            wind_speed = self.power_curve_look_up_table[:, 0]
            target = self.power_curve_look_up_table[:, 1]
        else:
            target = super().__call__(wind_speed)

        def func(params_array):
            model_output = self.call_static_func()(wind_speed, *params_array)
            # focal loss
            error = model_output - target
            if focal_error:
                focal_index = np.abs(error) > focal_error
                if np.sum(focal_index) == 0:
                    focal_index = np.full(error.shape, True)
            else:
                focal_index = np.full(error.shape, True)
            return float(np.sqrt(np.mean(error[focal_index] ** 2)))

        return func

    @property
    def params(self) -> ndarray:
        params_list = []
        for this_param in self.ordered_params:
            exec(f"params_list.append(self.{this_param})")
        return np.array(params_list)

    def update_params(self, *args, **kwargs):
        for i, this_param in enumerate(self.ordered_params):
            if args.__len__() == self.ordered_params.__len__():
                exec(f"self.{this_param} = args[{i}]")
            else:
                exec(f"self.{this_param} = float({kwargs}.get('{this_param}')) "
                     f"if {kwargs}.get('{this_param}') is not None else None")

    @property
    def _params_constraints(self) -> OrderedDict:
        constraints = dict(
            [
                ('a', [0.9, 1.1]),
                ('d', [-0.1, 0.1]),
                ('b_1', [-120.0, 0]),
                ('b_2', [0, 120.0]),
                ('c_1', [float_eps, 30.0]),
                ('c_2', [float_eps, 30.0]),
                ('g_1', [float_eps, 30.0]),
                ('g_2', [float_eps, 30.0]),
            ]
        )
        constraints = OrderedDict([(this_param, constraints[this_param]) for this_param in self.ordered_params])
        return constraints

    def _params_init(self):
        for this_param in self.ordered_params:
            source_code = f"self.{this_param} = (self._params_constraints['{this_param}'][0] + " \
                          f"self._params_constraints['{this_param}'][-1]) / 2"
            exec(source_code)
        return self.params

    def fit(self, *, ga_algorithm_param: dict = None,
            params_init_scheme: str,
            run_n_times: int = 3,
            save_to_file_path: Path = None,
            wind_speed: Union[IntFloatConstructedOneDimensionNdarray, ndarray] = None,
            focal_error: Union[int, float] = None):
        """
        Use GA to fit the 8PL PC.
        :param ga_algorithm_param: The parameters of GA algorithm
        :param params_init_scheme: Can be either 'self' or 'average',
        'self' means to use current self.params to initialise GA;
        'average' means to use the average of boundaries of constraints to initialise GA

        :param run_n_times: run total run_n_times times
        :param save_to_file_path
        :param wind_speed: Only fit the curve for this wind speed
        :param focal_error
        :return:
        """
        # %% Prepare the parameters of GA algorithm and the initial values for fitting
        wind_speed = IntFloatConstructedOneDimensionNdarray(wind_speed)
        ga_algorithm_param = ga_algorithm_param or {}
        if params_init_scheme == 'self':
            if any([this_param is None for this_param in self.params]):
                raise Exception("To use 'self' in 'params_init_scheme', all params must not be None. "
                                "Otherwise, please use 'average'")
            initialised_params = self.params
        elif params_init_scheme == 'average':
            initialised_params = self._params_init()
        else:
            raise ValueError("'params_init_scheme' can either be 'self' or 'average'")

        # %% Check if any current 8PL PC param is not None
        if any([this_param is not None for this_param in self.params]):
            warnings.warn("Existing params may be overwritten", UserWarning)

        # %% Init GA obj
        default_algorithm_param = Signature.from_callable(ga.__init__).parameters['algorithm_parameters'].default
        ga_algorithm_param = dict(ChainMap(ga_algorithm_param, default_algorithm_param))
        ga_model = ga(function=self._loss_func(wind_speed, focal_error=focal_error),
                      dimension=self.params.__len__(),
                      variable_type='real',
                      variable_boundaries=np.array(list(self._params_constraints.values())),
                      algorithm_parameters=ga_algorithm_param)

        # %% Run GA and save the results
        ga_model_run_results = []
        for _ in range(run_n_times):
            ga_model.run(init_solo=initialised_params)
            ga_model_run_results.append(ga_model.output_dict)
            initialised_params = ga_model.best_variable  # This will initialise the next GA using the current best
            if save_to_file_path:
                save_pkl_file(save_to_file_path, ga_model_run_results)

        # %% Update 8PL PC
        for this_param_index, this_param in enumerate(self.ordered_params):
            source_code = f"self.{this_param} = ga_model.best_variable[{this_param_index}]"  # The last is the best
            exec(source_code)

        return ga_model_run_results

    def plot(self, *,
             ws: ndarray = None,
             plot_region_boundary: bool = False,
             ax=None,
             mode='continuous',
             plot_recordings_and_mob: bool = True,
             **kwargs):
        if all((self.wind_speed_recording is not None,
                self.active_power_output_recording is not None,
                plot_recordings_and_mob)):
            passed_sig = Signature.from_callable(PowerCurveByMethodOfBins.plot).parameters.keys()
            safe_locals = locals()
            passed_args = {key: safe_locals[key] for key in passed_sig if (key != 'self' and key != 'kwargs')}

            ax = PowerCurveByMethodOfBins(wind_speed_recording=self.wind_speed_recording,
                                          active_power_output_recording=self.active_power_output_recording,
                                          cal_region_boundary=False).plot(**passed_args)
        else:
            ax = ax
        return super(PowerCurveByMethodOfBins, self).plot(ws, plot_region_boundary, ax, mode=mode, **kwargs)


class PowerCurveFittedBy5PLF(PowerCurveFittedBy8PLF):
    linestyle = '--'
    color = 'darkorange'
    label = '5PLF PC'
    ordered_params = ('a', 'd', 'b', 'c', 'g')

    __slots__ = ('a', 'd', 'b', 'c', 'g')

    @property
    def _params_constraints(self) -> OrderedDict:
        constraints = dict(
            [
                ('a', [1 - float_eps, 1 + float_eps]),
                ('d', [0 - float_eps, 0 + float_eps]),
                ('b', [-20.0, -5.]),
                ('c', [10., 15.]),
                ('g', [float_eps, 0.5]),
            ]
        )
        constraints = OrderedDict([(this_param, constraints[this_param]) for this_param in self.ordered_params])
        return constraints

    @classmethod
    def call_static_func(cls) -> Callable:
        source = f"def static_func(x, {', '.join(cls.ordered_params)}):\n" \
                 f"    output = d + (a - d) / np.power(1 + np.power(x / c, b), g)\n" \
                 f"    return np.array(output)"
        exec(source, globals())

        return globals()['static_func']


if __name__ == '__main__':
    cc = PowerCurveByMfr.init_multiple_instances(np.array([1.1, 1.2, 1.1]))
