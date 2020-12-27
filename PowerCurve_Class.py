from __future__ import annotations
from abc import ABCMeta, abstractmethod
from scipy.interpolate import interp1d
from typing import Tuple, Callable
from Ploting.fast_plot_Func import *
import pandas as pd
from Data_Preprocessing import float_eps
import warnings
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray, UncertaintyDataFrame, IntOneDimensionNdarray
from typing import Iterable
from numba import float32, guvectorize, boolean
import copy
from collections import OrderedDict, ChainMap
from geneticalgorithm import geneticalgorithm as ga
from inspect import Signature
from File_Management.load_save_Func import save_pkl_file, load_pkl_file
from project_utils import *
from Filtering.OutlierAnalyser_Class import DataCategoryData, DataCategoryNameMapper
from tqdm import tqdm
from PhysicalInstance_Class import PhysicalInstanceDataFrame
from parse import parse
from ErrorEvaluation_Class import DeterministicError
from Ploting.adjust_Func import *
from ConvenientDataType import IntFloatConstructedOneDimensionNdarray, StrOneDimensionNdarray, OneDimensionNdarray
from parse import *


class PowerCurve(metaclass=ABCMeta):
    cut_in_wind_speed = 4.  # [m/s]
    cut_out_wind_speed = 25.  # [m/s]
    restart_wind_speed = 20.  # [m/s]

    __slots__ = ('region_12_boundary', 'region_23_boundary', 'region_34_boundary', 'region_45_boundary',
                 'label', 'color', 'linestyle', 'rated_active_power_output')

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
        try:
            region_23_boundary = region_23_boundary_est[np.where(
                (self(region_23_boundary_est) - (
                        self.rated_active_power_output - tolerance)) > 0)[0][0]]
        except IndexError:
            region_23_boundary = region_23_boundary_est[np.argmax(self(region_23_boundary_est))]
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
        ws = ws if ws is not None else np.arange(float_eps, 40.5, 0.01)
        active_power_output = self(ws)
        kwargs = dict(ChainMap(kwargs,
                               {'linestyle': self.linestyle,
                                'color': self.color,
                                'label': self.label},
                               WS_POUT_2D_PLOT_KWARGS))
        if mode == 'continuous':
            return series(ws, active_power_output, ax, **kwargs)
        elif mode == 'discrete':
            return scatter(ws, active_power_output, ax, **kwargs)
        else:
            raise ValueError("'mode' should be either 'continuous' or 'discrete'")


class PowerCurveByMfr(PowerCurve):
    __slots__ = ('mfr_ws', 'mfr_p', 'air_density', 'label')

    def __init__(self,
                 air_density: Union[int, float, str] = None,
                 *, color=(0.64, 0.08, 0.18),
                 linestyle='-.', rated_active_power_output: Union[int, float] = 3000):
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
            air_density = self.infer_str_air_density(air_density)

        self.rated_active_power_output = rated_active_power_output

        mfr_pc_metadata = self.make_mfr_pc_metadata()
        self.mfr_ws = np.concatenate(([-np.inf],
                                      [mfr_pc_metadata.index.values[0] - 1],
                                      mfr_pc_metadata.index.values,
                                      [(25 if air_density != 'special' else 100) + float_eps * 10],
                                      [np.inf]))
        self.mfr_ws[self.mfr_ws == 0] = float_eps
        self.mfr_p = np.concatenate(
            ([0.],
             [0.],
             mfr_pc_metadata[air_density if air_density != 'special' else '1.12'].values,
             [0. if air_density != 'special' else self.rated_active_power_output],
             [0. if air_density != 'special' else self.rated_active_power_output])
        ) / self.rated_active_power_output
        self.air_density = air_density
        # self.label = 'Mfr PC\n(' + r'$\rho$' + f'={self.air_density} kg/m' + '$^3$' + ')'
        # self.label = 'Mfr PC (' + r'$\rho$' + f'={self.air_density} kg/m' + '$^3$)'
        self.label = 'Mfr PC ' + r'$\rho$' + '$_{' + f'{self.air_density}' + '}$'
        self.color = color
        self.linestyle = linestyle

    @classmethod
    def init_from_custom_wind_speed_and_power_data_sheet(cls, *,
                                                         ws: ndarray, power: ndarray, **kwargs) -> PowerCurveByMfr:
        intermittent_obj = cls.__new__(cls)
        intermittent_obj.mfr_ws = ws
        intermittent_obj.mfr_p = power
        intermittent_obj.air_density = "None"
        intermittent_obj.label = kwargs.get("label", "Custom PC")
        intermittent_obj.color = kwargs.get("color", (0.64, 0.08, 0.18))
        intermittent_obj.linestyle = kwargs.get("linestyle", '-.')
        intermittent_obj.rated_active_power_output = kwargs.get("rated_active_power_output", 3000)
        return intermittent_obj

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
        if (air_density.dtype not in (int, float)) and (not pd.api.types.is_string_dtype(air_density)):
            raise TypeError("the elements of 'air_density' must be type int, float, or str ")
        # %% to get the unique strings indicating air density
        if pd.api.types.is_string_dtype(air_density):
            air_density_str_tuple = air_density
        else:
            air_density_str_tuple = cls.infer_str_air_density(air_density)
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
        ws[ws == 0] = float_eps
        return interp1d(self.mfr_ws, self.mfr_p)(ws)

    def cal_with_hysteresis_control_using_high_resol_wind(
            self,
            high_resol_wind,
            mode: str = 'cross sectional',
            return_percentiles: UncertaintyDataFrame = None
    ) -> Union[ndarray, pd.DataFrame]:
        """
        TO calculate the Pout considering hysteresis and restart wind speed

        :param high_resol_wind: type Iterator
        should be the return of Wind (in Wind_Class.py) instance method "simulate_transient_wind_speed_time_series"
        axis 0 is the number of traces
        axis 1 is the transient time step

        :param mode: can be either "time series" or "cross sectional"
        refer to Wind (in Wind_Class.py) instance method "simulate_transient_wind_speed_time_series"

        :param return_percentiles: type UncertaintyDataFrame
        If specified, then return to a UncertaintyDataFrame instance, whose index is determined by return_percentiles

        :return: simulated distribution of Pout
        for ndarray:
        axis 0 is the No. (i.e., the index, or corresponding position) of wind speed or wind_speed_std
        axis 1 is the number of traces (re-average)
        for pd.DataFrame:
        index is return_percentiles,
        column is the No. (i.e., the index, or corresponding position) of wind speed or wind_speed_std
        """
        # To prevent change in the existing return_percentiles
        return_percentiles = copy.deepcopy(return_percentiles)
        # Numba type should be determined so it can compile. Dynamic is not allowed
        numba_var_mfr_ws, numba_var_mfr_p = self.mfr_ws, self.mfr_p
        numba_var_cut_out_wind_speed = self.cut_out_wind_speed
        numba_var_restart_wind_speed = self.restart_wind_speed

        @guvectorize([(float32[:], boolean[:], float32[:], boolean[:])],
                     '(n), (n)->(n), (n)', nopython=True, target='parallel')
        def hpc_with_hysteresis_control_using_high_resol_wind(x, control_signal_input, y, control_signal_output):
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

            this_high_resol_pout_results, hysteresis_control_signal = hpc_with_hysteresis_control_using_high_resol_wind(
                this_high_resol_wind,
                hysteresis_control_signal
            )
            this_high_resol_pout_results = np.mean(this_high_resol_pout_results, axis=1)
            if return_percentiles is not None:
                # The reason not using @guvectorize decorating np.percentile is it will actually be slower...
                return_percentiles.update_one_column(this_high_resol_wind_index,
                                                     data=this_high_resol_pout_results)
            else:
                results.append(this_high_resol_pout_results)
        if return_percentiles is not None:
            # Must save as pd.DataFrame for stability. As UncertaintyDataFrame may change frequently
            return return_percentiles.pd_view
        else:
            return np.array(results)

    @staticmethod
    def infer_str_air_density(air_density: Union[int, float, str, Iterable]) -> Union[str, Tuple[str, ...]]:
        candidates_as_columns = PowerCurveByMfr.make_mfr_pc_metadata().columns
        candidates = candidates_as_columns.values.astype(float)
        if not isinstance(air_density, Iterable):
            index = np.argmin(np.abs(candidates - float(air_density)))
            return candidates_as_columns[index]
        else:
            str_tuple = []
            for this_air_density in air_density:
                index = np.argmin(np.abs(candidates - this_air_density.astype(float)))
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
    def map_given_power_output_to_another_air_density(
            cls, *, old_air_density: Union[OneDimensionNdarray, ndarray],
            new_air_density: Union[OneDimensionNdarray, ndarray],
            old_power_output: Union[IntFloatConstructedOneDimensionNdarray, ndarray],
            wind_speed: Union[IntFloatConstructedOneDimensionNdarray, ndarray],
    ) -> ndarray:
        old_air_density = cls.infer_str_air_density(OneDimensionNdarray(old_air_density))
        new_air_density = cls.infer_str_air_density(OneDimensionNdarray(new_air_density))
        mapping = np.array([str(x) for x in list(zip(old_air_density, new_air_density))])

        old_power_output = IntFloatConstructedOneDimensionNdarray(old_power_output)
        new_power_output = np.full(old_power_output.shape, np.nan)

        for this_unique_mapping in np.unique(mapping):
            mask = this_unique_mapping == mapping
            parse_obj = parse("('{}', '{}')", this_unique_mapping)
            this_old_air_density = parse_obj[0]
            this_new_air_density = parse_obj[1]
            mfr_pc_obj_by_this_old_air_density = cls(this_old_air_density)
            mfr_pc_obj_by_this_new_air_density = cls(this_new_air_density)

            new_call = mfr_pc_obj_by_this_new_air_density(wind_speed[mask])
            old_call = mfr_pc_obj_by_this_old_air_density(wind_speed[mask])
            new_call[np.abs(new_call) < (float_eps * 1e8)] = float_eps
            old_call[np.abs(old_call) < (float_eps * 1e8)] = float_eps
            ratio = new_call / old_call
            new_power_output[mask] = old_power_output[mask] * ratio
        return new_power_output

    @classmethod
    def air_density_in_docs(cls):
        return '0.97', '1', '1.03', '1.06', '1.09', '1.12', '1.15', '1.18', '1.21', '1.225', '1.24', '1.27'

    @classmethod
    def init_all_instances_in_docs(cls, **kwargs):
        air_density_in_docs_float = np.array([float(x) for x in cls.air_density_in_docs()])
        return cls.init_multiple_instances(air_density_in_docs_float, **kwargs)


class PowerCurveByMethodOfBins(PowerCurve):
    __slots__ = ('wind_speed_recording', 'active_power_output_recording', 'power_curve_look_up_table', 'bin_width')

    def __init__(self,
                 wind_speed_recording: ndarray,
                 active_power_output_recording: ndarray,
                 *, interp_for_high_resol: bool = True,
                 cal_region_boundary: bool = False,
                 color='fuchsia',
                 linestyle='-',
                 label='Scatters PC',
                 bin_width: Union[int, float] = 0.5,
                 rated_active_power_output: Union[int, float] = None):
        self.wind_speed_recording = wind_speed_recording
        self.active_power_output_recording = active_power_output_recording
        self.bin_width = bin_width
        self.rated_active_power_output = rated_active_power_output

        if interp_for_high_resol:
            self.power_curve_look_up_table = self.__cal_power_curve_look_up_table()
        else:
            if (wind_speed_recording is None) or (wind_speed_recording is None):
                self.power_curve_look_up_table = None
            else:
                self.power_curve_look_up_table = np.stack((self.wind_speed_recording,
                                                           self.active_power_output_recording), axis=1)
        # 只有在把outlier都去除了的条件下去计算region boundary才有意义，不然根本连rated都达不到，算法会有问题
        if cal_region_boundary:
            self.region_12_boundary, self.region_23_boundary, self.region_34_boundary, self.region_45_boundary = \
                self.cal_region_boundary()
        self.color = color
        self.linestyle = linestyle
        self.label = label

    def __cal_power_curve_look_up_table(self) -> ndarray:
        """
        利用线性插值得到高精度(bin=0.05m/s)的power curve的查找表
        """
        core_ws = np.arange(float_eps, 30, 0.05)
        power_curve_look_up_table_hi_resol = np.full((core_ws.size, 2), np.nan)
        power_curve_look_up_table_hi_resol[:, 0] = core_ws
        del core_ws
        power_curve_look_up_table = self.corresponding_mob_obj.cal_mob_statistic_eg_quantile()
        power_curve_look_up_table_hi_resol[:, 1] = interp1d(
            np.concatenate((np.array([-100]), power_curve_look_up_table[:, 0], np.array([100]))),
            np.concatenate((np.array([float_eps]), power_curve_look_up_table[:, 1], np.array([float_eps]))))(
            power_curve_look_up_table_hi_resol[:, 0])
        return power_curve_look_up_table_hi_resol

    @property
    def corresponding_mob_obj(self):
        from BivariateAnalysis_Class import MethodOfBins
        return MethodOfBins(self.wind_speed_recording, self.active_power_output_recording,
                            bin_step=self.bin_width,
                            first_bin_left_boundary=0)

    def __call__(self, ws: Union[IntFloatConstructedOneDimensionNdarray, ndarray]) -> ndarray:
        ws = IntFloatConstructedOneDimensionNdarray(ws)
        ws[ws == 0] = float_eps
        non_nan_mask = ~np.bitwise_or(*np.isnan(self.power_curve_look_up_table).T)
        return interp1d(np.concatenate(([-100], self.power_curve_look_up_table[non_nan_mask, 0], [100])),
                        np.concatenate(([float_eps], self.power_curve_look_up_table[non_nan_mask, 1], [float_eps])))(ws)

    def plot(self,
             ws: ndarray = None,
             plot_region_boundary: bool = False,
             ax=None,
             plot_recording: bool = True,
             mode='continuous',
             **kwargs):
        ax = scatter(self.wind_speed_recording, self.active_power_output_recording, ax=ax,
                     **{'alpha': WS_POUT_SCATTER_ALPHA,
                        's': WS_POUT_SCATTER_SIZE,
                        'color': 'royalblue'}) if plot_recording else ax
        return super().plot(ws, plot_region_boundary, ax, mode=mode, **kwargs)


class PowerCurveFittedBy8PLF(PowerCurveByMethodOfBins):
    ordered_params = ('a', 'd', 'b_1', 'c_1', 'g_1', 'b_2', 'c_2', 'g_2')

    __slots__ = ('a', 'd', 'b_1', 'c_1', 'g_1', 'b_2', 'c_2', 'g_2', '_params_constraints')

    def __init__(self,
                 wind_speed_recording: ndarray = None,
                 active_power_output_recording: ndarray = None,
                 *, interp_for_high_resol: bool = True,
                 color='darkorange',
                 linestyle='--',
                 label='8PL PC',
                 bin_width=0.5,
                 cal_region_boundary=False,
                 **kwargs):
        super().__init__(wind_speed_recording=wind_speed_recording,
                         active_power_output_recording=active_power_output_recording,
                         interp_for_high_resol=interp_for_high_resol,
                         cal_region_boundary=cal_region_boundary,
                         color=color,
                         linestyle=linestyle,
                         label=label,
                         bin_width=bin_width)
        constraints = dict(
            [
                ('a', [1 - float_eps, 1 + float_eps]),
                ('d', [-float_eps, float_eps]),
                ('b_1', [-20., -0.]),
                ('c_1', [10., 20.]),
                ('g_1', [float_eps, 1.]),
                ('b_2', [0., 60.]),
                ('c_2', [20., 30.]),
                ('g_2', [float_eps, 10]),
            ]
        )
        constraints = OrderedDict([(this_param, constraints.get(this_param)) for this_param in self.ordered_params])
        self._params_constraints = constraints  # type: OrderedDict
        self.update_params(**kwargs)

    @classmethod
    def init_from_power_curve_by_method_of_bins_obj(
            cls,
            power_curve_by_method_of_bins_instance: PowerCurveByMethodOfBins
    ) -> PowerCurveFittedBy8PLF:
        return cls(wind_speed_recording=power_curve_by_method_of_bins_instance.wind_speed_recording,
                   active_power_output_recording=power_curve_by_method_of_bins_instance.active_power_output_recording,
                   bin_width=power_curve_by_method_of_bins_instance.bin_width)

    def __str__(self):
        return f"{self.__class__.__name__} obj with {self.params_ordered_dict}"

    @property
    def corresponding_mob_pc_obj(self) -> PowerCurveByMethodOfBins:
        return PowerCurveByMethodOfBins(wind_speed_recording=self.wind_speed_recording,
                                        active_power_output_recording=self.active_power_output_recording,
                                        cal_region_boundary=False,
                                        bin_width=self.bin_width)

    @classmethod
    def call_static_func(cls) -> Callable:
        source = f"def static_func(x, {', '.join(cls.ordered_params)}):\n" \
                 f"    left_side = d + (a - d) / np.power(1 + np.power(x / c_1, b_1), g_1)\n" \
                 f"    right_side = d + (a - d) / np.power(1 + np.power(x / c_2, b_2), g_2)\n" \
                 f"    output = np.min(np.stack((left_side, right_side), axis=0), axis=0)\n" \
                 f"    return np.array(output)"
        exec(source, globals())

        return globals()['static_func']

    def __call__(self, ws: Union[IntFloatConstructedOneDimensionNdarray, Sequence]) -> ndarray:
        ws = IntFloatConstructedOneDimensionNdarray(ws)
        return self.call_static_func()(ws, *self.params)

    def _loss_func(self, wind_speed, *, target, focal_error=None) -> Callable:

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

    @property
    def params_ordered_dict(self) -> OrderedDict:
        return OrderedDict([(this_param, self.params[i]) for i, this_param in enumerate(self.ordered_params)])

    def update_params(self, *args, **kwargs):
        for i, this_param in enumerate(self.ordered_params):
            if args.__len__() == self.ordered_params.__len__():
                exec(f"self.{this_param} = args[{i}]")
            else:
                exec(f"self.{this_param} = float({kwargs}.get('{this_param}')) "
                     f"if {kwargs}.get('{this_param}') is not None else None")

    @property
    def params_constraints(self) -> OrderedDict:
        return self._params_constraints

    @params_constraints.setter
    def params_constraints(self, value: dict):
        self._params_constraints.update(value)

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
            wind_speed: Union[IntFloatConstructedOneDimensionNdarray, ndarray],
            focal_error: Union[int, float] = None,
            function_timeout=10):
        """
        Use GA to fit the 8PL PC.
        :param ga_algorithm_param: The parameters of GA algorithm
        :param params_init_scheme: Can be either 'self' or 'average',
        'self' means to use current self.params to initialise GA;
        'average' means to use the average of boundaries of constraints to initialise GA

        :param run_n_times: run total run_n_times times
        :param save_to_file_path
        :param wind_speed: Only fit the curve for this wind speed. Compulsory, due to the fact that
        self.power_curve_look_up_table may be series recordings!
        :param focal_error
        :param function_timeout
        :return:
        """
        wind_speed = IntFloatConstructedOneDimensionNdarray(wind_speed)
        target = self.corresponding_mob_pc_obj(wind_speed)
        # %% Prepare the parameters of GA algorithm and the initial values for fitting
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
        ga_model = ga(function=self._loss_func(wind_speed, focal_error=focal_error, target=target),
                      dimension=self.params.__len__(),
                      variable_type='real',
                      variable_boundaries=np.array(list(self.params_constraints.values())),
                      function_timeout=function_timeout,
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
             plot_recording: bool = True,
             **kwargs):
        if all((self.wind_speed_recording is not None,
                self.active_power_output_recording is not None,
                plot_recording)):
            passed_sig = Signature.from_callable(PowerCurveByMethodOfBins.plot).parameters.keys()
            safe_locals = locals()
            passed_args = {key: safe_locals[key] for key in passed_sig if (key != 'self' and key != 'kwargs')}

            ax = self.corresponding_mob_pc_obj.plot(**passed_args)
        else:
            ax = ax
        return super(PowerCurveByMethodOfBins, self).plot(ws, plot_region_boundary, ax, mode=mode, **kwargs)


class PowerCurveFittedBy5PLF(PowerCurveFittedBy8PLF):
    ordered_params = ('a', 'd', 'b_1', 'c_1', 'g_1')

    __slots__ = ('a', 'd', 'b_1', 'c_1', 'g_1')

    def __init__(self, *args,
                 color='darkorange',
                 linestyle='--',
                 label='5PL PC', **kwargs):
        super().__init__(*args, **kwargs)
        self.color = color
        self.linestyle = linestyle
        self.label = label

    # @property
    # def params_constraints(self) -> OrderedDict:
    #     constraints = dict(
    #         [
    #             ('a', [0.95, 1.02]),
    #             ('d', [-0.02, 0.02]),
    #             ('b', [-20.0, -5.]),
    #             ('c', [10., 15.]),
    #             ('g', [float_eps, 0.5]),
    #         ]
    #     )
    #     constraints = OrderedDict([(this_param, constraints[this_param]) for this_param in self.ordered_params])
    #     return constraints

    @classmethod
    def call_static_func(cls) -> Callable:
        source = f"def static_func(x, {', '.join(cls.ordered_params)}):\n" \
                 f"    output = d + (a - d) / np.power(1 + np.power(x / c_1, b_1), g_1)\n" \
                 f"    return np.array(output)"
        exec(source, globals())

        return globals()['static_func']


class EquivalentWindFarmPowerCurve(PowerCurveFittedBy8PLF):
    __slots__ = ("total_wind_turbine_number", "index")

    def __init__(self, *args, total_wind_turbine_number: int, index=None, **kwargs):
        super().__init__(*args, interp_for_high_resol=False, **kwargs)
        # To make life easy, the constraints are set using Mfr PC fitting experience
        constraints = dict(
            [
                ('a', [1 - float_eps, 1 + float_eps]),
                ('d', [-float_eps, float_eps]),
                ('b_1', [-15., -7.5]),
                ('c_1', [10., 15.]),
                ('g_1', [float_eps, 0.6]),
                ('b_2', [0., 100.]),
                ('c_2', [20., 30.]),
                ('g_2', [float_eps, 2]),
            ]
        )
        constraints = OrderedDict([(this_param, constraints[this_param]) for this_param in self.ordered_params])
        self._params_constraints = constraints  # type: OrderedDict
        self.total_wind_turbine_number = total_wind_turbine_number
        self.index = index

    @classmethod
    def init_from_8p_pc_obj(cls, power_curve_fitted_by_8plf_obj: PowerCurveFittedBy8PLF,
                            *, total_wind_turbine_number: int,
                            **kwargs) -> EquivalentWindFarmPowerCurve:
        return cls(**power_curve_fitted_by_8plf_obj.params_ordered_dict,
                   total_wind_turbine_number=total_wind_turbine_number,
                   bin_width=power_curve_fitted_by_8plf_obj.bin_width, **kwargs)

    def __call__(self, ws: Union[IntFloatConstructedOneDimensionNdarray, Sequence, int, float], *,
                 operating_wind_turbine_number: Union[IntOneDimensionNdarray, Sequence, int] = None,
                 total_curtailment_amount: Union[IntFloatConstructedOneDimensionNdarray, Sequence, int] = None):
        ws = IntFloatConstructedOneDimensionNdarray(ws)
        operating_wind_turbine_number = IntOneDimensionNdarray(operating_wind_turbine_number)
        total_curtailment_amount = IntFloatConstructedOneDimensionNdarray(total_curtailment_amount)

        fully_operating_regime_power_output = super().__call__(ws)
        # Linear scaling and plus bias
        power_output = fully_operating_regime_power_output * (
                operating_wind_turbine_number / self.total_wind_turbine_number) + total_curtailment_amount
        power_output = np.array(power_output)
        return power_output

    def assess_fit_2d_scatters(
            self, *,
            ax=None,
            operating_regime: DataCategoryData,
            total_curtailment_amount: Union[IntFloatConstructedOneDimensionNdarray, Sequence, int] = None,
            original_scatters_pc: PowerCurve
    ):
        assert ((self.wind_speed_recording is not None) and (self.active_power_output_recording is not None))

        if total_curtailment_amount is None:
            total_curtailment_amount = self.maximum_likelihood_estimation_for_wind_farm_operation_regime(
                task='evaluate'
            )[1]

        # %% Define a function to obtain the corresponding PowerCurveByMethodOfBins obj
        def obtain_corresponding_power_curve_by_method_of_bins_obj(index_in_wind_farm_eq_obj: ndarray):
            _this_operating_regime_mob_pc = PowerCurveByMethodOfBins(
                *self.wind_farm_eq_obj.loc[index_in_wind_farm_eq_obj].values.T,
                bin_width=self.bin_width,
            )
            ws_inner = np.array(
                [x['this_bin_boundary'][1]
                 for x in _this_operating_regime_mob_pc.corresponding_mob_obj.mob.values()
                 if not x['this_bin_is_empty']]
            )
            power_output_inner = np.array(
                [np.mean(x['dependent_var_in_this_bin'])
                 for x in _this_operating_regime_mob_pc.corresponding_mob_obj.mob.values()
                 if not x['this_bin_is_empty']]
            )
            return _this_operating_regime_mob_pc, ws_inner, power_output_inner

        error_df = pd.DataFrame(
            columns=pd.MultiIndex.from_product(
                [['Raw', 'My model'], ['RMSE', 'MAE']],
                names=['Model used', 'error']
            ),
            index=[x for x in operating_regime.name_mapper['abbreviation'] if x != 'others']
        )
        ls_pc_label_existing_flag = False
        ls_pc_plus_clt_label_existing_flag = False
        regime_pc_existing_flag = False
        for i, (this_operating_regime_long_name, this_operating_regime_abbreviation) in enumerate(zip(
                operating_regime.name_mapper['long name'],
                operating_regime.name_mapper['abbreviation']
        )):
            if this_operating_regime_abbreviation == 'others':
                continue
            parse_obj = parse("({}, {}, {})", this_operating_regime_long_name)
            operating_wind_turbine_number = int(parse_obj[0])
            curtailed_wind_turbine_number = int(parse_obj[1])
            actual_recording = []
            output_dict = {x: [] for x in error_df.columns.get_level_values('Model used').unique()}

            # %% Define a function to obtain power outputs for error analysis and also for plot
            def obtain_power_output_for_error_and_plot(this_total_curtailment_amount):
                power_output_inner = self(
                    ws,
                    operating_wind_turbine_number=IntOneDimensionNdarray([operating_wind_turbine_number]),
                    total_curtailment_amount=this_total_curtailment_amount
                )
                power_output_plot_inner = self(
                    ws_plot,
                    operating_wind_turbine_number=IntOneDimensionNdarray([operating_wind_turbine_number]),
                    total_curtailment_amount=this_total_curtailment_amount
                )
                output_dict['My model'].extend(power_output_inner)
                output_dict['Raw'].extend(original_scatters_pc(ws))
                actual_recording.extend(bin_pout)
                return power_output_plot_inner

            # If there are any WT curtailment, note that the amount of curtailment is uncertain can be be any value!
            # So, the pain here is to do the more detailed check again, for all unique curtailment,
            # despite that the curtailed_wind_turbine_number is the same!
            if curtailed_wind_turbine_number > 0:
                curtailment_in_this_operating_regime = total_curtailment_amount[
                    operating_regime(this_operating_regime_abbreviation)
                ]
                # round to 4 decimal places to reduce noise effects
                curtailment_in_this_operating_regime_round = curtailment_in_this_operating_regime.round(2)
                # Do a for-loop for considering all different curtailment amount separately
                for _, curtailment in enumerate(np.unique(curtailment_in_this_operating_regime_round)):
                    index = curtailment_in_this_operating_regime_round[
                        curtailment == curtailment_in_this_operating_regime_round].index
                    this_operating_regime_mob_pc, ws, bin_pout = \
                        obtain_corresponding_power_curve_by_method_of_bins_obj(index)
                    ws_plot = np.arange(np.min(ws), np.max(ws) + 0.1, 0.01)
                    # Wind speed ndarray obj for calculation error
                    power_output_plot = obtain_power_output_for_error_and_plot(curtailment)

                    if not ls_pc_plus_clt_label_existing_flag:
                        label = 'LS 6P-PC\nplus CTL'
                        ls_pc_plus_clt_label_existing_flag = True
                    else:
                        label = None
                    ax = series(ws_plot, power_output_plot, ax=ax, color='red', linestyle='-', label=label)
                    ax = series(ws, bin_pout, ax=ax, color='black', linestyle='--')

            # Much easy and straight forward if there are no WT curtailment
            else:
                this_operating_regime_mob_pc, ws, bin_pout = obtain_corresponding_power_curve_by_method_of_bins_obj(
                    operating_regime(this_operating_regime_abbreviation)
                )
                ws_plot = np.arange(0, 50, 0.1)
                # Wind speed ndarray obj for calculation error
                power_output_plot = obtain_power_output_for_error_and_plot(0)

                if this_operating_regime_abbreviation == 'S1':
                    ax = self.corresponding_8p_pc_obj.plot(
                        ws=ws_plot, ax=ax, plot_recording=False, color='blue', linestyle='-', label='6P-PC'
                    )
                else:
                    label = 'LS 6P-PC' if not ls_pc_label_existing_flag else None
                    ls_pc_label_existing_flag = True
                    ax = series(ws_plot, power_output_plot, ax=ax, color='green', linestyle='-', label=label)

                if not regime_pc_existing_flag:
                    label_2 = 'Regime PC'
                    regime_pc_existing_flag = True
                else:
                    label_2 = None
                ax = series(ws, bin_pout, ax=ax, color='black', linestyle='--', label=label_2)

            # %% Calculate error
            for key in error_df.columns.get_level_values('Model used').unique():
                error_obj = DeterministicError(target=np.array(actual_recording).flatten(),
                                               model_output=np.array(output_dict[key]).flatten())
                error_df.loc[this_operating_regime_abbreviation,
                             (key, 'RMSE')] = error_obj.cal_root_mean_square_error()
                error_df.loc[this_operating_regime_abbreviation,
                             (key, 'MAE')] = error_obj.cal_mean_absolute_error()
        ax = scatter(*self.wind_farm_eq_obj.loc[~operating_regime('others'), ['wind speed',
                                                                              'active power output']].values.T,
                     ax=ax, color='silver', alpha=0.75, zorder=-1, label="Filtered data")
        # Adjust the order of the legend
        new_order_of_labels = ['Filtered data', 'Regime PC', '6P-PC', 'LS 6P-PC']
        if ls_pc_plus_clt_label_existing_flag:
            new_order_of_labels.append('LS 6P-PC\nplus CTL')

        ax = adjust_legend_order_in_ax(ax, new_order_of_labels=new_order_of_labels)

        # error_df.to_csv(fitting_file_path.parent / 'errors.csv')
        x_ticks = operating_regime.name_mapper.convert_sequence_data_key(
            'abbreviation', 'long name',
            sequence_data=error_df.index.values
        )
        ax_error = series(x_ticks, error_df[('My model', 'RMSE')].values, color='green',
                          label='RMSE', figure_size=(5, 5 * (0.618 ** 1.6)), marker='s', markersize=4)
        ax_error = series(x_ticks, error_df[('My model', 'MAE')].values, ax=ax_error,
                          marker='.', markersize=8, label='MAE',
                          x_label=r'WF Operating Regime [($\it{a}$, $\it{b}$, $\it{c}$)]', x_ticks_rotation=45,
                          y_label='Error [p.u.]')

    def assess_fit_time_series(
            self, *,
            operating_regime: DataCategoryData,
            total_curtailment_amount: Union[IntFloatConstructedOneDimensionNdarray, Sequence, int] = None,
            original_scatters_pc: PowerCurve
    ):
        assert ((self.wind_speed_recording is not None) and (self.active_power_output_recording is not None))

        if total_curtailment_amount is None:
            total_curtailment_amount = self.maximum_likelihood_estimation_for_wind_farm_operation_regime(
                task='evaluate'
            )[1].values

        mfr_pc_pout_est = PowerCurveByMfr('1.12')(self.wind_farm_eq_obj['wind speed'].values)
        scatters_pc_pout_est = original_scatters_pc(self.wind_farm_eq_obj['wind speed'].values)
        operating_wind_turbine_number = [int(parse("({}, {}, {})", x)[0])
                                         for x in operating_regime.name_mapper.convert_sequence_data_key(
                'abbreviation', 'long name', sequence_data=operating_regime.abbreviation
            )]
        my_model_pout_est = self(
            self.wind_farm_eq_obj['wind speed'].values,
            operating_wind_turbine_number=operating_wind_turbine_number,
            total_curtailment_amount=total_curtailment_amount
        )
        pout_actual = self.wind_farm_eq_obj['active power output'].values

        # Error calculation
        for this_model_est in ('mfr_pc_pout_est', 'scatters_pc_pout_est', 'my_model_pout_est'):
            error_obj = DeterministicError(target=pout_actual,
                                           model_output=eval(this_model_est))
            print(this_model_est + "\n" + f"RMSE={error_obj.cal_root_mean_square_error()} p.u.,\n"
                                          f"MAE={error_obj.cal_mean_absolute_error()} p.u.\n\n")

    @property
    def wind_farm_eq_obj(self):
        _wind_farm_eq_obj = PhysicalInstanceDataFrame(data={'wind speed': self.wind_speed_recording,
                                                            'active power output': self.active_power_output_recording},
                                                      index=self.index,
                                                      dependant_names=('active power output',),
                                                      obj_name='wind_farm_eq_obj',
                                                      predictor_names=('wind speed',))
        return _wind_farm_eq_obj

    @property
    def corresponding_8p_pc_obj(self) -> PowerCurveFittedBy8PLF:
        pc_obj = PowerCurveFittedBy8PLF(interp_for_high_resol=False, bin_width=self.bin_width)
        pc_obj.update_params(*self.params)
        return pc_obj

    def mle_initialisation(self, task: str = 'fit'):
        assert (task in ('fit', 'evaluate'))
        # Candidates
        possible_normally_operating_wind_turbine = range(0, 1 + self.total_wind_turbine_number)
        expected_rated_power_output_in_operating_regime = pd.Series(
            index=possible_normally_operating_wind_turbine,
            data=np.array(possible_normally_operating_wind_turbine) / self.total_wind_turbine_number
        )

        # %% To find the flat power output in the Pout-WS 2D scatter plot
        flat_power_output_mask = self.wind_farm_eq_obj.data_category_is_linearity(
            '30T',
            constant_error={'active power output': 0.00001}
        )

        # %% Find the cases when there is any WT curtailment
        abs_flat_diff = np.abs(self.active_power_output_recording[flat_power_output_mask] -
                               expected_rated_power_output_in_operating_regime.values[:, np.newaxis])
        abs_flat_diff = np.min(abs_flat_diff, axis=0)
        curtailment_happen_index = self.wind_farm_eq_obj.index[flat_power_output_mask][
            abs_flat_diff > (0.02 / self.total_wind_turbine_number)]
        curtailment_happen_mask = np.isin(self.index, curtailment_happen_index)
        total_pout_when_curtail = np.full_like(self.active_power_output_recording, fill_value=0.)
        total_pout_when_curtail[curtailment_happen_mask] = self.active_power_output_recording[curtailment_happen_mask]
        del flat_power_output_mask, abs_flat_diff

        # %% Using special Mfr PC to ensure the operating regime when WS > WS_cut_out and flat
        ws_above_limit_mask = self.wind_speed_recording > (PowerCurveByMfr.cut_out_wind_speed * 0.8)
        diff = (self.active_power_output_recording[ws_above_limit_mask] -
                expected_rated_power_output_in_operating_regime.values[:, np.newaxis] * 1.0125)
        temp = []
        for i in range(diff.shape[1]):
            temp.append(np.argwhere(diff[:, i] < 0)[0, 0])
        ws_above_limit_regime = np.full(ws_above_limit_mask.shape, fill_value=np.nan)
        ws_above_limit_regime[ws_above_limit_mask] = temp
        if task == 'evaluate':
            ws_above_limit_regime[self.active_power_output_recording <= 0.0125] = 0  # Correction for all shut down
        return (possible_normally_operating_wind_turbine,
                expected_rated_power_output_in_operating_regime,
                pd.Series(total_pout_when_curtail, index=self.index),
                curtailment_happen_mask,
                ws_above_limit_mask,
                ws_above_limit_regime)

    def maximum_likelihood_estimation_for_wind_farm_operation_regime(self, power_output_func: Callable = None, *,
                                                                     initialisation=None,
                                                                     return_fancy: bool = False,
                                                                     task: str = 'fit'):
        power_output_func = power_output_func or super(EquivalentWindFarmPowerCurve, self).__call__
        initialisation = initialisation or self.mle_initialisation(task)
        possible_normally_operating_wind_turbine = initialisation[0]
        expected_rated_power_output_in_operating_regime = initialisation[1]
        total_pout_when_curtail_values = initialisation[2].values
        curtailment_happen_mask = initialisation[3]
        ws_above_limit_mask = initialisation[4]
        ws_above_limit_regime = initialisation[5]

        # %% Find the closest normally operating WTs if no WT curtailment
        power_output = power_output_func(self.wind_speed_recording)
        power_output_scaled = np.tile(power_output, (possible_normally_operating_wind_turbine.__len__(), 1))
        power_output_scaled *= expected_rated_power_output_in_operating_regime.values[:, np.newaxis]
        abs_diff = np.abs(self.active_power_output_recording - power_output_scaled)
        normally_operating_number = np.argmin(abs_diff, axis=0)
        del power_output, power_output_scaled, abs_diff

        # %% Reassign when it is operating regime when WS > WS_cut_out and flat but there is not curtailment
        ws_above_limit_no_curt_mask = np.bitwise_and(ws_above_limit_mask, ~curtailment_happen_mask)
        normally_operating_number[ws_above_limit_no_curt_mask] = ws_above_limit_regime[ws_above_limit_no_curt_mask]

        # %% Reassign normally operating WTs when there is curtailment
        diff = (total_pout_when_curtail_values[curtailment_happen_mask] -
                expected_rated_power_output_in_operating_regime.values[:, np.newaxis])
        revise = []
        for i in range(diff.shape[1]):
            revise.append(np.argwhere(diff[:, i] > 0)[-1, 0])
        normally_operating_number[curtailment_happen_mask] = revise

        # %% Make correction that exceeding limit
        limits = normally_operating_number / self.total_wind_turbine_number
        above_limits_mask = self.active_power_output_recording > (limits * 1.0125)
        normally_operating_number[np.bitwise_and(above_limits_mask, ~curtailment_happen_mask)] += 1
        del limits

        # %% Summary
        curtailment_number = copy.deepcopy(total_pout_when_curtail_values)
        curtailment_number[curtailment_number != 0] = 1
        curtailment_number = curtailment_number.astype(int)
        shutdown_or_nan_number = self.total_wind_turbine_number - curtailment_number - normally_operating_number

        fancy = None
        if return_fancy:
            long_name_array = np.array([f"({a}, {b}, {c})" for a, b, c in zip(normally_operating_number,
                                                                              curtailment_number,
                                                                              shutdown_or_nan_number)])
            abbreviation_array_unique_sorted = np.array(
                sorted(np.unique(long_name_array),
                       key=lambda x: (int(parse(r"({}, {}, {})", x)[0]),
                                      int(parse(r"({}, {}, {})", x)[1]),
                                      int(parse(r"({}, {}, {})", x)[2])),
                       reverse=True)
            )
            name_mapper = DataCategoryNameMapper(data={
                'long name': abbreviation_array_unique_sorted,
                'abbreviation': [f"S{i}" for i in range(1, abbreviation_array_unique_sorted.__len__() + 1)],
                'code': [-1] * abbreviation_array_unique_sorted.__len__(),
                'description': abbreviation_array_unique_sorted
            })
            fancy = DataCategoryData(
                abbreviation=name_mapper.convert_sequence_data_key('long name', 'abbreviation',
                                                                   sequence_data=long_name_array),
                index=self.wind_farm_eq_obj.index,
                name_mapper=name_mapper
            )

        summary = dict((('normally_operating_number', normally_operating_number),
                        ('curtailment_number', curtailment_number),
                        ('shutdown_or_nan_number', shutdown_or_nan_number)))
        do_not_cal_error_mask = np.bitwise_or(above_limits_mask, curtailment_happen_mask)

        total_curtail_amount = copy.deepcopy(total_pout_when_curtail_values)
        total_curtail_amount[curtailment_happen_mask] = (total_curtail_amount - normally_operating_number /
                                                         self.total_wind_turbine_number)[curtailment_happen_mask]
        return summary, pd.Series(total_curtail_amount, index=initialisation[2].index), do_not_cal_error_mask, fancy

    def _loss_func(self, wind_speed, *,
                   target,
                   focal_error=None) -> Callable:
        # %% Assign weights for attention
        attention_weights = np.full_like(wind_speed, fill_value=np.nan)
        bins = np.histogram(wind_speed,
                            bins=np.arange(0, np.nanmax(wind_speed) + 5, 0.5))[1]
        for i in range(len(bins) - 1):
            this_bin_mask = np.bitwise_and(wind_speed >= bins[i], wind_speed < bins[i + 1])
            this_bin_number = np.sum(this_bin_mask)
            if this_bin_number == 0:
                continue
            else:
                attention_weights[this_bin_mask] = 1 / this_bin_number
        attention_weights /= (np.sum(attention_weights) / len(attention_weights))
        assert ~(np.isnan(attention_weights).any())
        initialisation = self.mle_initialisation('fit')

        def func(params_array):
            model_output = self.call_static_func()(wind_speed, *(params_array[:self.params.__len__()]))
            mle = self.maximum_likelihood_estimation_for_wind_farm_operation_regime(initialisation=initialisation)
            scale_fact = mle[0]['normally_operating_number']
            model_output *= (scale_fact / self.total_wind_turbine_number)
            # focal loss
            error = model_output - target
            if focal_error is not None:
                focal_index = np.abs(error) > focal_error
                if np.sum(focal_index) == 0:
                    focal_index = np.full(error.shape, True)
            else:
                focal_index = np.full(error.shape, True)
            # attention weights, importance
            error *= attention_weights
            return float(np.sqrt(np.mean(error[np.bitwise_and(focal_index, ~mle[-2])] ** 2)))

        return func

    def _params_init(self, save_to_file_path: Path = None):
        """
        This is to initialise the params using guessed fully operating regime
        :return:
        """
        mle = self.maximum_likelihood_estimation_for_wind_farm_operation_regime(PowerCurveByMfr('1.12').__call__)
        guess_file_path = save_to_file_path.parent / (save_to_file_path.stem + '_S1_guess.pkl')
        guess_params = load_pkl_file(guess_file_path)
        # Fit S1
        if guess_params is None:
            fully_mask = mle[0]['normally_operating_number'] == self.total_wind_turbine_number
            wind_speed = self.wind_speed_recording[fully_mask]
            power_output = self.active_power_output_recording[fully_mask]
            # Initialise a PowerCurveFittedBy5PLF obj for fitting
            guess_fully_5p_pc = PowerCurveFittedBy5PLF(wind_speed, power_output)
            guess_fully_5p_pc.params_constraints = {key: self.params_constraints[key] for key in ('b_1', 'c_1', 'g_1')}
            # Get the S1 before cut-out wind speed boundary
            guess_fully_mob_pc = PowerCurveByMethodOfBins(wind_speed, power_output,
                                                          rated_active_power_output=1,
                                                          cal_region_boundary=True)
            guess_fully_5p_pc.fit(run_n_times=50,
                                  params_init_scheme='average',
                                  wind_speed=np.arange(0, guess_fully_mob_pc.region_34_boundary + 0.1, 0.1),
                                  save_to_file_path=guess_file_path)
            print(f"_S1_guess obtained as 👉 {guess_fully_5p_pc}")
            guess_params = guess_fully_5p_pc.params
        else:
            guess_params = guess_params[-1]['variable']
        # For part "2", still using average constraints guess
        for i, this_param in enumerate(self.ordered_params):
            if '_2' in this_param:
                source_code = f"self.{this_param} = (self._params_constraints['{this_param}'][0] + " \
                              f"self._params_constraints['{this_param}'][-1]) / 2"
            else:
                source_code = f"self.{this_param} = {guess_params[i]}"
            exec(source_code)
        return self.params

    def fit(self, *, ga_algorithm_param: dict = None,
            params_init_scheme: str,
            run_n_times: int = 3,
            save_to_file_path: Path = None,
            focal_error: Union[int, float] = 0,
            function_timeout=10, **kwargs):
        wind_speed = IntFloatConstructedOneDimensionNdarray(self.wind_speed_recording)
        target = IntFloatConstructedOneDimensionNdarray(self.active_power_output_recording)
        assert (params_init_scheme in ('self', 'guess')), "'params_init_scheme' can either be 'self' or 'guess'"

        # %% Prepare the parameters of GA algorithm and the initial values for fitting
        ga_algorithm_param = ga_algorithm_param or {}
        if params_init_scheme == 'self':
            if any([this_param is None for this_param in self.params]):
                raise Exception("To use 'self' in 'params_init_scheme', all params must not be None. "
                                "Otherwise, please use 'average'")
            initialised_params = self.params
        else:
            initialised_params = self._params_init(save_to_file_path)

        # constraints
        variable_boundaries = np.array(list(self.params_constraints.values()))

        # dynamical loss func
        loss_func = self._loss_func(wind_speed,
                                    focal_error=focal_error,
                                    target=target)
        # %% Init GA obj
        default_algorithm_param = Signature.from_callable(ga.__init__).parameters['algorithm_parameters'].default
        ga_algorithm_param = dict(ChainMap(ga_algorithm_param, default_algorithm_param))
        ga_model = ga(function=loss_func,
                      dimension=self.params.__len__(),
                      variable_type='real',
                      variable_boundaries=variable_boundaries,
                      function_timeout=function_timeout,
                      algorithm_parameters=ga_algorithm_param)

        # %% Run GA and save the results
        ga_model_run_results = []
        for _ in tqdm(range(run_n_times)):
            ga_model.run(init_solo=initialised_params)
            print(ga_model.__str__())
            # %% For clustering computation
            this_run_loss = ga_model.output_dict['function']
            existing_results = load_pkl_file(save_to_file_path)
            if existing_results is None:
                ga_model_run_results.append(ga_model.output_dict)
                initialised_params = ga_model.best_variable  # This will initialise the next GA using the current best
                save_pkl_file(save_to_file_path, ga_model_run_results)
            else:
                # Keep the cloud version
                if loss_func(existing_results[-1]['variable']) < this_run_loss:
                    initialised_params = existing_results[-1]['variable']
                else:
                    ga_model_run_results.append(ga_model.output_dict)
                    initialised_params = ga_model.best_variable
                    # Update the existing by extending
                    save_pkl_file(save_to_file_path, ga_model_run_results)
        # %% Update 8PL PC
        for this_param_index, this_param in enumerate(self.ordered_params):
            source_code = f"self.{this_param} = ga_model.best_variable[{this_param_index}]"  # The last is the best
            exec(source_code)

        return ga_model_run_results


if __name__ == '__main__':
    cc = PowerCurveByMfr.init_multiple_instances(np.array([1.1, 1.2, 1.1]))
