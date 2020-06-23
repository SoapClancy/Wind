from BivariateAnalysis_Class import MethodOfBins
from abc import ABCMeta, abstractmethod
from functools import wraps
from numpy import ndarray
from scipy.interpolate import interp1d
import numpy as np
from typing import Union, Tuple
from Ploting.fast_plot_Func import series, scatter, vlines
from project_path_Var import project_path_
from scipy.io import loadmat
import pandas as pd


class PowerCurve(metaclass=ABCMeta):
    cut_in_wind_speed = 4
    rated_wind_speed = 18
    cut_out_wind_speed = 25
    rated_active_power_output = 3000

    __slots__ = ('linestyle', 'color', 'label',
                 'region_12_boundary', 'region_23_boundary', 'region_34_boundary', 'region_45_boundary',
                 'power_curve_look_up_table')

    def __init__(self, linestyle: str, color: str, label: str = None):
        self.linestyle = linestyle
        self.color = color
        self.label = label

    @abstractmethod
    def cal_active_power_output_according_to_wind_speed(self, ws: Union[ndarray, float, int]):
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
            (self.cal_active_power_output_according_to_wind_speed(region_12_boundary_est) - tolerance) > 0)[0][0]]
        region_23_boundary = region_23_boundary_est[np.where(
            (self.cal_active_power_output_according_to_wind_speed(region_23_boundary_est) - (
                    self.rated_active_power_output - tolerance)) > 0)[0][0]]
        try:
            region_34_boundary = region_34_boundary_est[np.where(
                (self.cal_active_power_output_according_to_wind_speed(region_34_boundary_est) - (
                        self.rated_active_power_output - tolerance)) < 0)[0][0]]
        except IndexError:
            region_34_boundary = np.nan
        try:
            region_45_boundary = region_45_boundary_est[np.where(
                (self.cal_active_power_output_according_to_wind_speed(region_45_boundary_est) - tolerance) < 0)[0][0]]
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

    def plot_power_curve(self, ws: ndarray = None, plot_region_boundary: bool = False, ax=None):
        ax = self.__plot_region_boundary(ax) if plot_region_boundary else ax
        ws = ws or np.arange(0, 28.5, 0.1)
        active_power_output = self.cal_active_power_output_according_to_wind_speed(ws)
        return series(ws, active_power_output, ax,
                      linestyle=self.linestyle, color=self.color, label=self.label,
                      y_lim=(-self.rated_active_power_output * 0.013, self.rated_active_power_output * 1.013),
                      x_lim=(0, 28.6),
                      x_label='Wind speed (m/s)', y_label='Active power output (kW)')


class PowerCurveByMfr(PowerCurve):
    __slots__ = ('mfr_ws', 'mfr_p')

    def __init__(self, linestyle='-.', color=(0.64, 0.08, 0.18), label='Mfr PC'):
        super().__init__(linestyle, color, label)
        mfr_data = loadmat(project_path_ + 'Data/Raw_measurements/Manu_Data.mat')['Manu_Data']
        self.mfr_ws = mfr_data[:, 1]
        self.mfr_p = mfr_data[:, 2] * 3000

    def cal_active_power_output_according_to_wind_speed(self, ws: Union[ndarray, float, int]):
        if not isinstance(ws, ndarray):
            ws = np.array([ws * 1.0])
        ws = ws.flatten()
        return interp1d(self.mfr_ws, self.mfr_p)(ws)


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

    def cal_active_power_output_according_to_wind_speed(self, ws: Union[ndarray, float, int]) -> ndarray:
        if not isinstance(ws, ndarray):
            ws = np.array([ws * 1.0])
        ws = ws.flatten()
        return interp1d(np.concatenate((np.array([-100]), self.power_curve_look_up_table[:, 0], np.array([100]))),
                        np.concatenate((np.array([0]), self.power_curve_look_up_table[:, 1], np.array([0]))))(ws)

    def plot_power_curve(self, ws: ndarray = None, plot_region_boundary: bool = True, plot_recording: bool = True):
        ax = scatter(self.wind_speed_recording, self.active_power_output_recording,
                     alpha=0.1) if plot_recording else None
        return super().plot_power_curve(ws, plot_region_boundary, ax)


def make_mfr_pc_metadata():
    """
    生成初始化PowerCurveByMfr必须要有的元数据，包括mfr_ws和mfr_p。
    数据会以pd.DataFrame的形式储存，行索引是WS [m/s]，int类型
    列名是air density [kg/m^3], str类型，
    元素是Pout [W]，float类型
    :return: None
    """
    metadata = 1


if __name__ == '__main__':
    make_mfr_pc_metadata()
