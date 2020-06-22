from Regression_Analysis.DeepLearning_Class import datetime_one_hot_encoder, MatlabLSTM, \
    prepare_data_for_nn
import numpy as np
import pandas as pd
from numpy import ndarray
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple, np_datetime64_to_datetime
from project_path_Var import project_path_
from File_Management.load_save_Func import load_exist_npy_file_otherwise_run_and_save, save_npy_file, \
    load_exist_pkl_file_otherwise_run_and_save, load_npy_file, load_pkl_file
from BivariateAnalysis_Class import BivariateOutlier
from File_Management.path_and_file_management_Func import try_to_find_path_otherwise_make_one
from UnivariateAnalysis_Class import CategoryUnivariate, UnivariatePDFOrCDFLike, UnivariateGaussianMixtureModel, \
    DeterministicUnivariateProbabilisticModel
from typing import Union, Tuple, List
from BivariateAnalysis_Class import Bivariate, MethodOfBins
from Ploting.fast_plot_Func import series, hist, scatter, scatter_density
from PowerCurve_Class import PowerCurveByMethodOfBins, PowerCurve, PowerCurveByMfr
from Filtering.simple_filtering_Func import linear_series_outlier, out_of_range_outlier, \
    change_point_outlier_by_sliding_window_and_interquartile_range
from Data_Preprocessing.TruncatedOrCircularToLinear_Class import TruncatedToLinear, CircularToLinear
import copy
from Data_Preprocessing.float_precision_control_Func import float_eps
from Correlation_Modeling.Copula_Class import VineGMCMCopula, GMCM, \
    FOUR_DIM_CVINE_CONSTRUCTION, THREE_DIM_CVINE_CONSTRUCTION
from TimeSeries_Class import SynchronousTimeSeriesData
import datetime
from Time_Processing.Season_Enum import SeasonTemplate1
from enum import Enum
from WT_Class import WT
import os
import random


class WF(SynchronousTimeSeriesData):
    results_path = project_path_ + 'Data/Results/'
    cut_in_wind_speed = 4
    rated_wind_speed = 18
    cut_out_wind_speed = 25
    rated_active_power_output = None

    __slots__ = ('name', 'measurements', 'outlier_category', 'outlier_category_detailed')
    """
    name: WF的名字
    measurements: 一个pd.DataFrame，包含同步测量的数据，DataFrame.columns包含:
        'time',
        'wind speed',
        'active power output',
        'environmental temperature'
    """

    def __init__(self, *, name: str, measurements: pd.DataFrame, outlier_category: ndarray,
                 outlier_category_detailed: pd.DataFrame):
        super().__init__(synchronous_data=measurements,
                         data_category=outlier_category,
                         category_detailed=outlier_category_detailed)
        self.name = name
        self.measurements = measurements
        self.outlier_category = outlier_category
        self.outlier_category_detailed = outlier_category_detailed

    def __str__(self):
        t1 = np_datetime64_to_datetime(self.measurements['time'].values[0]).strftime('%Y-%m-%d %H.%M')
        t2 = np_datetime64_to_datetime(self.measurements['time'].values[-1]).strftime('%Y-%m-%d %H.%M')
        current_season = self.get_current_season(season_template=SeasonTemplate1)
        if current_season == 'all seasons':
            return "{} WF from {} to {}".format(self.name, t1, t2)
        else:
            return "{} WF from {} to {} {}".format(self.name, t1, t2, current_season)

    def do_truncate(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None):
        self.measurements, self.outlier_category, self.outlier_category_detailed = super().do_truncate(
            start_time=start_time,
            end_time=end_time
        )
        return self.measurements, self.outlier_category, self.outlier_category_detailed

    def down_sample(self, aggregate_on_sample_number: int, aggregate_on_category: Tuple[int, ...] = (0,),
                    category_is_outlier: bool = True):
        self.name += '_down_sampled_on_{}'.format(aggregate_on_sample_number)
        self.measurements, self.outlier_category, self.outlier_category_detailed = super().down_sample(
            aggregate_on_sample_number,
            aggregate_on_category,
            category_is_outlier
        )
        return self.measurements, self.outlier_category, self.outlier_category_detailed

    @staticmethod
    def __transform_active_power_output_from_linear_to_original(active_power_output_linear: ndarray,
                                                                this_path_) -> ndarray:
        data_preprocessing_params = load_pkl_file(this_path_ + 'data_preprocessing_params.pkl')
        # 对于区域内的有功功率，进行truncated→linear转换；
        active_power_output_linear = TruncatedToLinear(
            data_preprocessing_params['min_active_power_output'],
            data_preprocessing_params['max_active_power_output']).inverse_transform(active_power_output_linear)
        return active_power_output_linear

    @staticmethod
    def __transform_data_to_linear_for_copula_model(data_to_be_transformed: ndarray, path_, dims: int) -> dict:
        transformed_data = {}.fromkeys(('a', 'b', 'model_boundary', 'a_mask', 'b_mask'))
        # 载入model_boundary数据
        model_boundary = load_npy_file(path_ + 'model_boundary.npy')
        transformed_data['model_boundary'] = model_boundary

        # 确定两个模型的mask
        _, model_a_global_mask, _, model_b_global_mask, _, _, _ = PowerCurve.cal_region_boundary_mask(
            model_boundary, data_to_be_transformed[:, 1])
        transformed_data['a_mask'] = model_a_global_mask
        transformed_data['b_mask'] = model_b_global_mask

        for i, model_this_global_mask in enumerate((model_a_global_mask, model_b_global_mask)):
            # 如果在某个区域没数据的话就continue
            if sum(model_this_global_mask) < 1:
                continue
            # 确定转换数据的预处理（线性化）的参数，理论上来说，这些参数只有在fit模型的时候才能被修改
            this_region = 'a' if i == 0 else 'b'
            this_path_ = path_ + this_region + '/'
            this_transformed_data = np.full((sum(model_this_global_mask), dims), np.nan)

            @load_exist_pkl_file_otherwise_run_and_save(this_path_ + 'data_preprocessing_params.pkl')
            def cal_data_preprocessing_params():
                min_active_power_output = np.nanmin(data_to_be_transformed[model_this_global_mask, 0])
                max_active_power_output = np.nanmax(data_to_be_transformed[model_this_global_mask, 0])
                min_wind_speed = np.nanmin(data_to_be_transformed[model_this_global_mask, 1])
                max_wind_speed = np.nanmax(data_to_be_transformed[model_this_global_mask, 1])

                return {'min_active_power_output': min_active_power_output - 10e8 * float_eps,
                        'max_active_power_output': max_active_power_output + 10e8 * float_eps,
                        'min_wind_speed': min_wind_speed - 10e8 * float_eps,
                        'max_wind_speed': max_wind_speed + 10e8 * float_eps}

            data_preprocessing_params = cal_data_preprocessing_params

            # 对于区域内的有功功率（默认在第0维），进行truncated→linear转换；
            this_transformed_data[:, 0] = TruncatedToLinear(
                data_preprocessing_params['min_active_power_output'],
                data_preprocessing_params['max_active_power_output']).transform(
                data_to_be_transformed[model_this_global_mask, 0])

            # 对于区域内的风速（默认在第1维），进行truncated→linear转换；
            this_transformed_data[:, 1] = TruncatedToLinear(
                data_preprocessing_params['min_wind_speed'],
                data_preprocessing_params['max_wind_speed']).transform(
                data_to_be_transformed[model_this_global_mask, 1])
            # 对于区域内的温度，不做变换
            this_transformed_data[:, 2] = copy.deepcopy(data_to_be_transformed[model_this_global_mask, 2])

            transformed_data[this_region] = this_transformed_data

        return transformed_data

    def __prepare_fitting_data_for_copula_model(self, path_, dims) -> dict:
        """
        准备copula模型的fitting的输入数据
        model_a_global_mask和model_b_global_mask代表两个区域/完全不同的两个模型
        """

        # 确定模型a和模型b的mask，并且储存boundary的计算值
        @load_exist_npy_file_otherwise_run_and_save(path_ + 'model_boundary.npy')
        def identify_model_boundary():
            pc = PowerCurveByMethodOfBins(self.measurements['wind speed'].values[self.outlier_category == 0],
                                          self.measurements['active power output'].values[self.outlier_category == 0])
            return np.array(pc.cal_region_boundary())

        # 将不需要的数据全部置为np.nan
        fitting_data = np.stack((self.measurements['active power output'].values,
                                 self.measurements['wind speed'].values,
                                 self.measurements['environmental temperature'].values),
                                axis=1)
        considered_data_mask = np.stack((self.outlier_category_detailed['active power output'].values == 0,
                                         self.outlier_category_detailed['wind speed'].values == 0,
                                         self.outlier_category_detailed['environmental temperature'].values == 0),
                                        axis=1)
        fitting_data[~considered_data_mask] = np.nan

        return self.__transform_data_to_linear_for_copula_model(fitting_data[:, :3], path_, dims)

    def fit_cvine_gmcm_model(self):
        """
        3维模型。维度分别是active power output, wind speed, environmental temperature
        """
        path_ = self.results_path + '3d_cvine_gmcm_model/' + self.__str__() + '/'
        try_to_find_path_otherwise_make_one((path_, path_ + 'a/', path_ + 'b/'))
        fitting_data = self.__prepare_fitting_data_for_copula_model(path_, 3)
        for this_region, this_fitting_data in fitting_data.items():
            if (this_region != 'a') and (this_region != 'b'):
                continue
            vine_gmcm_copula = VineGMCMCopula(this_fitting_data,
                                              construction=THREE_DIM_CVINE_CONSTRUCTION,
                                              gmcm_model_folder_for_construction_path_=path_ + this_region + '/',
                                              marginal_distribution_file_=path_ + this_region + '/marginal.pkl')
            vine_gmcm_copula.fit()

    def plot_wind_speed_to_active_power_output_scatter(self,
                                                       show_category_as_in_outlier: Union[Tuple[int, ...], str] = None,
                                                       **kwargs):
        title = self.name
        bivariate = Bivariate(self.measurements['wind speed'].values,
                              self.measurements['active power output'].values,
                              predictor_var_name='Wind speed (m/s)',
                              dependent_var_name='Active power output (kW)',
                              category=self.outlier_category)
        bivariate.plot_scatter(show_category=show_category_as_in_outlier,
                               title=title, save_format='png', alpha=0.75,
                               save_file_=self.results_path + self.name + str(
                                   show_category_as_in_outlier), x_lim=(0, 28.5),
                               y_lim=(-self.rated_active_power_output * 0.0125,
                                      self.rated_active_power_output * 1.02), **kwargs)

    def __train_lstm_model_to_forecast(self,
                                       training_set_period: Tuple[datetime.datetime,
                                                                  datetime.datetime],
                                       validation_pct: float,
                                       x_time_step: int,
                                       y_time_step: int,
                                       train_times: int,
                                       *, path_: str,
                                       predictor_var_name_list: List,
                                       dependent_var_name_list: List):
        try_to_find_path_otherwise_make_one(path_)

        temp = copy.copy(self)
        training_validation_set, _, _ = temp.do_truncate(training_set_period[0], training_set_period[1])
        del temp

        x_train, y_train, x_validation, y_validation = prepare_data_for_nn(
            datetime_=training_validation_set[['time']].values,
            x=training_validation_set[predictor_var_name_list].values,
            y=training_validation_set[dependent_var_name_list].values,
            validation_pct=validation_pct,
            x_time_step=x_time_step,
            y_time_step=y_time_step,
            path_=path_,
            including_year=False, including_weekday=False,  # datetime_one_hot_encoder
        )

        for i in range(train_times):
            lstm = MatlabLSTM(path_ + 'training_{}.mat'.format(i))
            lstm.train(x_train, y_train, x_validation, y_validation, int((i + 1.2) * 10000))

    def train_lstm_model_to_forecast_active_power_output(self,
                                                         training_set_period: Tuple[datetime.datetime,
                                                                                    datetime.datetime],
                                                         validation_pct: float = 0.2,
                                                         x_time_step: int = 144 * 28,
                                                         y_time_step: int = 144,
                                                         train_times: int = 6):
        """
        用LSTM网络去做active power output的回归。input维度包括：以前的active power output，wind speed和
        environmental temperature
        :return:
        """
        self.__train_lstm_model_to_forecast(training_set_period=training_set_period,
                                            validation_pct=validation_pct,
                                            x_time_step=x_time_step,
                                            y_time_step=y_time_step,
                                            train_times=train_times,
                                            path_=''.join((self.results_path, 'LSTM/', self.name + '/')),
                                            predictor_var_name_list=['wind speed', 'environmental temperature',
                                                                     'active power output'],
                                            dependent_var_name_list=['active power output'])

    def train_lstm_model_to_forecast_wind_speed(self,
                                                training_set_period: Tuple[datetime.datetime,
                                                                           datetime.datetime],
                                                validation_pct: float = 0.2,
                                                x_time_step: int = 144 * 28,
                                                y_time_step: int = 144,
                                                train_times: int = 6):
        self.__train_lstm_model_to_forecast(training_set_period=training_set_period,
                                            validation_pct=validation_pct,
                                            x_time_step=x_time_step,
                                            y_time_step=y_time_step,
                                            train_times=train_times,
                                            path_=''.join((self.results_path, 'LSTM/', self.name + '/ws/')),
                                            predictor_var_name_list=['wind speed'],
                                            dependent_var_name_list=['wind speed'])

    def train_lstm_model_to_forecast_temperature(self,
                                                 training_set_period: Tuple[datetime.datetime,
                                                                            datetime.datetime],
                                                 validation_pct: float = 0.2,
                                                 x_time_step: int = 144 * 28,
                                                 y_time_step: int = 144,
                                                 train_times: int = 6):
        self.__train_lstm_model_to_forecast(training_set_period=training_set_period,
                                            validation_pct=validation_pct,
                                            x_time_step=x_time_step,
                                            y_time_step=y_time_step,
                                            train_times=train_times,
                                            path_=''.join((self.results_path, 'LSTM/', self.name + '/temperature/')),
                                            predictor_var_name_list=['environmental temperature'],
                                            dependent_var_name_list=['environmental temperature'])

    def train_lstm_model_to_forecast_wind_speed_and_temperature(self,
                                                                training_set_period: Tuple[datetime.datetime,
                                                                                           datetime.datetime],
                                                                validation_pct: float = 0.2,
                                                                x_time_step: int = 144 * 28,
                                                                y_time_step: int = 144,
                                                                train_times: int = 6):
        self.__train_lstm_model_to_forecast(training_set_period=training_set_period,
                                            validation_pct=validation_pct,
                                            x_time_step=x_time_step,
                                            y_time_step=y_time_step,
                                            train_times=train_times,
                                            path_=''.join((self.results_path, 'LSTM/', self.name + '/ws_temperature/')),
                                            predictor_var_name_list=['wind speed', 'environmental temperature'],
                                            dependent_var_name_list=['wind speed', 'environmental temperature'])

    def __test_lstm_model_to_forecast(self,
                                      test_set_period: Tuple[datetime.datetime,
                                                             datetime.datetime],
                                      x_time_step: int,
                                      y_time_step: int,
                                      *, path_: str,
                                      lstm_file_name: str,
                                      predictor_var_name_list: list,
                                      dependent_var_name_list: list):
        temp = copy.copy(self)
        test_set, _, _ = temp.do_truncate(test_set_period[0], test_set_period[1])
        del temp
        x_test, y_test, _, _ = prepare_data_for_nn(
            datetime_=test_set[['time']].values,
            x=test_set[predictor_var_name_list].values,
            y=test_set[dependent_var_name_list].values,
            validation_pct=0,
            x_time_step=x_time_step,
            y_time_step=y_time_step,
            path_=path_,
            including_year=False, including_weekday=False,  # datetime_one_hot_encoder
        )
        lstm = MatlabLSTM(path_ + lstm_file_name)
        lstm_predict = lstm.test(x_test)

        ax = series(y_test.flatten())
        ax = series(lstm_predict.flatten(), ax=ax)

        return lstm_predict

    def test_lstm_model_to_forecast_active_power_output(self,
                                                        test_set_period: Tuple[datetime.datetime,
                                                                               datetime.datetime],
                                                        x_time_step: int = 144 * 28,
                                                        y_time_step: int = 144,
                                                        *, lstm_file_name: str):
        results = self.__test_lstm_model_to_forecast(test_set_period=test_set_period,
                                                     x_time_step=x_time_step,
                                                     y_time_step=y_time_step,
                                                     path_=''.join((self.results_path, 'LSTM/', self.name + '/')),
                                                     lstm_file_name=lstm_file_name,
                                                     predictor_var_name_list=['wind speed', 'environmental temperature',
                                                                              'active power output'],
                                                     dependent_var_name_list=['active power output'])
        return results

    def test_lstm_model_to_forecast_wind_speed(self,
                                               test_set_period: Tuple[datetime.datetime,
                                                                      datetime.datetime],
                                               x_time_step: int = 144 * 28,
                                               y_time_step: int = 144,
                                               *, lstm_file_name: str):
        results = self.__test_lstm_model_to_forecast(test_set_period=test_set_period,
                                                     x_time_step=x_time_step,
                                                     y_time_step=y_time_step,
                                                     path_=''.join((self.results_path, 'LSTM/', self.name + '/ws/')),
                                                     lstm_file_name=lstm_file_name,
                                                     predictor_var_name_list=['wind speed'],
                                                     dependent_var_name_list=['wind speed'])
        return results

    def test_lstm_model_to_forecast_temperature(self,
                                                test_set_period: Tuple[datetime.datetime,
                                                                       datetime.datetime],
                                                x_time_step: int = 144 * 28,
                                                y_time_step: int = 144,
                                                *, lstm_file_name: str):
        results = self.__test_lstm_model_to_forecast(test_set_period=test_set_period,
                                                     x_time_step=x_time_step,
                                                     y_time_step=y_time_step,
                                                     path_=''.join(
                                                         (self.results_path, 'LSTM/', self.name + '/temperature/')),
                                                     lstm_file_name=lstm_file_name,
                                                     predictor_var_name_list=['environmental temperature'],
                                                     dependent_var_name_list=['environmental temperature'])
        return results

    def test_lstm_model_to_forecast_wind_speed_and_temperature(self,
                                                               test_set_period: Tuple[datetime.datetime,
                                                                                      datetime.datetime],
                                                               x_time_step: int = 144 * 28,
                                                               y_time_step: int = 144,
                                                               *, lstm_file_name: str):
        results = self.__test_lstm_model_to_forecast(test_set_period=test_set_period,
                                                     x_time_step=x_time_step,
                                                     y_time_step=y_time_step,
                                                     path_=''.join((self.results_path, 'LSTM/',
                                                                    self.name + '/ws_temperature/')),
                                                     lstm_file_name=lstm_file_name,

                                                     predictor_var_name_list=['wind speed',
                                                                              'environmental temperature'],
                                                     dependent_var_name_list=['wind speed',
                                                                              'environmental temperature'])
        return results

    def cal_measurements_from_wind_turbine_objects(self,
                                                   wt_obj_in_iterator: Tuple[WT, ...],
                                                   considered_outlier_category: ndarray):
        """
        从一组wind turbine对象生成它们组成的wind farm对象。这是个naive的aggregate方法。不包含wind farm的filling missing方法。
        :param wt_obj_in_iterator:
        :param considered_outlier_category:
        :return:
        """
        pass
