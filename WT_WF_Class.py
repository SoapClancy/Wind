from Regression_Analysis.DeepLearning_Class import datetime_one_hot_encoder, MatlabLSTM, \
    prepare_data_for_nn
import numpy as np
import pandas as pd
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple, np_datetime64_to_datetime
from File_Management.load_save_Func import *
from BivariateAnalysis_Class import BivariateOutlier, Bivariate, MethodOfBins
from File_Management.path_and_file_management_Func import try_to_find_folder_path_otherwise_make_one, try_to_find_file
from UnivariateAnalysis_Class import CategoryUnivariate, UnivariatePDFOrCDFLike, UnivariateGaussianMixtureModel, \
    DeterministicUnivariateProbabilisticModel
from typing import Union, Tuple, List, Iterable
from BivariateAnalysis_Class import Bivariate, MethodOfBins
from Ploting.fast_plot_Func import *
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
from PhysicalInstance_Class import PhysicalInstanceDataFrame, PhysicalInstanceSeries, PhysicalInstance
from pathlib import Path
from project_utils import project_path_, WS_POUT_SCATTER_ALPHA, WS_POUT_2D_PLOT_KWARGS, WS_POUT_SCATTER_SIZE
from collections import ChainMap
import warnings
import re
from Filtering.OutlierAnalyser_Class import DataCategoryNameMapper, DataCategoryData
from Filtering.sklearn_novelty_and_outlier_detection_Func import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ConvenientDataType import UncertaintyDataFrame
from tqdm import tqdm


class WTandWFBase(PhysicalInstanceDataFrame):
    results_path = project_path_ / 'Data/Results/'  # type: Path

    __slots__ = ("cut_in_wind_speed", "cut_out_wind_speed", "rated_active_power_output")

    @property
    def _constructor(self):
        return super()._constructor

    @property
    def _constructor_expanddim(self):
        return super()._constructor_expanddim

    @property
    def _constructor_sliced(self):
        return super()._constructor_sliced

    def __init__(self, *args, cut_in_wind_speed=4, rated_active_power_output, cut_out_wind_speed=25, **kwargs):
        super().__init__(*args, **kwargs)
        self.cut_in_wind_speed = cut_in_wind_speed
        self.cut_out_wind_speed = cut_out_wind_speed
        self.rated_active_power_output = rated_active_power_output
        if 'active power output' in self.columns:
            # Make all -0.02 p.u. ~ 0 p.u. Pout to be equal to 0 p.u
            self.loc[self['active power output'].between(*self.rated_active_power_output * np.array([-0.02, 0])),
                     'active power output'] = 0
            # Make all 1.0 p.u. ~ 1.02 p.u. Pout to be equal to 1.0 p.u.
            self.loc[self['active power output'].between(*self.rated_active_power_output * np.array([1, 1.02])),
                     'active power output'] = self.rated_active_power_output

    def plot(self, *,
             ax=None,
             plot_mfr: Iterable[PowerCurveByMfr] = None,
             plot_scatter_pc: bool = False,
             **kwargs):
        ax = scatter(self['wind speed'].values,
                     self['active power output'].values / self.rated_active_power_output,
                     ax=ax,
                     **dict(ChainMap(kwargs,
                                     WS_POUT_2D_PLOT_KWARGS,
                                     {'alpha': WS_POUT_SCATTER_ALPHA,
                                      's': WS_POUT_SCATTER_SIZE,
                                      'color': 'royalblue'}
                                     ))
                     )
        if plot_mfr:
            for this_mfr_pc in plot_mfr:
                ax = this_mfr_pc.plot(ax=ax)
        if plot_scatter_pc:
            ax = PowerCurveByMethodOfBins(self['wind speed'].values,
                                          self['active power output'].values / self.rated_active_power_output).plot(
                ax=ax,
                plot_recording=False,
                **dict(ChainMap(kwargs, WS_POUT_2D_PLOT_KWARGS))
            )
        return ax

    def update_air_density_to_last_column(self):
        if 'air density' in self.columns:
            return self['air density'].values
        else:
            from Wind_Class import cal_air_density, celsius_to_kelvin
            air_density = cal_air_density(celsius_to_kelvin(self['environmental temperature'].values),
                                          self['relative humidity'].values / 100,
                                          self['barometric pressure'].values * 100)
            self['air density'] = air_density
            return air_density


class WT(WTandWFBase):

    def __init__(self, *args, rated_active_power_output=3000, **kwargs):
        super().__init__(*args, rated_active_power_output=rated_active_power_output, **kwargs)

    @property
    def data_category_name_mapper(self) -> DataCategoryNameMapper:
        meta = [["missing data", "missing", -1, "N/A"],
                ["normal data", "normal", 0, "N/A"],
                ["Low Pout-high WS", "CAT-I.a", 1, "due to WT cut-out effects"],
                ["Low Pout-high WS", "CAT-I.b", 2, "caused by the other sources"],
                ["Low maximum Pout", "CAT-II", 3, "curtailment"],
                ["Linear Pout-WS", "CAT-III", 4, "e.g., constant WS-variable Pout"],
                ["Scattered", "CAT-IV.a", 5, "averaging window or WT cut-out effects"],
                ["Scattered", "CAT-IV.b", 6, "the others"]]

        mapper = DataCategoryNameMapper.init_from_template(rows=len(meta))
        mapper[:] = meta
        return mapper

    @property
    def default_results_saving_path(self):
        return {
            "outlier": self.results_path / f"Filtering/{self.__str__()}/results.pkl"
        }

    def outlier_detector(self, how_to_detect_scattered: str = 'isolation forest', *,
                         save_file_path: Path = None) -> DataCategoryData:
        assert (how_to_detect_scattered in ('isolation forest', 'hist')), "Check 'how_to_detect_scattered'"
        save_file_path = save_file_path or self.default_results_saving_path["outlier"]
        try_to_find_folder_path_otherwise_make_one(save_file_path.parent)
        if try_to_find_file(save_file_path):
            warnings.warn(f"{self.__str__()} has results in {save_file_path}")
            return load_pkl_file(save_file_path)['DataCategoryData obj']
        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        outlier = super().outlier_detector()  # type: DataCategoryData
        # %% CAT-I
        cat_i_outlier_mask = self.data_category_inside_boundary(
            {'wind speed': (self.cut_in_wind_speed, self.cut_out_wind_speed),
             'active power output': (-np.inf, 0)}
        )
        outlier.abbreviation[cat_i_outlier_mask] = "CAT-I"
        del cat_i_outlier_mask
        # %% CAT-II
        cat_ii_outlier_mask = self.data_category_is_linearity(
            '30T',
            constant_error={'active power output': self.rated_active_power_output * 0.0005}
        )
        cat_ii_outlier_mask = np.bitwise_and(
            cat_ii_outlier_mask,
            self.data_category_inside_boundary(
                {'active power output': (self.rated_active_power_output * 0.1, self.rated_active_power_output * 0.9)}
            )
        )
        outlier.abbreviation[cat_ii_outlier_mask] = "CAT-II"
        del cat_ii_outlier_mask
        # %% CAT-III
        cat_iii_outlier_mask = self.data_category_is_linearity('60T', constant_error={'wind speed': 0.01})
        outlier.abbreviation[cat_iii_outlier_mask] = "CAT-III"
        del cat_iii_outlier_mask
        # %% CAT-IV
        current_normal_index = np.argwhere(outlier.abbreviation == 'normal').flatten()
        if how_to_detect_scattered == 'hist':
            cat_iv_outlier_mask = BivariateOutlier(
                predictor_var=self.iloc[current_normal_index]['wind speed'].values,
                dependent_var=self.iloc[current_normal_index]['active power output'].values,
                bin_step=0.5
            ).identify_interquartile_outliers_based_on_method_of_bins()
        else:
            current_normal_data = StandardScaler().fit_transform(self.concerned_data().iloc[current_normal_index])
            cat_iv_outlier_mask = use_isolation_forest(current_normal_data)
            del current_normal_data

        # cat_iv_outlier_mask[100:] = False  # #######################################################################
        outlier.abbreviation[current_normal_index[cat_iv_outlier_mask]] = "CAT-IV"
        del current_normal_index, cat_iv_outlier_mask
        # %% CAT-IV.a and CAT-IV.b
        outlier.abbreviation[np.all((
            (
                outlier.abbreviation == "CAT-IV",
                self['active power output'] >= PowerCurveByMfr.init_all_instances_in_docs()[0](self['wind speed']),
                self['active power output'] <= PowerCurveByMfr.init_all_instances_in_docs()[-1](self['wind speed'])
            )
        ), axis=0)] = "CAT-IV.a"
        outlier.abbreviation[np.bitwise_and(outlier.abbreviation == "CAT-IV",
                                            np.isnan(self['wind speed std.']))] = "CAT-IV.b"
        self.update_air_density_to_last_column()
        current_cat_iv_index = np.argwhere(outlier.abbreviation == 'CAT-IV').flatten()
        current_cat_iv_data = self[['wind speed',
                                    'wind speed std.',
                                    'active power output',
                                    'air density']].iloc[current_cat_iv_index]
        current_cat_iv_data['Mfr-PC obj'] = PowerCurveByMfr.init_multiple_instances(
            air_density=current_cat_iv_data['air density'].values)
        # To find the recordings with same wind speed, wind speed std., and air density. This will increase the
        # computation efficiency in further simulation
        unique_rows, unique_label = current_cat_iv_data.unique(['wind speed', 'wind speed std.', 'Mfr-PC obj'])
        # Do the simulation (ONLY in unique_rows)
        from Wind_Class import Wind
        # Use the buffer
        buffer = try_to_find_file(save_file_path.parent / 'temp.pkl')
        if buffer:
            buffer = load_pkl_file(save_file_path.parent / 'temp.pkl')
            cat_iv_a_outlier_index = buffer['cat_iv_a_outlier_index']
            cat_iv_b_outlier_index = buffer['cat_iv_b_outlier_index']
        else:
            cat_iv_a_outlier_index, cat_iv_b_outlier_index = [], []
        for i in tqdm(range(unique_rows.shape[0])):
            # Use the buffer
            if buffer:
                if i <= buffer['i']:
                    continue
            this_unique_row = unique_rows.iloc[i]
            # prepare high resolution wind
            wind = Wind(this_unique_row['wind speed'], this_unique_row['wind speed std.'])
            high_resol_wind = wind.simulate_transient_wind_speed_time_series(
                resolution=10,
                traces_number_for_each_recording=1_000_000,
            )
            # prepare mfr-pc
            this_unique_mfr_pc = this_unique_row['Mfr-PC obj']
            this_pout_uncertainty = this_unique_mfr_pc.cal_with_hysteresis_control_using_high_resol_wind(
                high_resol_wind,
                return_percentiles=UncertaintyDataFrame.init_from_template(
                    columns_number=len(high_resol_wind),
                    percentiles=None),
            )
            # Compare to judge. Note that this is a kind of inverse of unique
            pout_actual = current_cat_iv_data[unique_label == i]
            for j in range(pout_actual.shape[0]):
                this_pout_actual = pout_actual['active power output'].iloc[j] / self.rated_active_power_output
                if all((this_pout_actual >= this_pout_uncertainty.loc['3_Sigma_low', 0],
                        this_pout_actual <= this_pout_uncertainty.loc['3_Sigma_high', 0])):
                    cat_iv_a_outlier_index.append(current_cat_iv_data[unique_label == i].index[j])
                elif all((this_pout_actual >= this_pout_uncertainty.loc['3_Sigma_low', 0],
                          this_pout_actual <= float(this_unique_mfr_pc(this_unique_row['wind speed'])))):
                    cat_iv_a_outlier_index.append(current_cat_iv_data[unique_label == i].index[j])
                else:
                    cat_iv_b_outlier_index.append(current_cat_iv_data[unique_label == i].index[j])
            # Save progress
            if (i % 25) == 0:
                save_pkl_file(save_file_path.parent / 'temp.pkl',
                              {"i": i,
                               "outlier": outlier,
                               "cat_iv_a_outlier_index": cat_iv_a_outlier_index,
                               "cat_iv_b_outlier_index": cat_iv_b_outlier_index})
        outlier.abbreviation[self.index.isin(cat_iv_a_outlier_index)] = "CAT-IV.a"
        outlier.abbreviation[self.index.isin(cat_iv_b_outlier_index)] = "CAT-IV.b"
        # %% CAT-I.a and CAT-I.b
        cat_iv_a_outlier_index = self.index[outlier.abbreviation == "CAT-IV.a"]
        for i in range(cat_iv_a_outlier_index.shape[0] - 1):
            this_window_mask = np.bitwise_and(self.index > cat_iv_a_outlier_index[i],
                                              self.index < cat_iv_a_outlier_index[i + 1])
            if all((np.unique(outlier.abbreviation[this_window_mask]).shape[0] == 1,
                    "CAT-I" in outlier.abbreviation[this_window_mask])):
                outlier.abbreviation[this_window_mask] = "CAT-I.a"
        outlier.abbreviation[outlier.abbreviation == "CAT-I"] = "CAT-I.b"
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        save_pkl_file(save_file_path,
                      {'raw_ndarray_data': np.array(outlier.abbreviation),
                       'raw_index': outlier.index,
                       'DataCategoryData obj': outlier})
        return outlier

    def outlier_plot(self, outlier: DataCategoryData = None, ax=None, *, plot_individual: bool = False):
        outlier = outlier or load_pkl_file(self.default_results_saving_path["outlier"])['DataCategoryData obj']
        self["active power output"] /= self.rated_active_power_output

        if sum(outlier("CAT-I.a")) > 0:
            ax = scatter(*self[outlier("CAT-I.a")][["wind speed", "active power output"]].values.T, label="CAT-I.a",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="darkorange", marker="1", s=24, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-I.b")) > 0:
            ax = scatter(*self[outlier("CAT-I.b")][["wind speed", "active power output"]].values.T, label="CAT-I.b",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="fuchsia", marker="2", s=24, zorder=8, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-II")) > 0:
            ax = scatter(*self[outlier("CAT-II")][["wind speed", "active power output"]].values.T, label="CAT-II",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="black", marker="x", s=16, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-III")) > 0:
            ax = scatter(*self[outlier("CAT-III")][["wind speed", "active power output"]].values.T, label="CAT-III",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="cyan", marker="+", s=24, zorder=11, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-IV.a")) > 0:
            ax = scatter(*self[outlier("CAT-IV.a")][["wind speed", "active power output"]].values.T, label="CAT-IV.a",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="green", marker="3", s=24, zorder=9, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-IV.b")) > 0:
            ax = scatter(*self[outlier("CAT-IV.b")][["wind speed", "active power output"]].values.T, label="CAT-IV.b",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="red", marker="4", s=24, **WS_POUT_2D_PLOT_KWARGS)
        ax = scatter(*self[outlier("normal")][["wind speed", "active power output"]].values.T, label="Others",
                     ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                     color="royalblue", zorder=10, **WS_POUT_2D_PLOT_KWARGS)
        self["active power output"] *= self.rated_active_power_output

        return ax

    def outlier_report(self, outlier: DataCategoryData = None):
        outlier = outlier or load_pkl_file(self.defaoult_results_saving_path["outlier"])['DataCategoryData obj']

        report_pd = pd.DataFrame(index=np.unique(outlier.abbreviation),
                                 columns=['number', 'percentage'], dtype=float)
        for this_outlier in np.unique(outlier.abbreviation):
            this_outlier_number = sum(outlier(this_outlier))
            report_pd.loc[this_outlier, 'number'] = this_outlier_number
            report_pd.loc[this_outlier, 'percentage'] = this_outlier_number / outlier.abbreviation.shape[0] * 100
        report_pd.rename(index={"normal": "others"}, inplace=True)
        bar(report_pd.index, report_pd['percentage'].values, y_label="Recording percentage [%]",
            autolabel_format="{:.2f}", y_lim=(-1, 85))
        plt.xticks(rotation=45)

        bar(report_pd.index, report_pd['number'].values, y_label="Recording number",
            autolabel_format="{:.0f}", y_lim=(-1, np.max(report_pd['number'].values) * 1.2))
        plt.xticks(rotation=45)

        report_pd.to_csv(self.default_results_saving_path["outlier"].parent / "report.csv")

    def get_current_season(self, season_template: Enum = SeasonTemplate1) -> tuple:
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        synchronous = SynchronousTimeSeriesData(self.measurements,
                                                self.outlier_category,
                                                self.outlier_category_detailed)
        return synchronous.get_current_season(season_template=season_template)

    def do_truncate(self, start_time: datetime.datetime = None, end_time: datetime.datetime = None):
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        synchronous = SynchronousTimeSeriesData(self.measurements,
                                                self.outlier_category,
                                                self.outlier_category_detailed)
        self.measurements, self.outlier_category, self.outlier_category_detailed = synchronous.do_truncate(
            start_time=start_time,
            end_time=end_time
        )

    def do_truncate_by_season(self, season_to_be_queried: str, season_template: Enum = SeasonTemplate1):
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        synchronous = SynchronousTimeSeriesData(self.measurements,
                                                self.outlier_category,
                                                self.outlier_category_detailed)
        self.measurements, self.outlier_category, self.outlier_category_detailed = synchronous.do_truncate_by_season(
            season_to_be_queried=season_to_be_queried,
            season_template=season_template
        )

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
                min_absolute_wind_direction_in_truncated = -np.sqrt(2) - 10e8 * float_eps
                max_absolute_wind_direction_in_truncated = np.sqrt(2) + 10e8 * float_eps
                return {'min_active_power_output': min_active_power_output - 10e8 * float_eps,
                        'max_active_power_output': max_active_power_output + 10e8 * float_eps,
                        'min_wind_speed': min_wind_speed - 10e8 * float_eps,
                        'max_wind_speed': max_wind_speed + 10e8 * float_eps,
                        'min_absolute_wind_direction_in_truncated': min_absolute_wind_direction_in_truncated,
                        'max_absolute_wind_direction_in_truncated': max_absolute_wind_direction_in_truncated}

            data_preprocessing_params = cal_data_preprocessing_params

            # 对于区域内的有功功率，进行truncated→linear转换；
            this_transformed_data[:, 0] = TruncatedToLinear(
                data_preprocessing_params['min_active_power_output'],
                data_preprocessing_params['max_active_power_output']).transform(
                data_to_be_transformed[model_this_global_mask, 0])
            # 对于区域内的风速，进行truncated→linear转换；
            if dims >= 2:
                this_transformed_data[:, 1] = TruncatedToLinear(
                    data_preprocessing_params['min_wind_speed'],
                    data_preprocessing_params['max_wind_speed']).transform(
                    data_to_be_transformed[model_this_global_mask, 1])
            # 对于区域内的风向，进行circular→truncated→linear转换
            if dims >= 3:
                this_transformed_data[:, 2] = CircularToLinear(
                    data_preprocessing_params['min_absolute_wind_direction_in_truncated'],
                    data_preprocessing_params['max_absolute_wind_direction_in_truncated'], 360).transform(
                    data_to_be_transformed[model_this_global_mask, 2])
            # 对于区域内的温度，不做变换
            if dims >= 4:
                this_transformed_data[:, 3] = copy.deepcopy(data_to_be_transformed[model_this_global_mask, 3])
            # 将该区域的结果写入最终结果
            transformed_data[this_region] = this_transformed_data
        return transformed_data

    def __transform_data_to_linear_for_2d_gmcm_model(self, data_to_be_transformed: ndarray, path_) -> dict:
        return self.__transform_data_to_linear_for_copula_model(data_to_be_transformed, path_, 2)

    def __transform_data_to_linear_for_3d_vine_gmcm_model(self, data_to_be_transformed: ndarray, path_) -> dict:
        return self.__transform_data_to_linear_for_copula_model(data_to_be_transformed, path_, 3)

    def __transform_data_to_linear_for_4d_vine_gmcm_model(self, data_to_be_transformed: ndarray, path_) -> dict:
        return self.__transform_data_to_linear_for_copula_model(data_to_be_transformed, path_, 4)

    def __prepare_fitting_data_for_vine_gmcm_model(self, path_, dims) -> dict:
        """
        准备vine_gmcm的fitting 数据
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
                                 self.measurements['absolute wind direction'].values,
                                 self.measurements['environmental temperature'].values),
                                axis=1)
        considered_data_mask = np.stack((self.outlier_category_detailed['active power output'].values == 0,
                                         self.outlier_category_detailed['wind speed'].values == 0,
                                         self.outlier_category_detailed['absolute wind direction'].values == 0,
                                         self.outlier_category_detailed['environmental temperature'].values == 0),
                                        axis=1)
        fitting_data[~considered_data_mask] = np.nan

        # 转换数据
        if dims == 4:
            return self.__transform_data_to_linear_for_4d_vine_gmcm_model(fitting_data, path_)
        elif dims == 3:
            return self.__transform_data_to_linear_for_3d_vine_gmcm_model(fitting_data[:, :3], path_)

    def __prepare_fitting_data_for_4d_vine_gmcm_model(self, path_):
        return self.__prepare_fitting_data_for_vine_gmcm_model(path_, 4)

    def __prepare_fitting_data_for_3d_vine_gmcm_model(self, path_):
        return self.__prepare_fitting_data_for_vine_gmcm_model(path_, 3)

    def fit_4d_cvine_gmcm_model(self):
        """
        对于4维的vine模型，只考虑每个pair copula对应的两个变量对应的outlier_category_detailed是0的情况。
        维度的名字依次是：'active power output', 'wind speed', 'absolute wind direction', 'environmental temperature'。
        因为有两个区域，所以其实本质上是两个独立的4d_vine_gmcm_model
        """
        path_ = self.results_path + '4d_cvine_gmcm_model/' + self.__str__() + '/'
        try_to_find_folder_path_otherwise_make_one((path_, path_ + 'a/', path_ + 'b/'))
        fitting_data = self.__prepare_fitting_data_for_4d_vine_gmcm_model(path_)
        for this_region, this_fitting_data in fitting_data.items():
            if (this_region != 'a') and (this_region != 'b'):
                continue
            vine_gmcm_copula = VineGMCMCopula(this_fitting_data,
                                              construction=FOUR_DIM_CVINE_CONSTRUCTION,
                                              gmcm_model_folder_for_construction_path_=path_ + this_region + '/',
                                              marginal_distribution_file_=path_ + this_region + '/marginal.pkl')
            vine_gmcm_copula.fit()

    def fit_3d_cvine_gmcm_model(self, use_ws_ahead: int = 0):
        """
        对于3维的vine模型，只考虑每个pair copula对应的两个变量对应的outlier_category_detailed是0的情况。
        维度的名字依次是：'active power output', 'wind speed', 'absolute wind direction'。
        因为有两个区域，所以其实本质上是两个独立的4d_vine_gmcm_model
        """
        if use_ws_ahead == 0:
            path_ = self.results_path + '3d_cvine_gmcm_model/' + self.__str__() + '/'
        else:
            path_ = self.results_path + '3d_cvine_gmcm_model_use_ws_ahead_{}/'.format(use_ws_ahead) + \
                    self.__str__() + '/'

        try_to_find_folder_path_otherwise_make_one((path_, path_ + 'a/', path_ + 'b/'))
        fitting_data = self.__prepare_fitting_data_for_3d_vine_gmcm_model(path_)

        for this_region, this_fitting_data in fitting_data.items():
            # DEBUG: For the IET paper, only consider region_a
            if this_region != 'a':
                continue

            if use_ws_ahead != 0:
                this_fitting_data[:, 2] = np.roll(this_fitting_data[:, 2], -1)
                this_fitting_data[-1, 2] = np.nan

            vine_gmcm_copula = VineGMCMCopula(this_fitting_data,
                                              construction=THREE_DIM_CVINE_CONSTRUCTION,
                                              gmcm_model_folder_for_construction_path_=path_ + this_region + '/',
                                              marginal_distribution_file_=path_ + this_region + '/marginal.pkl')
            vine_gmcm_copula.fit()
            # DEBUG: For the IET paper
            # flag = np.bitwise_and(self.outlier_category_detailed['wind speed'].values == 0,
            #                       self.outlier_category_detailed['absolute wind direction'].values == 0)
            # ws = self.measurements['wind speed'].values[flag]
            # wd = self.measurements['absolute wind direction'].values[flag]
            # GMCM(gmcm_model_file_=path_ + this_region + '/GMCM_(2, 3).mat',
            #      ndarray_data=np.stack((ws, wd), axis=1),
            #      marginal_distribution_file_=path_ + this_region + '/marginal_for_GMCM_(2, 3).pkl',
            #      gmcm_fitting_k=8,
            #      gmcm_max_fitting_iteration=2500,
            #      gmcm_fitting_attempt=1,
            #      )

    def fit_2d_conditional_probability_model_by_gmm(self, *, bin_step: float, gmm_args: dict = None, **kwargs):
        gmm_args = gmm_args or {}
        _path = kwargs.get('_path') or (self.results_path + '2d_conditional_probability_by_gmm/' + self.__str__() + \
                                        f' bin_step={bin_step}/')

        try_to_find_folder_path_otherwise_make_one(_path)
        mask = np.bitwise_or(self.outlier_category == 0,
                             self.outlier_category == 5)
        bivariate = Bivariate(self.measurements['wind speed'].values[mask],
                              self.measurements['active power output'].values[mask],
                              bin_step=bin_step)

        # TODO 全部换成Path类
        @load_exist_pkl_file_otherwise_run_and_save(_path + (kwargs.get('model_name') or 'model.pkl'))
        def load_or_make():
            return bivariate.fit_mob_using_gaussian_mixture_model(**gmm_args)

        load_or_make()

    def estimate_active_power_output_by_2d_conditional_probability_model_by_gmm(self,
                                                                                wind_speed_ndarray: ndarray,
                                                                                *, bin_step: float,
                                                                                if_no_available_mode: Union[int, str] =
                                                                                'nearest_not_none_bin_keys') -> \
            Tuple[UnivariateGaussianMixtureModel, ...]:
        path_ = self.results_path + '2d_conditional_probability_by_gmm/' + self.__str__() + \
                ' bin_step={}/'.format(bin_step)
        model = load_pkl_file(path_ + 'model.pkl')
        # 计算(其实是选择)输出的条件概率模型
        power_output_model = []
        for this_predictor_var in wind_speed_ndarray:
            this_model_idx = MethodOfBins.find_mob_key_according_to_mob_or_mob_fitting_like_dict(this_predictor_var,
                                                                                                 model)
            if if_no_available_mode == 'nearest_not_none_bin_keys':
                power_output_model.append(
                    UnivariateGaussianMixtureModel(model[this_model_idx['nearest_not_none_bin_keys']]
                                                   ['this_bin_probability_model']))
            else:
                assert isinstance(if_no_available_mode, int)
                temp = []
                for i in range(if_no_available_mode):
                    temp = UnivariateGaussianMixtureModel(model[this_model_idx['not_none_bin_keys'][i]]
                                                          ['this_bin_probability_model'])
                power_output_model.append(temp)
        return tuple(power_output_model)

    @staticmethod
    def estimate_active_power_output_by_mfr_power_curve(wind_speed_ndarray: ndarray):
        return PowerCurveByMfr()(wind_speed_ndarray)

    def __add_active_power_output_dim_for_copula_based_estimating_method(self, input_ndarray: ndarray,
                                                                         linspace_number: int) -> ndarray:
        """
        因为estimate的时候其实是条件概率，它们的输入少了第一维度（即：active power output），所以在将数据放入联合概率模型
        之前要补充一个维度
        """
        active_power_output_dim = np.linspace(-30, self.rated_active_power_output + 30, linspace_number)
        active_power_output_dim = np.tile(active_power_output_dim, input_ndarray.shape[0])
        input_ndarray = np.repeat(input_ndarray, linspace_number, 0)
        return np.concatenate((active_power_output_dim.reshape(-1, 1), input_ndarray), axis=1)

    @staticmethod
    def __transform_estimating_method_results_to_normalised_pdf_like(
            unnormalised_pdf_like: ndarray, linspace_number: int) -> Tuple[UnivariatePDFOrCDFLike, ...]:
        normalised_pdf_like = []
        for i in range(0, unnormalised_pdf_like.shape[0], linspace_number):
            this_normalised_pdf_like = UnivariatePDFOrCDFLike(
                pdf_like_ndarray=unnormalised_pdf_like[i:i + linspace_number, :])
            normalised_pdf_like.append(this_normalised_pdf_like)
        return tuple(normalised_pdf_like)

    def __estimate_active_power_output_by_copula_model(self, input_ndarray: ndarray, path_, dims) -> tuple:
        if input_ndarray.ndim == 1:
            input_ndarray = np.expand_dims(input_ndarray, 1)
        estimated_active_power_output_pdf_like = np.array([None for _ in range(input_ndarray.shape[0])])
        # 设置模型精度，并且准备增维以估计联合概率，计算region的mask
        linspace_number = 500
        input_ndarray_modify = self.__add_active_power_output_dim_for_copula_based_estimating_method(
            input_ndarray, linspace_number)
        if dims == 4:
            prepared_data = self.__transform_data_to_linear_for_4d_vine_gmcm_model(input_ndarray_modify, path_)
            this_construction = FOUR_DIM_CVINE_CONSTRUCTION
        elif dims == 3:
            prepared_data = self.__transform_data_to_linear_for_3d_vine_gmcm_model(input_ndarray_modify, path_)
            this_construction = THREE_DIM_CVINE_CONSTRUCTION
        elif dims == 2:
            prepared_data = self.__transform_data_to_linear_for_2d_gmcm_model(input_ndarray_modify, path_)
            this_construction = None
        else:
            raise Exception("Unsupported dims")
        (region_1_mask, region_a_mask_in_input_data, region_rated_mask,
         region_b_mask_in_input_data, region_5_mask, hard_rated_mask, hard_cut_off_mask) = \
            PowerCurve.cal_region_boundary_mask(prepared_data['model_boundary'], input_ndarray[:, 0])
        # 对于region_a和region_b，采用高级不确定模型去估计
        for this_region, this_prepared_data in prepared_data.items():
            """
            DEBUG for IET
            """
            # DEBUG for IET
            if this_region == 'b':
                continue

            if ((this_region != 'a') and (this_region != 'b')) or (this_prepared_data is None):
                continue
            # 只有对于3维或以上的模型才用VineGMCMCopula
            if dims >= 3:
                vine_gmcm_copula = VineGMCMCopula(construction=this_construction,
                                                  gmcm_model_folder_for_construction_path_=path_ + this_region + '/',
                                                  marginal_distribution_file_=path_ + this_region + '/marginal.pkl')
                pdf_ = vine_gmcm_copula.cal_joint_pdf(ndarray_data_like=this_prepared_data)
            else:
                gmcm_copula = GMCM(gmcm_model_file_=path_ + this_region + '/GMCM_(1, 2).mat',
                                   marginal_distribution_file_=path_ + this_region + '/marginal.pkl')
                pdf_ = gmcm_copula.cal_joint_pdf(ndarray_data_like=this_prepared_data)
            pdf_ = np.stack((pdf_, input_ndarray_modify[prepared_data[this_region + '_mask'], 0]), axis=1)
            pdf_ = self.__transform_estimating_method_results_to_normalised_pdf_like(pdf_, linspace_number)
            if this_region == 'a':
                estimated_active_power_output_pdf_like[region_a_mask_in_input_data] = pdf_
            else:
                estimated_active_power_output_pdf_like[region_b_mask_in_input_data] = pdf_
        """
        DEBUG for IET
        """
        # DEBUG for IET
        estimated_active_power_output_pdf_like[region_1_mask] = DeterministicUnivariateProbabilisticModel(0.)
        estimated_active_power_output_pdf_like[hard_rated_mask] = DeterministicUnivariateProbabilisticModel(
            1. * self.rated_active_power_output)
        estimated_active_power_output_pdf_like[hard_cut_off_mask] = DeterministicUnivariateProbabilisticModel(0.)

        return tuple(estimated_active_power_output_pdf_like)

    def estimate_active_power_output_by_2d_gmcm_model(self, input_ndarray: ndarray) -> tuple:
        """
        运用2维的GMCM模型去估计有功输出。
        :param input_ndarray 2维。维度的名称依次是：'wind speed'
        :return: UnivariatePDFLike组成的tuple
        """
        path_ = self.results_path + '2d_gmcm_model/' + self.__str__() + '/'
        return self.__estimate_active_power_output_by_copula_model(input_ndarray, path_, 2)

    def estimate_active_power_output_by_2d_gmcm_model_with_uncertain_inputs(self, input_ndarray: ndarray) -> tuple:
        path_ = self.results_path + '2d_gmcm_model/' + self.__str__() + '/'
        estimated_active_power_output_pdf_like = np.array([None for _ in range(input_ndarray.shape[0])])

        linspace_number = 500
        # 准备下界
        input_ndarray_modify_with_lower = self.__add_active_power_output_dim_for_copula_based_estimating_method(
            input_ndarray[:, [0]], linspace_number)
        prepared_data_with_lower = self.__transform_data_to_linear_for_2d_gmcm_model(input_ndarray_modify_with_lower,
                                                                                     path_)
        # 准备上界
        input_ndarray_modify_with_upper = self.__add_active_power_output_dim_for_copula_based_estimating_method(
            input_ndarray[:, [1]], linspace_number)
        prepared_data_with_higher = self.__transform_data_to_linear_for_2d_gmcm_model(input_ndarray_modify_with_upper,
                                                                                      path_)
        prepared_data = np.stack((prepared_data_with_lower, prepared_data_with_higher))

        # 开始计算
        """
        停止！！！因为不知道用哪个boundary！！！这是模型不连续的坏处
        """
        (region_1_mask, region_a_mask_in_input_data, region_rated_mask,
         region_b_mask_in_input_data, region_5_mask, hard_rated_mask, hard_cut_off_mask) = \
            PowerCurve.cal_region_boundary_mask(prepared_data['model_boundary'], input_ndarray[:, 0])

    def estimate_active_power_output_by_3d_cvine_gmcm_model(self, input_ndarray: ndarray, use_ws_ahead=0) -> tuple:
        """
        运用3维的vine模型去估计有功输出。
        :param input_ndarray 3维。维度的名称依次是：'wind speed', 'absolute wind direction'
        :param use_ws_ahead 主要服务于PMAPS 2020 paper
        :return: UnivariatePDFLike组成的tuple
        """
        if use_ws_ahead == 0:
            path_ = self.results_path + '3d_cvine_gmcm_model/' + self.__str__() + '/'
        elif use_ws_ahead == 1:
            path_ = self.results_path + '3d_cvine_gmcm_model_use_ws_ahead_1/' + self.__str__() + '/'
        else:
            raise
        return self.__estimate_active_power_output_by_copula_model(input_ndarray, path_, 3)

    def estimate_active_power_output_by_4d_cvine_gmcm_model(self, input_ndarray: ndarray) -> tuple:
        """
        运用4维的vine模型去估计有功输出。运用这个函数前必须让fit_4d_vine_gmcm_model完整地跑一遍。
        :param input_ndarray 3维。维度的名称依次是：'wind speed', 'absolute wind direction', 'environmental temperature'
        :return: UnivariatePDFLike组成的tuple
        """
        path_ = self.results_path + '4d_cvine_gmcm_model/' + self.__str__() + '/'
        return self.__estimate_active_power_output_by_copula_model(input_ndarray, path_, 4)

    def plot_wind_speed_to_active_power_output_mob(self, show_category_as_in_outlier: Union[Tuple[int, ...], str]):
        show_category_mask = CategoryUnivariate(self.outlier_category).cal_tuple_category_mask(
            show_category_as_in_outlier)
        month = np.array([x.month for x in datetime64_ndarray_to_datetime_tuple(self.measurements['time'].values)])

        # Summer
        mask_summer = np.bitwise_and(show_category_mask, np.bitwise_and(month >= 4, month <= 9))
        mob = MethodOfBins(self.measurements['wind speed'].values,
                           self.measurements['active power output'].values,
                           bin_step=0.5, first_bin_left_boundary=0, last_bin_left_boundary=28.5,
                           considered_data_mask_for_mob_calculation=mask_summer)
        summer_mob_statistic = mob.cal_mob_statistic_eg_quantile()
        ax = mob.plot_mob_statistic(x_label='Wind speed (m/s)', y_label='Active power output (kW)',
                                    x_lim=(0, 29), y_lim=(-3000 * 0.0125, 3000 * 1.0125),
                                    scatter_color='g',
                                    series_linestyle='-', series_color='r', label='Summer', title=self.name)

        # Winter
        mask_winter = np.bitwise_and(show_category_mask, ~np.bitwise_and(month >= 4, month <= 9))
        mob = MethodOfBins(self.measurements['wind speed'].values,
                           self.measurements['active power output'].values,
                           bin_step=0.5, first_bin_left_boundary=0, last_bin_left_boundary=28.5,
                           considered_data_mask_for_mob_calculation=mask_winter)
        winter_mob_statistic = mob.cal_mob_statistic_eg_quantile()
        mob.plot_mob_statistic(ax=ax, x_label='Wind speed (m/s)', y_label='Active power output (kW)',
                               x_lim=(0, 29), y_lim=(-3000 * 0.0125, 3000 * 1.0125),
                               scatter_color='b',
                               series_linestyle='--', series_color='k', label='Winter', title=self.name,
                               save_file_=self.results_path + self.name + '1')
        max_diff_idx = np.nanargmax(np.abs(summer_mob_statistic[:, 1] - winter_mob_statistic[:, 1]))
        print('{} summer and winter maximum active power output difference is at {} m/s '
              'and the value is {:.2f} kW'.format(self.name,
                                                  summer_mob_statistic[max_diff_idx, 0],
                                                  abs(summer_mob_statistic[max_diff_idx, 1] - winter_mob_statistic[
                                                      max_diff_idx, 1])))

        # Summer_highest
        hottest_in_summer = self.measurements['environmental temperature'].values > np.nanpercentile(
            self.measurements['environmental temperature'].values[mask_summer], 95)
        hottest_in_summer = np.bitwise_and(mask_summer, hottest_in_summer)
        mob = MethodOfBins(self.measurements['wind speed'].values,
                           self.measurements['active power output'].values,
                           bin_step=0.5, first_bin_left_boundary=0, last_bin_left_boundary=28.5,
                           considered_data_mask_for_mob_calculation=hottest_in_summer)
        hottest_in_summer_mob_statistic = mob.cal_mob_statistic_eg_quantile()
        ax = mob.plot_mob_statistic(x_label='Wind speed (m/s)', y_label='Active power output (kW)',
                                    x_lim=(0, 29), y_lim=(-3000 * 0.0125, 3000 * 1.0125),
                                    scatter_color='g',
                                    series_linestyle='-', series_color='r', label='Hottest in summer', title=self.name)

        # Winter_lowest
        coldest_in_winter = self.measurements['environmental temperature'].values < np.nanpercentile(
            self.measurements['environmental temperature'].values[mask_winter], 5)
        coldest_in_winter = np.bitwise_and(mask_winter, coldest_in_winter)
        mob = MethodOfBins(self.measurements['wind speed'].values,
                           self.measurements['active power output'].values,
                           bin_step=0.5, first_bin_left_boundary=0, last_bin_left_boundary=28.5,
                           considered_data_mask_for_mob_calculation=coldest_in_winter)
        coldest_in_winter_mob_statistic = mob.cal_mob_statistic_eg_quantile()
        mob.plot_mob_statistic(ax=ax, x_label='Wind speed (m/s)', y_label='Active power output (kW)',
                               x_lim=(0, 29), y_lim=(-3000 * 0.0125, 3000 * 1.0125),
                               scatter_color='b',
                               series_linestyle='--', series_color='k', label='Coldest in winter', title=self.name,
                               save_file_=self.results_path + self.name + '2')
        max_diff_idx = np.nanargmax(
            np.abs(hottest_in_summer_mob_statistic[:, 1] - coldest_in_winter_mob_statistic[:, 1]))
        print('{} hottest summer and coldest winter maximum active power output difference is at {} '
              'm/s and the value is {:.2f} kW'.format(self.name,
                                                      hottest_in_summer_mob_statistic[max_diff_idx, 0],
                                                      abs(hottest_in_summer_mob_statistic[max_diff_idx, 1] -
                                                          coldest_in_winter_mob_statistic[max_diff_idx, 1])))

    def identify_outlier(self):
        try_to_find_folder_path_otherwise_make_one(self.results_path + 'Filtering/' + self.__str__() + '/')

        # 先对每一个维度进行outlier分析
        @load_exist_pkl_file_otherwise_run_and_save(
            self.results_path + 'Filtering/' + self.__str__() + '/outlier_category_detailed.pkl')
        def load_or_make_outlier_category_detailed():
            self.outlier_category_detailed = pd.DataFrame(np.full(self.measurements.shape, 0, dtype=int),
                                                          columns=self.measurements.columns)
            # wind speed outlier
            self.outlier_category_detailed.loc[
                self.__identify_missing_data_outlier('wind speed'), 'wind speed'] = -1
            self.outlier_category_detailed.loc[
                self.__identify_out_of_range_outlier('wind speed', 0, 50), 'wind speed'] = 1
            self.outlier_category_detailed.loc[
                self.__identify_linear_series_outlier('wind speed'), 'wind speed'] = 2

            # active power output outlier
            self.outlier_category_detailed.loc[
                self.__identify_missing_data_outlier('active power output'), 'active power output'] = -1
            self.outlier_category_detailed.loc[self.__identify_shut_down_outlier(), 'active power output'] = 1
            self.outlier_category_detailed.loc[self.__identify_change_point_outlier('active power output'),
                                               'active power output'] = 2
            self.outlier_category_detailed.loc[self.__identify_curtailment_outlier(), 'active power output'] = 3
            self.outlier_category_detailed.loc[self.__identify_interquartile_outlier(), 'active power output'] = 5

            # absolute wind direction outlier
            self.outlier_category_detailed.loc[
                self.__identify_missing_data_outlier('absolute wind direction'), 'absolute wind direction'] = -1
            self.outlier_category_detailed.loc[
                self.__identify_out_of_range_outlier('absolute wind direction', 0, 360), 'absolute wind direction'] = 1
            self.outlier_category_detailed.loc[
                self.__identify_linear_series_outlier('absolute wind direction'), 'absolute wind direction'] = 2

            # environmental temperature outlier
            self.outlier_category_detailed.loc[
                self.__identify_missing_data_outlier('environmental temperature'), 'environmental temperature'] = -1

            return self.outlier_category_detailed

        # 因为在这个project中，active power output是中心变量，所以单独对它与wind speed组成的二维序列进行outlier分析
        @load_exist_npy_file_otherwise_run_and_save(
            self.results_path + 'Filtering/' + self.__str__() + '/outlier_category.npy')
        def load_or_make():
            self.outlier_category = self.outlier_category_detailed['active power output'].values
            self.outlier_category[self.outlier_category_detailed['wind speed'].values == 2] = 4
            return self.outlier_category

        self.outlier_category_detailed = load_or_make_outlier_category_detailed()
        self.outlier_category = load_or_make()

    def update_outlier_category(self, old_category: int, new_category: int):
        CategoryUnivariate(self.outlier_category).update_category(old_category, new_category)
        save_npy_file(self.results_path + 'Filtering/' + self.__str__() + '/outlier_category.npy',
                      self.outlier_category)

    def add_new_outlier_category(self, new_outlier_category_mask: ndarray, new_outlier_assigned_number: int):
        """
        可能以后的分析中需要新添加outlier type，以满足具体的分析需要
        """
        self.outlier_category[new_outlier_category_mask] = new_outlier_assigned_number
        save_npy_file(self.results_path / 'Filtering/' / self.__str__() / '/outlier_category.npy',
                      self.outlier_category)


class WF(WTandWFBase):
    def __init__(self, *args, rated_active_power_output: Union[int, float], **kwargs):
        super().__init__(*args, rated_active_power_output=rated_active_power_output, **kwargs)

    @classmethod
    def init_from_wind_turbine_instances(cls, wind_turbine_instances: Iterable[WT], *, obj_name: str):
        """
        To initialise a WF instance from a group of WT instances.
        Can only work on averaging WS and Pout.
        :return:
        """
        wind_farm_df = pd.DataFrame()
        rated_active_power_output = []
        for i, this_wind_turbine in enumerate(wind_turbine_instances):
            rated_active_power_output.append(this_wind_turbine.rated_active_power_output)
            wind_farm_df = pd.merge(wind_farm_df, this_wind_turbine.pd_view()[['wind speed', 'active power output']],
                                    how='outer', left_index=True, right_index=True,
                                    suffixes=(f'_WT{i}', f'_WT{i + 1}'))
        # Adjust to multi index, so the indexing will be easy
        new_columns = pd.MultiIndex.from_arrays([[re.findall(r'.*(?=_)', x)[0] for x in wind_farm_df.columns],
                                                 [re.findall(r'(?<=_).*', x)[0] for x in wind_farm_df.columns]],
                                                names=('Physical Quantity', 'WT No.'))
        wind_farm_df.columns = new_columns
        # Averaging
        # Note that the treatment for WS and Pout are different
        # For Pout, equivalent, the non-missing values are summed up and divided by the number of WTs in the WF
        # For WS, equivalent, the non-missing values are averaged directly
        wind_farm_df['active power output'] = wind_farm_df['active power output'].fillna(value=0)
        wind_farm_df['active power output'] *= wind_farm_df['active power output'].shape[1]
        wind_farm_instance = cls(wind_farm_df.mean(1, level='Physical Quantity', skipna=True),
                                 rated_active_power_output=sum(rated_active_power_output),
                                 obj_name=obj_name,
                                 predictor_names=('wind speed',),
                                 dependant_names=('active power output',))
        return wind_farm_instance

    def power_curve_by_method_of_bins(self, cal_region_boundary: bool = False) -> PowerCurveByMethodOfBins:
        return PowerCurveByMethodOfBins(wind_speed_recording=self['wind speed'].values,
                                        active_power_output_recording=self['active power output'].values,
                                        cal_region_boundary=cal_region_boundary)

    # def __str__(self):
    #     t1 = np_datetime64_to_datetime(self.index.values[0]).strftime('%Y-%m-%d %H.%M')
    #     t2 = np_datetime64_to_datetime(self.index.values[-1]).strftime('%Y-%m-%d %H.%M')
    #     current_season = self.get_current_season(season_template=SeasonTemplate1)
    #     if current_season == 'all seasons':
    #         return "{} WF from {} to {}".format(self.name, t1, t2)
    #     else:
    #         return "{} WF from {} to {} {}".format(self.name, t1, t2, current_season)

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
        try_to_find_folder_path_otherwise_make_one((path_, path_ + 'a/', path_ + 'b/'))
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
        try_to_find_folder_path_otherwise_make_one(path_)

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


if __name__ == '__main__':
    tt = WF(
        np.arange(120).reshape((8, 15)),
        obj_name='tt_name',
        predictor_names=('tt_predictor_names',),
        dependant_names=('tt_dependant_names',),
        rated_active_power_output=3000
    )
    # print(tt)
    cc = tt[0]
