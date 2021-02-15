from __future__ import annotations

import numpy as np
import pandas as pd
from Time_Processing.format_convert_Func import datetime64_ndarray_to_datetime_tuple, np_datetime64_to_datetime
from File_Management.load_save_Func import *
from BivariateAnalysis_Class import BivariateOutlier, Bivariate, MethodOfBins
from File_Management.path_and_file_management_Func import try_to_find_folder_path_otherwise_make_one, try_to_find_file
from UnivariateAnalysis_Class import CategoryUnivariate, UnivariatePDFOrCDFLike, UnivariateGaussianMixtureModel, \
    DeterministicUnivariateProbabilisticModel
from typing import Union, Tuple, List, Iterable, Sequence
from BivariateAnalysis_Class import Bivariate, MethodOfBins
from Ploting.fast_plot_Func import *
from PowerCurve_Class import *
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
from ConvenientDataType import UncertaintyDataFrame, StrOneDimensionNdarray
from tqdm import tqdm
from parse import parse
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from HighDimensionalAnalysis_Class import OneDimensionBinnedData


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
             mfr_kwargs: Sequence[dict] = None,
             mfr_mode: str = 'continuous',
             plot_scatter_pc: bool = False,
             save_to_buffer: bool = False,
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
            for i, this_mfr_pc in enumerate(plot_mfr):
                if mfr_kwargs is not None:
                    this_mfr_kwargs = mfr_kwargs[i]
                else:
                    this_mfr_kwargs = {}
                ax = this_mfr_pc.plot(ax=ax, mode=mfr_mode, **this_mfr_kwargs)
        if plot_scatter_pc:
            ax = PowerCurveByMethodOfBins(self['wind speed'].values,
                                          self['active power output'].values / self.rated_active_power_output).plot(
                ws=np.arange(0, 50, 0.5),
                ax=ax,
                plot_recording=False,
                save_to_buffer=save_to_buffer,
                **dict(ChainMap(kwargs, WS_POUT_2D_PLOT_KWARGS))
            )
        return ax

    def twin_time_series_plot(self, *, time_window_mask: Sequence[bool] = slice(None),
                              x_axis_format: str = '%H',
                              x_label: str = 'Time of a Day [Hour]',
                              wind_speed_y_lim=(-0.05, 27.55),
                              power_output_y_lim=WS_POUT_2D_PLOT_KWARGS['y_lim']):

        time_x = self.index[time_window_mask]
        ax = series(x=time_x, y=self.loc[time_window_mask, 'wind speed'].values, figure_size=(5, 3.3 * 0.618),
                    x_axis_format=x_axis_format, x_label=x_label,
                    marker='*', markersize=6, color='royalblue', linestyle='-',
                    y_lim=wind_speed_y_lim, y_label='Wind Speed [m/s]', label='Wind speed')

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Active Power Output [p.u.]', fontdict={'size': 10})  # we already handled the x-label with ax1
        series(x=time_x, y=self.loc[time_window_mask, 'active power output'].values, ax=ax2,
               x_axis_format=x_axis_format, marker='o', markersize=3, color='green', linestyle='--',
               y_lim=power_output_y_lim, label='Power output')
        plt.grid(False)
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.get_legend().remove()
        ax2.legend(lines + lines2, labels + labels2, loc=0, prop={'size': 10})
        return ax, ax2

    def update_air_density_to_last_column(self):
        if 'air density' in self.columns:
            return self['air density'].values
        else:
            from Wind_Class import cal_air_density, celsius_to_kelvin
            air_density = cal_air_density(celsius_to_kelvin(self['environmental temperature'].values),
                                          self['relative humidity'].values / 100,
                                          self['barometric pressure'].values * 100)

            self.insert(self.shape[1], column='air density', value=air_density)
            return air_density


class WT(WTandWFBase):

    def __init__(self, *args, rated_active_power_output=3000, **kwargs):
        super().__init__(*args, rated_active_power_output=rated_active_power_output, **kwargs)

    @property
    def default_results_saving_path(self):
        saving_path = {
            "outlier": self.results_path / f"Filtering/{self.__str__()}/results.pkl",
            "power curve": self.results_path / f"PowerCurve/{self.__str__()}/results.pkl"
        }
        for x in saving_path.values():
            try_to_find_folder_path_otherwise_make_one(x.parent)
        return saving_path

    @property
    def outlier_name_mapper(self) -> DataCategoryNameMapper:
        meta = [["missing data", "missing", -1, "N/A"],
                ["Normal data", "normal", 0, "the recordings that can be captured by the simulation"],
                ["Low maximum Pout", "CAT-I", 1, "curtailment"],
                ["Linear Pout-WS", "CAT-II", 2, "e.g., constant WS-variable Pout"],
                ["Low Pout-high WS", "CAT-III", 3, "Low Pout-high WS caused by the other sources"],
                ["Scattered", "CAT-IV", 4, "the recordings rejected by the simulation"]]

        mapper = DataCategoryNameMapper.init_from_template(rows=len(meta))
        mapper[:] = meta
        return mapper

    def outlier_detector(self, how_to_detect_scattered: str = 'sim', *,
                         save_file_path: Path = None,
                         prior_sim_knowledge_path: Path = None) -> DataCategoryData:
        assert (how_to_detect_scattered in ('isolation forest', 'hist', 'sim')), "Check 'how_to_detect_scattered'"
        save_file_path = save_file_path or self.default_results_saving_path["outlier"]
        if try_to_find_file(save_file_path):
            warnings.warn(f"{self.__str__()} has results in {save_file_path}, so return to the existing file")
            return load_pkl_file(save_file_path)['DataCategoryData obj']
        assert (prior_sim_knowledge_path is not None)
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
        # cat_iii_outlier_mask = self.data_category_is_linearity(
        #     '60T',
        #     general_linearity_error={'wind speed': 0.001,
        #                              'active power output': self.rated_active_power_output * 0.0005}
        # )

        outlier.abbreviation[cat_iii_outlier_mask] = "CAT-III"
        del cat_iii_outlier_mask
        # %% CAT-IV
        from Wind_Class import Wind
        sigma_func = Wind.learn_transition_by_looking_at_actual_high_resol()
        # Get a Callable that can calculates the 10s to 10s wind speed variation sigma
        if how_to_detect_scattered == 'hist':
            current_others_index = np.argwhere(outlier('others')).flatten()
            cat_iv_outlier_mask = BivariateOutlier(
                predictor_var=self.iloc[current_others_index]['wind speed'].values,
                dependent_var=self.iloc[current_others_index]['active power output'].values,
                bin_step=0.5
            ).identify_interquartile_outliers_based_on_method_of_bins()
            outlier.abbreviation[current_others_index[cat_iv_outlier_mask]] = "CAT-IV"
        elif how_to_detect_scattered == 'isolation forest':
            current_others_index = np.argwhere(outlier('others')).flatten()
            current_normal_data = StandardScaler().fit_transform(self.concerned_data().iloc[current_others_index])
            cat_iv_outlier_mask = use_isolation_forest(current_normal_data)
            outlier.abbreviation[current_others_index[cat_iv_outlier_mask]] = "CAT-IV"
            del current_normal_data
        # The follows is the key contribution, which is the implementation of the proposed simulation
        else:
            # %% The region inside mfr PC range must be normal, which is due to air density variation!
            low_mfr_pc = PowerCurveByMfr.init_all_instances_in_docs()[0]
            high_mfr_pc = PowerCurveByMfr.init_all_instances_in_docs()[-1]
            outlier.abbreviation[np.all((
                (
                    outlier(['CAT-I', 'others']),
                    self['active power output'] >= low_mfr_pc(self['wind speed']) * self.rated_active_power_output,
                    self['active power output'] <= high_mfr_pc(self['wind speed']) * self.rated_active_power_output
                )
            ), axis=0)] = "normal"

            to_be_simulated_mask = outlier(['CAT-I', 'others'])
            try_to_find_folder_path_otherwise_make_one(prior_sim_knowledge_path.parent)
            prior_sim_knowledge = load_pkl_file(prior_sim_knowledge_path)  # type:Union[pd.DataFrame, None]
            level_0_name = "wind speed"
            level_1_name = "wind speed std."
            level_2_name = "air density"
            # Basic information needed for prior_sim_knowledge and the simulation
            ws_std_resol = 0.1
            ws_binned_data_obj = OneDimensionBinnedData(self[level_0_name][to_be_simulated_mask].values,
                                                        bin_step=0.5, first_bin_left_boundary=0)
            # Note, ws_std_resol in the actual recording is only 0.1! So, if first_bin_left_boundary here is 0,
            # and medium point of WS std. bin is used to simulate the wind, the results will always overestimate variety
            ws_std_binned_data_obj = OneDimensionBinnedData(self[level_1_name][to_be_simulated_mask].values,
                                                            bin_step=ws_std_resol,
                                                            first_bin_left_boundary=-ws_std_resol / 2)
            mfr_pc_densities = PowerCurveByMfr.air_density_in_docs()
            uncertainty_data_frame_template_obj = UncertaintyDataFrame.init_from_template(1)
            prior_sim_knowledge_columns = uncertainty_data_frame_template_obj.index

            # prior_sim_knowledge should be initialised if not existing
            if prior_sim_knowledge is None:
                prior_sim_knowledge = pd.DataFrame(
                    columns=prior_sim_knowledge_columns,
                    index=pd.MultiIndex.from_tuples(
                        ((str(ws_binned_data_obj.bin[0]), str(ws_std_binned_data_obj.bin[0]), mfr_pc_densities[0]),),
                        names=(level_0_name, level_1_name, level_2_name)
                    ),
                    dtype=float
                )

            # Iterate over the recordings that are to be checked (to_be_simulated_mask)
            mfr_pc_obj = PowerCurveByMfr(mfr_pc_densities[0])
            any_update_flag = False
            self.update_air_density_to_last_column()
            for i, this_recording in tqdm(enumerate(self[to_be_simulated_mask].iterrows())):
                this_recording_index = this_recording[0]
                this_recording_ws = this_recording[1]['wind speed']
                this_recording_ws_std = this_recording[1]['wind speed std.']
                this_recording_air_density = this_recording[1]['air density']
                this_recording_pout = this_recording[1]['active power output'] / self.rated_active_power_output

                # %% ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ DEBUG ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                # if not ((this_recording_ws > 25) and (this_recording_pout > 0.2)):
                #     continue
                # %% ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ DEBUG ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                this_recording_ws_bin = ws_binned_data_obj(this_recording_ws)
                this_recording_ws_std_bin = ws_std_binned_data_obj(this_recording_ws_std)
                this_multi_index_obj = pd.MultiIndex.from_tuples(
                    ((str(this_recording_ws_bin), str(this_recording_ws_std_bin), mfr_pc_densities[0]),),
                    names=(level_0_name, level_1_name, level_2_name)
                )
                not_existing_flag = np.sum(this_multi_index_obj.isin(prior_sim_knowledge.index)) == 0
                existing_but_nan_flag = False
                if np.sum(this_multi_index_obj.isin(prior_sim_knowledge.index)) != 0:
                    existing_but_nan_flag = np.any(
                        np.isnan(prior_sim_knowledge.loc[this_multi_index_obj[0]].values)
                    )
                # If it does not have index for current WS and WS std.
                if not_existing_flag or existing_but_nan_flag:
                    any_update_flag = True
                    # Firstly, create current_sim_knowledge based on current key
                    current_sim_knowledge = pd.DataFrame(
                        columns=prior_sim_knowledge_columns,
                        index=this_multi_index_obj,
                        dtype=float
                    )
                    # Secondly, simulate (using mfr_pc_densities[0])
                    # prepare high resolution wind, using medium point of WS bin and WS std. bin
                    wind = Wind(this_recording_ws_bin[1], this_recording_ws_std_bin[1])
                    high_resol_wind = wind.simulate_transient_wind_speed_time_series(
                        resolution=10,
                        traces_number_for_each_recording=1_000_000,
                        sigma_func=sigma_func
                    )
                    # prepare mfr pc
                    this_pout_uncertainty = mfr_pc_obj.cal_with_hysteresis_control_using_high_resol_wind(
                        high_resol_wind,
                        return_percentiles=uncertainty_data_frame_template_obj
                    )
                    # Finally, update the value,
                    current_sim_knowledge.iloc[0] = this_pout_uncertainty.values.flatten()
                    if not_existing_flag:
                        prior_sim_knowledge = pd.concat((prior_sim_knowledge, current_sim_knowledge))
                    else:
                        prior_sim_knowledge.loc[this_multi_index_obj[0]] = this_pout_uncertainty.values.flatten()

                    # For every 1000 updates, also save and update prior_sim_knowledge in the disk
                    if i % 50 == 0:
                        prior_sim_knowledge = prior_sim_knowledge.sort_index()
                        save_pkl_file(prior_sim_knowledge_path, prior_sim_knowledge)

                # Check the table again, which now should have the value for current WS and WS std.
                this_recording_sim = prior_sim_knowledge.loc[this_multi_index_obj]
                this_recording_sim = UncertaintyDataFrame(this_recording_sim.values.T,
                                                          columns=(0,),
                                                          index=this_recording_sim.columns)
                # based_pout_sim_low, based_pout_sim_high = this_recording_sim(by_sigma=1.5).values.flatten()
                # based_pout_sim_low, based_pout_sim_high = this_recording_sim(
                #     preserved_data_percentage=95).values.flatten()
                based_pout_sim_low, based_pout_sim_high = this_recording_sim(by_sigma=3).values.flatten()

                # Mapping the value using the relationship among Mfr_PC_rho_x
                new_power_output = PowerCurveByMfr.map_given_power_output_to_another_air_density(
                    old_air_density=np.array([mfr_pc_densities[0]] * 2),
                    new_air_density=np.array([this_recording_air_density] * 2),
                    old_power_output=np.array([based_pout_sim_low, based_pout_sim_high]),
                    wind_speed=np.array([this_recording_ws] * 2),
                )
                based_pout_sim_low, based_pout_sim_high = new_power_output

                if (this_recording_pout >= based_pout_sim_low) and (this_recording_pout <= based_pout_sim_high):
                    outlier.abbreviation[outlier.index == this_recording_index] = 'normal'
                else:
                    if outlier.abbreviation[outlier.index == this_recording_index][0] == 'others':
                        outlier.abbreviation[outlier.index == this_recording_index] = 'CAT-IV'

            # After the iteration, if any updates happen, should save and update prior_sim_knowledge in the disk
            if any_update_flag:
                def prior_sim_knowledge_index_sort_key(this_index):
                    parse_obj_level_0 = parse(r"[{} {} {}]", this_index[0])
                    parse_obj_level_1 = parse(r"[{} {} {}]", this_index[1])
                    level_0_val = float(parse_obj_level_0[0])
                    level_1_val = float(parse_obj_level_1[0])
                    return level_0_val, level_1_val

                sorted_index = sorted(prior_sim_knowledge.index, key=prior_sim_knowledge_index_sort_key)
                prior_sim_knowledge = prior_sim_knowledge.reindex(sorted_index)

                save_pkl_file(prior_sim_knowledge_path, prior_sim_knowledge)

            # Use it to determine whether this_recording is a CAT-IV outlier or not.

        """"""
        # outlier.abbreviation[current_others_index[cat_iv_outlier_mask]] = "CAT-IV"
        # del current_others_index, cat_iv_outlier_mask
        # # %% CAT-IV.a and CAT-IV.b
        # outlier.abbreviation[np.all((
        #     (
        #         outlier("CAT-IV"),
        #         self['active power output'] >= PowerCurveByMfr.init_all_instances_in_docs()[0](self['wind speed']),
        #         self['active power output'] <= PowerCurveByMfr.init_all_instances_in_docs()[-1](self['wind speed'])
        #     )
        # ), axis=0)] = "CAT-IV.a"
        # # draft analysis for separating CAT-IV.a and CAT-IV.b
        #
        # outlier.abbreviation[np.bitwise_and(outlier("CAT-IV"),
        #                                     np.isnan(self['wind speed std.']))] = "CAT-IV.b"
        # # Make sure there are air density recordings
        # self.update_air_density_to_last_column()
        # current_cat_iv_data = self[['wind speed',
        #                             'wind speed std.',
        #                             'active power output',
        #                             'air density']][outlier('CAT-IV')]
        # current_cat_iv_data['Mfr PC obj'] = PowerCurveByMfr.init_multiple_instances(
        #     air_density=current_cat_iv_data['air density'].values)
        # # To find the recordings with same wind speed, wind speed std., and air density. This will increase the
        # # computation efficiency in further simulation
        # unique_rows, unique_label = current_cat_iv_data.unique(['wind speed', 'wind speed std.', 'Mfr PC obj'])
        # # Use the buffer
        # buffer = try_to_find_file(save_file_path.parent / 'temp.pkl')
        # if buffer:
        #     buffer = load_pkl_file(save_file_path.parent / 'temp.pkl')
        #     cat_iv_a_outlier_index = buffer['cat_iv_a_outlier_index']
        #     cat_iv_b_outlier_index = buffer['cat_iv_b_outlier_index']
        #     pout_uncertainty_list = buffer['pout_uncertainty_list']
        # else:
        #     cat_iv_a_outlier_index, cat_iv_b_outlier_index, pout_uncertainty_list = [], [], []
        # # Do the simulation (ONLY in unique_rows)
        # for i in tqdm(range(unique_rows.shape[0])):
        #     # Use the buffer
        #     if buffer:
        #         if i <= buffer['i']:
        #             continue
        #
        #     this_unique_row = unique_rows.iloc[i]
        #     # prepare high resolution wind
        #     wind = Wind(this_unique_row['wind speed'], this_unique_row['wind speed std.'])
        #     high_resol_wind = wind.simulate_transient_wind_speed_time_series(
        #         resolution=10,
        #         traces_number_for_each_recording=1_000_000,
        #         sigma_func=sigma_func
        #     )
        #     # prepare mfr pc
        #     this_unique_mfr_pc = this_unique_row['Mfr PC obj']
        #     this_pout_uncertainty = this_unique_mfr_pc.cal_with_hysteresis_control_using_high_resol_wind(
        #         high_resol_wind,
        #         return_percentiles=UncertaintyDataFrame.init_from_template(
        #             columns_number=len(high_resol_wind),
        #             percentiles=None),
        #     )
        #     pout_uncertainty_list.append(this_pout_uncertainty)
        #     # Compare to judge. Note that this is a kind of inverse of unique
        #     pout_actual = current_cat_iv_data[unique_label == i]
        #     for j in range(pout_actual.shape[0]):
        #         this_pout_actual = pout_actual['active power output'].iloc[j] / self.rated_active_power_output
        #         if all((this_pout_actual >= pout_uncertainty_list[i].loc['1.5_Sigma_low', 0],
        #                 this_pout_actual <= pout_uncertainty_list[i].loc['1.5_Sigma_high', 0])):
        #             cat_iv_a_outlier_index.append(current_cat_iv_data[unique_label == i].index[j])
        #         elif all((this_pout_actual >= pout_uncertainty_list[i].loc['1.5_Sigma_low', 0],
        #                   this_pout_actual <= float(this_unique_mfr_pc(this_unique_row['wind speed'])))):
        #             cat_iv_a_outlier_index.append(current_cat_iv_data[unique_label == i].index[j])
        #         else:
        #             cat_iv_b_outlier_index.append(current_cat_iv_data[unique_label == i].index[j])
        #     # Save progress
        #     if (i % 25) == 0:
        #         save_pkl_file(save_file_path.parent / 'temp.pkl',
        #                       {"i": i,
        #                        "outlier": outlier,
        #                        "cat_iv_a_outlier_index": cat_iv_a_outlier_index,
        #                        "cat_iv_b_outlier_index": cat_iv_b_outlier_index,
        #                        "pout_uncertainty_list": pout_uncertainty_list})
        # outlier.abbreviation[self.index.isin(cat_iv_a_outlier_index)] = "CAT-IV.a"
        # outlier.abbreviation[self.index.isin(cat_iv_b_outlier_index)] = "CAT-IV.b"

        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        save_pkl_file(save_file_path,
                      {'raw_ndarray_data': np.array(outlier.abbreviation),
                       'raw_index': outlier.index,
                       'DataCategoryData obj': outlier})
        return outlier

    def outlier_plot(self, outlier: DataCategoryData = None, ax=None, *, plot_individual: bool = False, **kwargs):
        outlier = outlier or load_pkl_file(self.default_results_saving_path["outlier"])['DataCategoryData obj']
        self.loc[:, "active power output"] /= self.rated_active_power_output

        # if sum(outlier("CAT-I.a")) > 0:
        #     ax = scatter(*self[outlier("CAT-I.a")][["wind speed", "active power output"]].values.T, label="CAT-I.a",
        #                  ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
        #                  color="darkorange", marker="1", s=24, zorder=8, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-I")) > 0:
            ax = scatter(*self[outlier("CAT-I")][["wind speed", "active power output"]].values.T, label="CAT-I",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="black", marker="x", s=16, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-II")) > 0:
            ax = scatter(*self[outlier("CAT-II")][["wind speed", "active power output"]].values.T, label="CAT-II",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="red", marker="|", s=28, zorder=11, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-III")) > 0:
            ax = scatter(*self[outlier("CAT-III")][["wind speed", "active power output"]].values.T, label="CAT-III",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="darkorange", marker="v", s=14, **WS_POUT_2D_PLOT_KWARGS)
        # if sum(outlier("CAT-IV.a")) > 0:
        #     ax = scatter(*self[outlier("CAT-IV.a")][["wind speed", "active power output"]].values.T, label="CAT-IV.a",
        #                  ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
        #                  color="green", marker="3", s=24, zorder=9, **WS_POUT_2D_PLOT_KWARGS)
        if sum(outlier("CAT-IV")) > 0:
            ax = scatter(*self[outlier("CAT-IV")][["wind speed", "active power output"]].values.T, label="CAT-IV",
                         ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                         color="green", marker="3", s=24, zorder=-9, **WS_POUT_2D_PLOT_KWARGS)
        # ax = scatter(*self[outlier("others")][["wind speed", "active power output"]].values.T, label="Others",
        #              ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
        #              color="royalblue", zorder=10, **WS_POUT_2D_PLOT_KWARGS, **kwargs)
        ax = scatter(*self[outlier("normal")][["wind speed", "active power output"]].values.T, label="Normal",
                     ax=ax if not plot_individual else None, alpha=WS_POUT_SCATTER_ALPHA,
                     color="royalblue", zorder=10, **WS_POUT_2D_PLOT_KWARGS, **kwargs)
        self.loc[:, "active power output"] *= self.rated_active_power_output

        return ax

    def outlier_report(self, outlier: DataCategoryData = None, *, save_to_buffer=False):
        outlier = outlier or load_pkl_file(self.default_results_saving_path["outlier"])['DataCategoryData obj']
        return outlier.report(self.default_results_saving_path["outlier"].parent / "report.csv",
                              save_to_buffer=save_to_buffer)

    def select_data_and_get_power_curve_model(self, task: str, **kwargs) -> PowerCurveFittedBy8PLF:
        assert (task in ('load', 'fit')), "'Task' is not in ('load', 'fit')"
        pc_file_path = self.default_results_saving_path["power curve"]
        outlier = load_pkl_file(self.default_results_saving_path["outlier"])['DataCategoryData obj']
        # selected_data_mask = outlier(("others", "CAT-I.a", "CAT-IV.a"))
        selected_data_mask = outlier("normal")

        pc_obj = PowerCurveFittedBy8PLF(
            wind_speed_recording=self['wind speed'].values[selected_data_mask],
            active_power_output_recording=self['active power output'].values[
                                              selected_data_mask] / self.rated_active_power_output,
            **kwargs
        )

        current_results = load_pkl_file(pc_file_path)
        if task == 'fit':
            if current_results is not None:
                current_best = current_results[-1]['variable']
                pc_obj.update_params(*current_best[:pc_obj.params.__len__()])  # The last the best
                params_init_scheme = 'self'
            else:
                params_init_scheme = 'average'
            pc_obj.fit(ga_algorithm_param={'max_num_iteration': 2500,
                                           'max_iteration_without_improv': 1000,
                                           'population_size': 100},
                       params_init_scheme=params_init_scheme,
                       run_n_times=100,
                       save_to_file_path=pc_file_path,
                       focal_error=0.001,
                       wind_speed=np.arange(0, 28.5, 0.1),
                       function_timeout=6000)
        else:
            current_best = current_results[-1]['variable']
            pc_obj.update_params(*current_best[:pc_obj.params.__len__()])

        return pc_obj

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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

        data_preprocessing_params = load_pkl_file(this_path_ + 'data_preprocessing_params.pkl')
        # 对于区域内的有功功率，进行truncated→linear转换；
        active_power_output_linear = TruncatedToLinear(
            data_preprocessing_params['min_active_power_output'],
            data_preprocessing_params['max_active_power_output']).inverse_transform(active_power_output_linear)
        return active_power_output_linear

    @staticmethod
    def __transform_data_to_linear_for_copula_model(data_to_be_transformed: ndarray, path_, dims: int) -> dict:
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        return self.__transform_data_to_linear_for_copula_model(data_to_be_transformed, path_, 2)

    def __transform_data_to_linear_for_3d_vine_gmcm_model(self, data_to_be_transformed: ndarray, path_) -> dict:
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        return self.__transform_data_to_linear_for_copula_model(data_to_be_transformed, path_, 3)

    def __transform_data_to_linear_for_4d_vine_gmcm_model(self, data_to_be_transformed: ndarray, path_) -> dict:
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        return self.__transform_data_to_linear_for_copula_model(data_to_be_transformed, path_, 4)

    def __prepare_fitting_data_for_vine_gmcm_model(self, path_, dims) -> dict:
        """
        准备vine_gmcm的fitting 数据
        model_a_global_mask和model_b_global_mask代表两个区域/完全不同的两个模型
        """
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        return self.__prepare_fitting_data_for_vine_gmcm_model(path_, 4)

    def __prepare_fitting_data_for_3d_vine_gmcm_model(self, path_):
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        return self.__prepare_fitting_data_for_vine_gmcm_model(path_, 3)

    def fit_4d_cvine_gmcm_model(self):
        """
        对于4维的vine模型，只考虑每个pair copula对应的两个变量对应的outlier_category_detailed是0的情况。
        维度的名字依次是：'active power output', 'wind speed', 'absolute wind direction', 'environmental temperature'。
        因为有两个区域，所以其实本质上是两个独立的4d_vine_gmcm_model
        """
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        path_ = self.results_path / '4d_cvine_gmcm_model/' / self.__str__() / '/'
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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)
        gmm_args = gmm_args or {}
        _path = kwargs.get('_path') or (self.results_path + '2d_conditional_probability_by_gmm/' + self.__str__() + \
                                        f' bin_step={bin_step}/')

        try_to_find_folder_path_otherwise_make_one(_path)
        mask = np.bitwise_or(self.outlier_category == 0,
                             self.outlier_category == 5)
        bivariate = Bivariate(self.measurements['wind speed'].values[mask],
                              self.measurements['active power output'].values[mask],
                              bin_step=bin_step)

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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

        return PowerCurveByMfr()(wind_speed_ndarray)

    def __add_active_power_output_dim_for_copula_based_estimating_method(self, input_ndarray: ndarray,
                                                                         linspace_number: int) -> ndarray:
        """
        因为estimate的时候其实是条件概率，它们的输入少了第一维度（即：active power output），所以在将数据放入联合概率模型
        之前要补充一个维度
        """
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

        active_power_output_dim = np.linspace(-30, self.rated_active_power_output + 30, linspace_number)
        active_power_output_dim = np.tile(active_power_output_dim, input_ndarray.shape[0])
        input_ndarray = np.repeat(input_ndarray, linspace_number, 0)
        return np.concatenate((active_power_output_dim.reshape(-1, 1), input_ndarray), axis=1)

    @staticmethod
    def __transform_estimating_method_results_to_normalised_pdf_like(
            unnormalised_pdf_like: ndarray, linspace_number: int) -> Tuple[UnivariatePDFOrCDFLike, ...]:
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

        normalised_pdf_like = []
        for i in range(0, unnormalised_pdf_like.shape[0], linspace_number):
            this_normalised_pdf_like = UnivariatePDFOrCDFLike(
                pdf_like_ndarray=unnormalised_pdf_like[i:i + linspace_number, :])
            normalised_pdf_like.append(this_normalised_pdf_like)
        return tuple(normalised_pdf_like)

    def __estimate_active_power_output_by_copula_model(self, input_ndarray: ndarray, path_, dims) -> tuple:
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

        path_ = self.results_path + '2d_gmcm_model/' + self.__str__() + '/'
        return self.__estimate_active_power_output_by_copula_model(input_ndarray, path_, 2)

    def estimate_active_power_output_by_2d_gmcm_model_with_uncertain_inputs(self, input_ndarray: ndarray) -> tuple:
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

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
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

        path_ = self.results_path + '4d_cvine_gmcm_model/' + self.__str__() + '/'
        return self.__estimate_active_power_output_by_copula_model(input_ndarray, path_, 4)

    def identify_outlier(self):
        # TODO Deprecated
        warnings.warn("Deprecated", DeprecationWarning)

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


class WF(WTandWFBase):
    __slots__ = ("cut_in_wind_speed", "cut_out_wind_speed", "rated_active_power_output", "number_of_wind_turbine")

    def __init__(self, *args, rated_active_power_output: Union[int, float],
                 number_of_wind_turbine: int = None, **kwargs):
        super().__init__(*args, rated_active_power_output=rated_active_power_output, **kwargs)
        self.number_of_wind_turbine = number_of_wind_turbine

    @classmethod
    def init_from_wind_turbine_instances(
            cls, wind_turbine_instances: Sequence[WT], *,
            obj_name: str,
            wind_turbine_instances_data_category: Sequence[DataCategoryData] = None
    ) -> Tuple[WF, pd.DataFrame]:
        """
        To initialise a WF instance from a group of WT instances.
        Can only work on averaging WS and Pout.

        Specifically, if 'wind_turbine_instances_data_category' is provided, then only initialise using 'shutdown',
        'curtailed', 'operating' WT recordings, and will also return valid total_curtailment_amount
        :return:
        """
        wind_farm_df = pd.DataFrame()
        rated_active_power_output = []
        total_curtailment_amount = pd.DataFrame()
        for i, this_wind_turbine in enumerate(wind_turbine_instances):
            this_wind_turbine = copy.deepcopy(this_wind_turbine)
            # If WT data category information is available
            if wind_turbine_instances_data_category is not None:
                # Only consider 'shutdown', 'curtailed', 'operating', the rest (i.e., 'nan') are NaN
                this_wind_turbine[~wind_turbine_instances_data_category[i](
                    ('shutdown', 'curtailed', 'operating')
                )] = np.nan
                # Curtailment amount is important information, if available
                total_curtailment_amount = pd.merge(
                    total_curtailment_amount,
                    this_wind_turbine[wind_turbine_instances_data_category[i]('curtailed')][['active power output']],
                    how='outer', left_index=True, right_index=True, suffixes=(f'_WT{i}', f'_WT{i + 1}')
                )
            # 'non-missing' means that both the 'wind speed' and 'active power output' must be simultaneously not NaN,
            # To achieve this, the data will be modified intentionally: i.e., to discard more data
            any_nan_mask = this_wind_turbine[['wind speed', 'active power output']].isna().any(1).values
            this_wind_turbine.loc[any_nan_mask, ['wind speed', 'active power output']] = np.nan

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
        # For Pout, equivalent, the non-missing values are summed up
        # For WS, equivalent, the non-missing values are averaged directly
        # Note 'non-missing' means that both the 'wind speed' and 'active power output' must be simultaneously not NaN
        wind_farm_df = pd.DataFrame(
            {'wind speed': wind_farm_df['wind speed'].mean(1, skipna=True).values,
             'active power output': wind_farm_df['active power output'].sum(1, skipna=True).values},
            index=wind_farm_df.index
        )

        wind_farm_instance = cls(wind_farm_df,
                                 rated_active_power_output=sum(rated_active_power_output),
                                 number_of_wind_turbine=wind_turbine_instances.__len__(),
                                 obj_name=obj_name,
                                 predictor_names=('wind speed',),
                                 dependant_names=('active power output',))
        # Curtailment amount is important information, if available
        if wind_turbine_instances_data_category is not None:
            total_curtailment_amount = total_curtailment_amount.reindex(wind_farm_df.index).fillna(0).sum(axis=1)
        return wind_farm_instance, total_curtailment_amount

    @property
    def default_results_saving_path(self):
        saving_path = {
            "outlier": self.results_path / f"Filtering/{self.__str__()}/results.pkl",

            "operating regime": self.results_path / f"OperatingRegime/{self.__str__()}/report.csv",

            "operating regime single senor": self.results_path / f"OperatingRegime/{self.__str__()}/"
                                                                 f"single_sensor_classification.pkl",

            "fully operating regime power curve": self.results_path / f"PowerCurve/{self.__str__()}/fully_OPR_8PL.pkl",

            "fully operating regime power curve single senor": self.results_path / f"PowerCurve/{self.__str__()}/"
                                                                                   f"fully_OPR_8PL_single_senor.pkl",
            "resample_and_also_resample_operating_regime": self.results_path / f"resample/{self.__str__()}/results.pkl"
        }
        for x in saving_path.values():
            try_to_find_folder_path_otherwise_make_one(x.parent)
        return saving_path

    @staticmethod
    def infer_operating_regime_from_wind_turbine_instances_data_category(
            wind_turbine_instances_data_category: Sequence[DataCategoryData],
    ) -> DataCategoryData:
        """

        :param wind_turbine_instances_data_category:
        :return:
        """
        # %% Must be note that, in the classification of operating regime, WT-level 'shutdown' and 'nan' are treated
        # as the same group. Because they both have no contribution to the WF-level total power output calculation.
        # However, interestingly, note that WF-level wind speed calculate will be different!
        # Because 'shutdown' can provide OK wind speed, but 'nan' can not, especially due to elementwise deletion.
        wind_turbine_instances_data_category = copy.deepcopy(wind_turbine_instances_data_category)
        for obj in wind_turbine_instances_data_category:
            obj.abbreviation[np.isin(obj.abbreviation, ('shutdown', 'nan'))] = 'shutdown_or_nan'
        states_unique = ['operating', 'curtailed', 'shutdown_or_nan']

        def parse_by_states_unique_func(x) -> dict:
            return parse('(' + ', '.join(map(lambda y: '{' + y + '}', states_unique)) + ')', x).named

        # %% Obtain a pd.DataFrame obj that stores all WT-level information for convenience
        wind_turbine_instances_data_category_df = pd.DataFrame(dtype=str)
        for i, this_wind_turbine_instances_data_category in enumerate(wind_turbine_instances_data_category):
            wind_turbine_instances_data_category_df = pd.merge(
                wind_turbine_instances_data_category_df,
                this_wind_turbine_instances_data_category.pd_view,
                how='outer', left_index=True, right_index=True,
                suffixes=(f'_WT{i}', f'_WT{i + 1}')
            )
        wind_turbine_instances_data_category_df.fillna('shutdown_or_nan', inplace=True)
        wind_turbine_instances_data_category_df = wind_turbine_instances_data_category_df.astype(str)
        # %% Obtain a pd.DataFrame obj from the WF-level point of view
        operating_regime_df = pd.DataFrame(
            columns=states_unique + ['combination'],
            index=wind_turbine_instances_data_category_df.index,
        )
        for this_state in states_unique:
            operating_regime_df[this_state] = np.sum(
                wind_turbine_instances_data_category_df.values == this_state, 1
            ).astype(int)

        assert (int(np.unique(np.sum(operating_regime_df[states_unique].values, 1))) == len(
            wind_turbine_instances_data_category)), "'wind_turbine_instances_data_category' sum on axis_1 is wrong"
        operating_regime_df['combination'] = list(zip(*operating_regime_df[states_unique].values.T))
        operating_regime_df['combination'] = operating_regime_df['combination'].astype(str)
        # %% Infer a DataCategoryNameMapper obj
        combination_unique = np.unique(operating_regime_df['combination'])
        combination_unique = sorted(combination_unique,
                                    key=lambda x: (parse_by_states_unique_func(x)['operating'],
                                                   parse_by_states_unique_func(x)['curtailed'],
                                                   parse_by_states_unique_func(x)['shutdown_or_nan']),
                                    reverse=True)
        operating_regime_name_mapper = DataCategoryNameMapper.init_from_template()
        abbreviation_i = 0
        for i, this_combination in enumerate(combination_unique):
            # The rules for not considering:
            # WF-level outliers: if there are any WT recordings unexplainable
            if np.sum(this_combination == operating_regime_df['combination']) / len(operating_regime_df) < 0.0001:
                # if int(parse_by_states_unique_func(this_combination)['nan']) > 0:
                abbreviation_set = 'others'
            else:
                abbreviation_i += 1
                abbreviation_set = f"S{abbreviation_i}"
            operating_regime_name_mapper.loc[i] = [this_combination,
                                                   abbreviation_set,
                                                   -1,
                                                   parse_by_states_unique_func(this_combination).__str__()]
        # %% Obtain a DataCategoryData obj
        operating_regime = DataCategoryData(
            abbreviation=operating_regime_name_mapper.convert_sequence_data_key(
                'long name',
                'abbreviation',
                sequence_data=operating_regime_df['combination']
            ),
            index=operating_regime_df.index,
            name_mapper=operating_regime_name_mapper
        )
        # operating_regime.report(sorted_kwargs={'key': lambda x: "0" + x[1:] if x[1:].__len__() < 2 else x[1:]})
        return operating_regime

    @property
    def outlier_name_mapper(self) -> DataCategoryNameMapper:
        meta = [["missing data", "missing", -1, "N/A"],
                ["others", "others", 0, "N/A"],
                ["Linear Pout-WS", "CAT-III", 4, "e.g., constant WS-variable Pout"]]
        mapper = DataCategoryNameMapper.init_from_template(rows=len(meta))
        mapper[:] = meta
        return mapper

    def outlier_detector(self, *data_category_is_linearity_args,
                         save_file_path: Path = None,
                         extra_boundary_rules: Iterable[dict] = None,
                         **data_category_is_linearity_kwargs) -> DataCategoryData:
        save_file_path = save_file_path or self.default_results_saving_path["outlier"]
        if try_to_find_file(save_file_path):
            warnings.warn(f"{self.__str__()} has results in {save_file_path}")
            return load_pkl_file(save_file_path)['DataCategoryData obj']
        outlier = super().outlier_detector()  # type: DataCategoryData
        # %% CAT-III
        if data_category_is_linearity_args == ():
            data_category_is_linearity_args = ('60T',)
        if data_category_is_linearity_kwargs == {}:
            data_category_is_linearity_kwargs = {"constant_error": {'wind speed': 0.01}}
        cat_iii_outlier_mask = self.data_category_is_linearity(*data_category_is_linearity_args,
                                                               **data_category_is_linearity_kwargs)

        if extra_boundary_rules is not None:
            for this_boundary in extra_boundary_rules:
                cat_iii_outlier_mask = np.bitwise_or(cat_iii_outlier_mask,
                                                     self.data_category_inside_boundary(this_boundary))
        outlier.abbreviation[cat_iii_outlier_mask] = "CAT-III"

        ################################################################################################################
        # outlier = super(WF, self).outlier_detector()  # type: DataCategoryData
        # cat_iii_outlier_mask = self.data_category_is_linearity('6T', constant_error={'wind speed': 0.001})
        # if extra_boundary_rules is not None:
        #     for this_boundary in extra_boundary_rules:
        #         cat_iii_outlier_mask = np.bitwise_or(cat_iii_outlier_mask,
        #                                              self.data_category_inside_boundary(this_boundary))
        # outlier.abbreviation[cat_iii_outlier_mask] = "CAT-III"
        # ax = self[outlier('CAT-III')].plot(color='r', zorder = 2)
        # ax = self[outlier('others')].plot(ax=ax)
        # self[outlier('CAT-III')].plot(color='r')
        # self[outlier('others')].plot()
        # outlier.report()
        # debug_see = self[outlier('CAT-III')].pd_view()
        ################################################################################################################
        save_pkl_file(save_file_path,
                      {'raw_ndarray_data': np.array(outlier.abbreviation),
                       'raw_index': outlier.index,
                       'DataCategoryData obj': outlier})
        return outlier

    def outlier_detector_for_extra_feature(self) -> ndarray:
        """
        Return outlier for extra dim
        :return:
        """
        considered_extra_feature = (set(self.columns) - {'wind speed',
                                                         'active power output'}) & set(FEATURE_NORMAL_RANGE)
        # Outside boundary outlier
        out_of_range_outlier_mask = ~self.data_category_inside_boundary({key: FEATURE_NORMAL_RANGE[key]
                                                                         for key in considered_extra_feature})
        # Linear outlier
        linear_outlier_mask = self.data_category_is_linearity('30T',
                                                              constant_error={key: 0.00001
                                                                              for key in considered_extra_feature})
        # Combine
        outlier_mask = np.bitwise_or(out_of_range_outlier_mask, linear_outlier_mask)
        return outlier_mask

    def operating_regime_detector(self, task: str = 'load') -> Tuple[EquivalentWindFarmPowerCurve, DataCategoryData]:
        assert (task in ('load', 'fit')), "'Task' is not in ('load', 'fit')"
        pc_file_path = self.default_results_saving_path["fully operating regime power curve single senor"]
        operating_regime_file_path = self.default_results_saving_path["operating regime single senor"]

        # %% Prepare run the GA for a EquivalentWindFarmPowerCurve obj
        if task == 'fit':
            num_mask = np.bitwise_and(~np.isnan(self['wind speed'].values),
                                      ~np.isnan(self['active power output'].values))
        else:
            num_mask = np.full_like(self['wind speed'].values, fill_value=True).astype(bool)
        wf_pc_obj = EquivalentWindFarmPowerCurve(
            total_wind_turbine_number=self.number_of_wind_turbine,
            wind_speed_recording=self['wind speed'].values[num_mask],
            active_power_output_recording=self['active power output'].values[num_mask] / self.rated_active_power_output,
            index=self.index[num_mask]
        )
        # If there are any fitting results in the saving path, then they can be used as initials
        if try_to_find_file(pc_file_path):
            current_best = load_pkl_file(pc_file_path)[-1]['variable']
            wf_pc_obj.update_params(*current_best[:wf_pc_obj.params.__len__()])  # The last the best
            params_init_scheme = 'self'
        else:
            params_init_scheme = 'guess'
        if task == 'fit':
            wf_pc_obj.fit(
                ga_algorithm_param={'max_num_iteration': 10,
                                    'max_iteration_without_improv': 1000000,
                                    'population_size': 500},
                params_init_scheme=params_init_scheme,
                run_n_times=10000000,
                save_to_file_path=pc_file_path,
                focal_error=0.001,
                function_timeout=6000
            )

        else:
            print(f"best found = {wf_pc_obj}")
            operating_regime = wf_pc_obj.maximum_likelihood_estimation_for_wind_farm_operation_regime(
                task='evaluate',
                return_fancy=True
            )[-1]
            save_pkl_file(operating_regime_file_path, operating_regime)
            return wf_pc_obj, operating_regime

    def resample_and_also_resample_operating_regime(self,
                                                    resample_args: tuple = ('10T',),
                                                    resample_kwargs: dict = None, *,
                                                    operating_regime_file_path: Path = None,
                                                    additional_outlier_mask: ndarray):
        """
        Resample the wind farm, and most importantly, resample the operating regime.
        The method to resample the operating regime is to select the most frequent values in new sampling window.
        :return:
        """

        @load_exist_pkl_file_otherwise_run_and_save(
            self.default_results_saving_path['resample_and_also_resample_operating_regime'])
        def func():
            nonlocal operating_regime_file_path
            nonlocal resample_kwargs
            self_copy = copy.deepcopy(self)

            # For single sensor WF, the operating regime must be detected using as high resolution data as possible.
            # Therefore, object function 'operating_regime_detector' must be called (so there will be results in
            # self.default_results_saving_path["operating regime single senor"], or as specified)
            if operating_regime_file_path is None:
                operating_regime_file_path = self.default_results_saving_path["operating regime single senor"]
            operating_regime = load_pkl_file(operating_regime_file_path)  # type: DataCategoryData
            assert operating_regime is not None

            resample_kwargs = resample_kwargs or {
                'resampler_obj_func_source_code': "agg(lambda x: np.mean(x.values))"
            }
            # Check outlier
            existing_outlier = load_pkl_file(self_copy.default_results_saving_path["outlier"])['DataCategoryData obj']
            self_copy.loc[np.bitwise_or(~existing_outlier('others'),
                                        additional_outlier_mask), :] = np.nan
            # Resample self
            resampled_self = self_copy.resample(*resample_args, **resample_kwargs)
            resampled_self.obj_name = self_copy.obj_name + f" resampled"

            # Resample operating regime
            def rolling_func(x):
                (values, counts) = np.unique(x, return_counts=True)
                index = np.argmax(counts)
                return int(values[index])

            rolling_obj = operating_regime.pd_view.applymap(lambda x: int(x[1:])).rolling(*resample_args)
            resampled_operating_regime = rolling_obj.apply(rolling_func, raw=True)
            resampled_operating_regime = resampled_operating_regime.reindex(resampled_self.index, method='nearest')
            resampled_operating_regime = resampled_operating_regime.astype(int)
            resampled_operating_regime = DataCategoryData(
                abbreviation=resampled_operating_regime.applymap(lambda x: f"S{int(x)}").values.flatten(),
                index=resampled_operating_regime.index
            )

            return resampled_self, resampled_operating_regime

        return func()

    def plot(self, *,
             ax=None,
             plot_mfr: Iterable[PowerCurveByMfr] = None,
             operating_regime: DataCategoryData = None,
             plot_individual: bool = False,
             not_show_color_bar=False,
             **kwargs):
        if operating_regime is None:
            ax = super(WF, self).plot(ax=ax, plot_mfr=plot_mfr, **kwargs)
        else:
            # Analyse the unique abbreviation
            unique_abbreviation = np.unique(operating_regime.abbreviation)
            unique_abbreviation = unique_abbreviation[unique_abbreviation != 'others']
            unique_abbreviation_sort = sorted(unique_abbreviation, key=lambda x: int(parse(r"S{}", x)[0]))
            unique_abbreviation_sort = np.append(unique_abbreviation_sort, 'others')
            # Prepare assigned colors
            cmap_name = 'jet'  # 'copper', 'jet', 'cool'
            custom_cm = plt.cm.get_cmap(cmap_name, unique_abbreviation_sort.__len__())
            color_list = custom_cm(range(unique_abbreviation_sort.__len__()))[np.newaxis, :, :3]
            for i, this_operating_regime in enumerate(unique_abbreviation_sort):
                if this_operating_regime in ('S1', 'others'):
                    ax = super(WF, self[operating_regime('S1')]).plot(ax=ax if not plot_individual else None,
                                                                      plot_mfr=plot_mfr, zorder=-1,
                                                                      color=tuple(color_list[:, 0, :].squeeze()),
                                                                      **kwargs)
                this_operating_regime_mask = operating_regime(this_operating_regime)
                ax = scatter(self[this_operating_regime_mask]['wind speed'],
                             self[this_operating_regime_mask]['active power output'] / self.rated_active_power_output,
                             ax=ax if not plot_individual else None,
                             color=tuple(color_list[:, i, :].squeeze()),
                             **kwargs)

            # Color bar codes
            if not not_show_color_bar:
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_name, unique_abbreviation_sort.__len__()), norm=norm)
                sm.set_array([])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("top", size="5%", pad=0.05)
                cbar = plt.colorbar(sm, cax=cax, ax=ax, ticks=(), orientation='horizontal')
                for j, lab in enumerate(unique_abbreviation_sort):
                    if lab == 'others':
                        lab = 'Others'
                    else:
                        lab = operating_regime.name_mapper.infer_from_abbreviation(lab)['long name'].values[0]
                    cbar.ax.text((2 * j + 1) / (unique_abbreviation_sort.__len__() * 2), 3, lab, ha='center',
                                 va='center',
                                 fontsize=10, rotation=45)
                """
                top=0.89,
                bottom=0.125,
                left=0.11,
                right=0.995,
                hspace=0.2,
                wspace=0.2
                """

        return ax


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
