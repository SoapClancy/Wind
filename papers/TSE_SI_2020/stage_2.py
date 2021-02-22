from Ploting.fast_plot_Func import *
from project_utils import *
from prepare_datasets import load_croatia_data
import numpy as np
import datetime
from WT_WF_Class import WF
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from Regression_Analysis.DataSet_Class import DeepLearningDataSet
from Filtering.OutlierAnalyser_Class import DataCategoryData
import copy
import pandas as pd
from typing import Callable
from prepare_datasets import load_dalry_wind_farm_toy
from Correlation_Modeling.Copula_Class import VineGMCMCopula, FOUR_DIM_CVINE_CONSTRUCTION


class WindFarmDataSet:
    def __init__(self, *args, wind_farm: WF,
                 # operating_regime: DataCategoryData,
                 **kwargs):
        assert isinstance(wind_farm, WF)
        # assert isinstance(operating_regime, DataCategoryData)
        wind_farm_extended = self._down_sampling_to_one_hour(wind_farm)
        wind_farm_extended = self._expand_feature_dim_for_month_and_operating_regime(wind_farm_extended)
        super().__init__(*args,
                         original_data_set=wind_farm_extended,
                         **kwargs)

    @staticmethod
    def _expand_feature_dim_for_month_and_operating_regime(wind_farm):
        wind_farm['month'] = wind_farm.index.month
        return wind_farm

    @staticmethod
    def _down_sampling_to_one_hour(wind_farm):
        wind_farm_extended = copy.deepcopy(wind_farm.pd_view()).astype(float)
        wind_farm_extended = wind_farm_extended.resample('60T').mean()
        return wind_farm_extended


def get_data(wind_farm_name: str, task: str):
    assert task in ("training", "test")

    # Meta data according to wind farm names
    if wind_farm_name == "DalryWFToy":
        wind_farm_obj = load_dalry_wind_farm_toy()  # type: WF
        wind_farm_obj = wind_farm_obj[np.bitwise_and(wind_farm_obj["wind speed"] < 18,
                                                     wind_farm_obj["wind speed"] >= 3)]
        wind_farm_obj = wind_farm_obj[wind_farm_obj.concerned_dim]
        not_nan_mask = ~np.any(np.isnan(wind_farm_obj.values), axis=1)
        wind_farm_obj = wind_farm_obj[not_nan_mask]
        training_mask = wind_farm_obj.index < datetime.datetime(2009, 1, 1)
    else:
        raise NotImplemented

    # Deep copy, since there are some global variables. Then, slice according to "training" or "test"
    mask = training_mask if task == "training" else ~training_mask
    wind_farm_obj = wind_farm_obj[mask]
    wind_farm_obj = wind_farm_obj.reindex(['active power output', 'wind speed', 'air density', 'wind direction'],
                                          axis=1)
    return wind_farm_obj


def get_model(wind_farm_name: str, task: str):
    assert task in ("training", "test")
    folder_path = project_path_ / Path(rf"Data\Results\4d_cvine_gmcm_model\{wind_farm_name}\\")
    if task == "test":
        assert try_to_find_file(folder_path / Path('./marginal_training.pkl'))

    try_to_find_folder_path_otherwise_make_one(folder_path)
    data = get_data(wind_farm_name, task)

    vine_gmcm_copula = VineGMCMCopula(
        data.values,
        construction=FOUR_DIM_CVINE_CONSTRUCTION,
        gmcm_model_folder_for_construction_path_=folder_path,
        marginal_distribution_file_=folder_path / Path('marginal_training.pkl')
    )
    return vine_gmcm_copula


def train_model(wind_farm_name: str):
    _model = get_model(wind_farm_name, "training")
    only_at_edge_idx = None
    # only_at_edge_idx = 5
    gmcm_fitting_attempt = 5
    _model.fit(only_at_edge_idx=only_at_edge_idx, gmcm_fitting_attempt=gmcm_fitting_attempt)

    return _model


def test_model(wind_farm_name: str):
    # Get data
    test_data = get_data(wind_farm_name, "test")
    # Build and get model
    _model = get_model(wind_farm_name, "test")
    # Select samples to test
    test_sample_input = test_data[list(test_data.predictor_names)]
    test_sample_output = test_data[list(test_data.dependant_names)]

    # expand_dim
    i = 15
    slice_obj = slice(i * 24 * 7, (i + 1) * 24 * 7)
    test_sample_input_consider = test_sample_input.values[slice_obj]
    test_sample_input_consider = np.concatenate((np.full((test_sample_input_consider.shape[0], 1), np.nan),
                                                 test_sample_input_consider), axis=1)
    test_sample_output_consider = test_sample_output[slice_obj]
    # Formal test
    conditional_pdfs = _model.cal_conditional_pdf_unnormalised(ndarray_data_like=test_sample_input_consider,
                                                               linspace_number=2000)
    tt = 1
    600 / 168 / 10 * 300
    # Plot
    pred_mean = []
    pred_2p5 = []
    pred_97p5 = []
    for pred_pdf in conditional_pdfs:
        pred_mean.append(pred_pdf.mean_)
        pred_2p5.append(pred_pdf.find_nearest_inverse_cdf(0.025))
        pred_97p5.append(pred_pdf.find_nearest_inverse_cdf(0.975))
    ax = series(test_sample_input.index[slice_obj], pred_mean, label="Mean", figure_size=(10, 2.4))
    ax = series(test_sample_input.index[slice_obj], test_sample_output_consider.values.flatten(),
                color="r", label="True", ax=ax)
    ax = series(test_sample_input.index[slice_obj], pred_2p5, linestyle="--",
                color="g", label="2.5 - 97.5 percentiles", ax=ax)
    ax = series(test_sample_input.index[slice_obj], pred_97p5, linestyle="--",
                color="g", ax=ax,
                x_label="Time index [Hour]", y_label="Wind Farm Power Output\n[kW]")


if __name__ == "__main__":
    # train_model("DalryWFToy")
    test_model("DalryWFToy")
