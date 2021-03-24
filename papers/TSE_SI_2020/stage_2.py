from Ploting.fast_plot_Func import *
from project_utils import *
import numpy as np
import datetime
from WT_WF_Class import WF
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
import copy
from Correlation_Modeling.Copula_Class import VineGMCMCopula, FOUR_DIM_CVINE_CONSTRUCTION
from papers.TSE_SI_2020.stage_1 import get_natural_resources_or_opr_or_copula_data
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, CLUSTER_TO_WF_MAPPER, WF_TO_CLUSTER_MAPPER, \
    Croatia_WF_LOCATION_MAPPER, AVAILABLE_WF_NAMES
from PowerCurve_Class import PowerCurveByMethodOfBins
from BivariateAnalysis_Class import MethodOfBins
from Filtering.sklearn_novelty_and_outlier_detection_Func import use_isolation_forest
import pandas as pd
from UnivariateAnalysis_Class import DeterministicUnivariateProbabilisticModel, UnivariatePDFOrCDFLike
import json
from papers.TSE_SI_2020.utils import preds_continuous_var_plot, cal_continuous_var_error, \
    turn_preds_into_univariate_pdf_or_cdf_like
from ErrorEvaluation_Class import EnergyBasedError
from collections import ChainMap
from PowerCurve_Class import PowerCurveByMfr
from Filtering.OutlierAnalyser_Class import DataCategoryData, DataCategoryNameMapper

PRED_BY = "mean"
assert PRED_BY in {"mean", "median"}

WS_REGION = {
    'Glunca': [2., 16., 25.],
    'Jelinak': [1., 18.5, 26.5],
    # 'Zelengrad': [2., 18.3, 27.3],
    'Zelengrad': [2., 19.5, 27.3],

    'Bruska': [2.2, 18., 26.5],
    # 'Bruska': [2.2, 20., 26.5],
    'Lukovac': [2.0, 17.0, 26.5],
    'Katuni': [2., 17., 26.5]
}


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


def _set_ws_region(wind_farm_name):
    wf_obj_full, wf_obj = get_data(wind_farm_name, 'training')
    # wf_obj_full.plot(plot_scatter_pc=True)

    if wind_farm_name in WS_REGION:
        masks = [np.bitwise_and(wf_obj_full['wind speed'].values >= WS_REGION[wind_farm_name][0],
                                wf_obj_full['wind speed'].values < WS_REGION[wind_farm_name][1]),
                 np.bitwise_and(wf_obj_full['wind speed'].values >= WS_REGION[wind_farm_name][1],
                                wf_obj_full['wind speed'].values < WS_REGION[wind_farm_name][2])]
        for mask in masks:
            wf_obj_full[mask].plot(plot_scatter_pc=True)


def get_data(wind_farm_name: str, task: str, force_outlier_filter=False) -> Tuple[WF, WF]:
    assert task in ("training", "test")

    #
    wf_obj = get_natural_resources_or_opr_or_copula_data(wind_farm_name, task, res_name='Copula',
                                                         use_corr_impute='')
    wf_obj = wf_obj.reindex(['active power output',
                             'wind speed',
                             'air density',
                             'wind direction',
                             'normally operating number'], axis=1)

    mask = wf_obj['normally operating number'].values == NUMBER_OF_WT_MAPPER[wind_farm_name]
    wf_obj_full = wf_obj.loc[mask, ['active power output', 'wind speed', 'air density', 'wind direction']]

    force_outlier_filter_idx = None
    if task == 'training':
        if wind_farm_name == 'Glunca':
            temp = (0, 100)
        elif wind_farm_name == 'Jelinak':
            temp = (0, 67)
        elif wind_farm_name == 'Zelengrad':
            temp = (0, 90)
        elif wind_farm_name == 'Bruska':
            temp = (0, 95)
        elif wind_farm_name == 'Lukovac':
            temp = (0, 100)
        else:
            temp = (0, 97)

        outlier_mask = MethodOfBins(predictor_var=wf_obj_full['wind speed'].values,
                                    dependent_var=wf_obj_full['active power output'].values,
                                    bin_step=0.5).identify_percentile_outlier(*temp)
        # wf_obj_full.plot(title='before remove', plot_scatter_pc=True)
        # wf_obj_full.loc[~outlier_mask].plot(title='IF remove', plot_scatter_pc=True)
        wf_obj_full.loc[outlier_mask] = np.nan

        force_outlier_filter_idx = wf_obj_full.loc[outlier_mask].index
        # wf_obj_full.plot(title='remove DONE', plot_scatter_pc=True)

        #
        # wf_obj_full.plot(title='wf_obj_full to return', plot_scatter_pc=True)
        # wf_obj[wf_obj['normally operating number'].values == NUMBER_OF_WT_MAPPER[wind_farm_name] - 1].plot(title='-1')
        # wf_obj.plot(title='wf_obj to return')
    if force_outlier_filter:
        wf_obj.loc[force_outlier_filter_idx] = np.nan

    return wf_obj_full, wf_obj


def get_model(wind_farm_name: str, task: str):
    assert task in ("training", "test")
    folder_path = project_path_ / Path(rf"Data\Results\4d_cvine_gmcm_model\{wind_farm_name}\\")
    folder_path_left = folder_path / 'left'
    folder_path_right = folder_path / 'right'

    if task == "test":
        assert try_to_find_file(folder_path_left / Path('./marginal_training.pkl'))
        assert try_to_find_file(folder_path_right / Path('./marginal_training.pkl'))

    try_to_find_folder_path_otherwise_make_one(folder_path_left)
    try_to_find_folder_path_otherwise_make_one(folder_path_right)

    data = get_data(wind_farm_name, task)[0]
    left_mask = np.bitwise_and(data['wind speed'].values >= WS_REGION[wind_farm_name][0],
                               data['wind speed'].values < WS_REGION[wind_farm_name][1])
    right_mask = np.bitwise_and(data['wind speed'].values >= WS_REGION[wind_farm_name][1],
                                data['wind speed'].values < WS_REGION[wind_farm_name][2])

    vine_gmcm_copula_left = VineGMCMCopula(
        data[left_mask].values,
        construction=FOUR_DIM_CVINE_CONSTRUCTION,
        gmcm_model_folder_for_construction_path_=folder_path_left,
        marginal_distribution_file_=folder_path_left / Path('marginal_training.pkl')
    )
    vine_gmcm_copula_right = VineGMCMCopula(
        data[right_mask].values,
        construction=FOUR_DIM_CVINE_CONSTRUCTION,
        gmcm_model_folder_for_construction_path_=folder_path_right,
        marginal_distribution_file_=folder_path_right / Path('marginal_training.pkl')
    )
    # for j in range(4):
    #     hist(vine_gmcm_copula_left.ndarray_data[:, j])
    #     hist(vine_gmcm_copula_left.ndarray_data_in_uniform[:, j])

    return vine_gmcm_copula_left, vine_gmcm_copula_right


def train_model(wind_farm_name: str):
    _model_left, _model_right = get_model(wind_farm_name, "training")
    # only_at_edge_idx = None
    only_at_edge_idx = 5
    gmcm_fitting_attempt = 1

    _model = _model_left

    _model.fit(only_at_edge_idx=only_at_edge_idx, gmcm_fitting_attempt=gmcm_fitting_attempt,
               gmcm_fitting_k=6
               )

    return _model_left, _model_right


def get_copula_outputs(wind_farm_name: str,
                       copula_model_left: VineGMCMCopula,
                       copula_model_right: VineGMCMCopula, *,
                       copula_inputs: pd.DataFrame,
                       linspace_number: int):
    assert list(copula_inputs.columns) == ["wind speed", "air density", "wind direction"]

    # divide by region
    left_copula_inputs = []
    right_copula_inputs = []

    use_zero_idx = []
    use_left_idx = []
    use_right_idx = []

    for i in range(copula_inputs.shape[0]):
        if copula_inputs.iloc[i]['wind speed'] < WS_REGION[wind_farm_name][0]:
            use_zero_idx.append(i)
        elif copula_inputs.iloc[i]['wind speed'] < WS_REGION[wind_farm_name][1]:
            use_left_idx.append(i)
            left_copula_inputs.append(list(copula_inputs.iloc[i].values))
        elif copula_inputs.iloc[i]['wind speed'] < WS_REGION[wind_farm_name][2]:
            use_right_idx.append(i)
            right_copula_inputs.append(list(copula_inputs.iloc[i].values))
        else:
            use_zero_idx.append(i)
    left_copula_inputs = np.array(left_copula_inputs)
    right_copula_inputs = np.array(right_copula_inputs)

    left_copula_pred, right_copula_pred = None, None
    if left_copula_inputs.__len__() != 0:
        # expand_dim
        left_copula_inputs = np.concatenate((np.full((left_copula_inputs.shape[0], 1), np.nan),
                                             left_copula_inputs), axis=1)
        # run copula
        left_copula_pred = copula_model_left.cal_conditional_pdf_unnormalised(ndarray_data_like=left_copula_inputs,
                                                                              linspace_number=linspace_number)
    if right_copula_inputs.__len__() != 0:
        right_copula_inputs = np.concatenate((np.full((right_copula_inputs.shape[0], 1), np.nan),
                                              right_copula_inputs), axis=1)

        right_copula_pred = copula_model_right.cal_conditional_pdf_unnormalised(ndarray_data_like=right_copula_inputs,
                                                                                linspace_number=linspace_number)

    # merge results
    conditional_pdfs = []
    temp_a = 0
    temp_b = 0
    for i in range(copula_inputs.shape[0]):
        if i in use_zero_idx:
            conditional_pdfs.append(DeterministicUnivariateProbabilisticModel(0.))
        elif i in use_left_idx:
            conditional_pdfs.append(left_copula_pred[temp_a])
            temp_a += 1
        else:
            conditional_pdfs.append(right_copula_pred[temp_b])
            temp_b += 1

    return conditional_pdfs


def transform_copula_conditional_pdfs(wind_farm_name: str, copula_outputs: list, *,
                                      pred_normally_operating_number: ndarray,
                                      actual_normally_operating_number: ndarray):
    assert copula_outputs.__len__() == pred_normally_operating_number.__len__()
    assert pred_normally_operating_number.ndim == 1
    assert pred_normally_operating_number.shape == actual_normally_operating_number.shape

    ans = {
        'pred_opr': None,
        'actual_opr': None,
        'full_opr': None
    }
    full_opr = np.broadcast_to(NUMBER_OF_WT_MAPPER[wind_farm_name], pred_normally_operating_number.shape)

    for j, cared_opr in enumerate((pred_normally_operating_number, actual_normally_operating_number, full_opr)):
        transformed_dist_list = []
        for i in range(copula_outputs.__len__()):
            scale_factor = cared_opr[i] / NUMBER_OF_WT_MAPPER[wind_farm_name]
            transformed_dist = copy.deepcopy(copula_outputs[i])
            if type(transformed_dist) == UnivariatePDFOrCDFLike:
                pdf_like_ndarray = transformed_dist.pdf_like_ndarray
                pdf_like_ndarray[:, 1] *= scale_factor
                transformed_dist = UnivariatePDFOrCDFLike(
                    pdf_like_ndarray=pdf_like_ndarray  # Call the constructor, make sure every renormalisation works
                )

            transformed_dist_list.append(transformed_dist)

        if j == 0:
            ans['pred_opr'] = copy.deepcopy(transformed_dist_list)
        elif j == 1:
            ans['actual_opr'] = copy.deepcopy(transformed_dist_list)
        else:
            ans['full_opr'] = copy.deepcopy(transformed_dist_list)

    return ans


def test_model(wind_farm_name: str):
    # Get data
    test_data = get_data(wind_farm_name, "test")[1]
    test_data = test_data.iloc[test_data.shape[0] // 2:]
    # Build and get model
    vine_gmcm_copula_left, vine_gmcm_copula_right = get_model(wind_farm_name, "test")
    # run model
    copula_outputs = get_copula_outputs(wind_farm_name,
                                        copula_model_left=vine_gmcm_copula_left,
                                        copula_model_right=vine_gmcm_copula_right,
                                        copula_inputs=test_data[['wind speed', 'air density', 'wind direction']],
                                        linspace_number=500)

    # transform
    copula_trans = transform_copula_conditional_pdfs(
        wind_farm_name,
        copula_outputs,
        pred_normally_operating_number=test_data['normally operating number'].values.flatten(),  # fake
        actual_normally_operating_number=test_data['normally operating number'].values.flatten(),
    )

    tt = 1
    copula_outputs = copula_trans['pred_opr']
    samples = []
    for ele in copula_outputs:
        samples.append(ele.sample(3000) / WF_RATED_POUT_MAPPER[wind_farm_name])
    samples = np.array(samples)

    # Plot
    # pred_mean = []
    # pred_2p5 = []
    # pred_97p5 = []
    # for pred_pdf in copula_outputs:
    #     pred_mean.append(pred_pdf.mean_)
    #     pred_2p5.append(pred_pdf.find_nearest_inverse_cdf(0.025))
    #     pred_97p5.append(pred_pdf.find_nearest_inverse_cdf(0.975))
    # series(x=test_data.index, y=test_data.iloc[:, 1].values.flatten(), label="Mean", figure_size=(10, 2.4),
    #        x_axis_format='%m-%d %H')
    # ax = series(x=test_data.index, y=pred_mean, label="Mean", figure_size=(10, 2.4),
    #             x_axis_format='%m-%d %H')
    # ax = series(test_data.index, test_data.iloc[:, 0].values.flatten(),
    #             color="r", label="True", ax=ax)
    # ax = series(test_data.index, pred_2p5, linestyle="--",
    #             color="g", label="2.5 - 97.5 percentiles", ax=ax)
    # ax = series(test_data.index, pred_97p5, linestyle="--",
    #             color="g", ax=ax,
    #             x_label="Time index [Hour]", y_label=f"{wind_farm_name} Power Output\n[MW]")

    preds_continuous_var_plot(wf_name=wind_farm_name,
                              preds_samples=samples,
                              target_pout=test_data.iloc[:, 0].values.flatten(),
                              name='Pout')
    temp = copy.deepcopy([x if isinstance(x, UnivariatePDFOrCDFLike)
                          else UnivariatePDFOrCDFLike.init_from_samples_by_ecdf(x.sample(3000))
                          for x in copula_outputs])
    temp = turn_preds_into_univariate_pdf_or_cdf_like(wind_farm_name, preds=np.array(temp), per_unitise=True)
    error = cal_continuous_var_error(
        target=test_data.iloc[:, 0].values.flatten() / WF_RATED_POUT_MAPPER[wind_farm_name],
        model_output=temp,
        name='Pout'
    )
    if PRED_BY == 'mean':
        energy_error = EnergyBasedError(target=test_data.iloc[:, 0].values.flatten(),
                                        model_output=np.array([x.mean_ for x in copula_outputs]),
                                        time_step=1.)
    else:
        energy_error = EnergyBasedError(target=test_data.iloc[:, 0].values.flatten(),
                                        model_output=np.array([x.cal_median_val() for x in copula_outputs]),
                                        time_step=1.)
    energy_error_val = energy_error.do_calculation(
        drop_keys=('target_total_when_over', 'target_total_when_under', 'target_total')
    )

    print(f"Finished {wind_farm_name} ")
    print(error)
    return pd.DataFrame(dict(ChainMap(energy_error_val, error)), index=[wind_farm_name])


def test_all_models():
    file_path = project_path_ / fr"Data\Results\Forecasting\errors\stage2"
    try_to_find_folder_path_otherwise_make_one(file_path)

    errors = pd.DataFrame()
    for wf in AVAILABLE_WF_NAMES:
        error = test_model(wind_farm_name=wf)
        errors = pd.concat([errors, error])

    errors.to_csv(file_path / "all_wf_copula_errors.csv")


def plot_one_wf_with_opr(wind_farm_name):
    wf_obj = get_data(wind_farm_name, 'training')[1]
    tt = 1

    abbreviation = []
    for x in wf_obj.iloc[:, -1].values:
        if np.isnan(x):
            x = 0
        else:
            x = int(x)
        abbreviation.append(f"S{NUMBER_OF_WT_MAPPER[wind_farm_name] + 1 - x}")
    operating_regime_name_mapper = DataCategoryNameMapper.init_from_template()

    for i in range(NUMBER_OF_WT_MAPPER[wind_farm_name] + 1):
        operating_regime_name_mapper.loc[i] = [str(i),
                                               f"S{NUMBER_OF_WT_MAPPER[wind_farm_name] + 1 - i}",
                                               -1,
                                               '']

    operating_regime = DataCategoryData(
        abbreviation=abbreviation,
        index=wf_obj.index,
        name_mapper=operating_regime_name_mapper
    )
    wf_obj.plot(operating_regime=operating_regime,
                not_show_color_bar=True,
                title=wind_farm_name)


def plot_all_wf_with_opr():
    for wf in AVAILABLE_WF_NAMES:
        plot_one_wf_with_opr(wind_farm_name=wf)


if __name__ == "__main__":
    # _set_ws_region("Glunca")
    # _set_ws_region("Jelinak")
    # _set_ws_region("Zelengrad")
    # _set_ws_region("Bruska")
    # _set_ws_region("Lukovac")
    # _set_ws_region("Katuni")

    # train_model("Glunca")
    # test_model("Glunca")

    # train_model("Jelinak")
    # test_model("Jelinak")

    # train_model("Zelengrad")
    # test_model("Zelengrad")
    #
    # train_model("Bruska")
    # test_model("Bruska")
    # _test_data = get_data('Bruska', "test")[1].iloc[168:].pd_view()

    # train_model("Lukovac")
    # test_model("Lukovac")

    # train_model("Katuni")
    # test_model("Katuni")
    pass
    test_all_models()

    # plot_all_wf_with_opr()
