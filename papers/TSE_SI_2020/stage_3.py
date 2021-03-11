from papers.TSE_SI_2020.stage_2 import get_model as get_copula_model
from papers.TSE_SI_2020.stage_2 import get_copula_outputs, transform_copula_conditional_pdfs
from papers.TSE_SI_2020.stage_2 import get_data as stage_2_get_data
from Ploting.fast_plot_Func import *
from project_utils import *
from WT_WF_Class import WF
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, CLUSTER_TO_WF_MAPPER, WF_TO_CLUSTER_MAPPER, \
    Croatia_WF_LOCATION_MAPPER, AVAILABLE_WF_NAMES
import pandas as pd
from UnivariateAnalysis_Class import Univariate, UnivariateProbabilisticModel, UnivariatePDFOrCDFLike, ECDF
from typing import List
import datetime
from ConvenientDataType import StrOneDimensionNdarray, UncertaintyDataFrame
from Ploting.uncertainty_plot_Func import plot_from_uncertainty_like_dataframe
from Ploting.adjust_Func import adjust_legend_in_ax
from ErrorEvaluation_Class import ProbabilisticError, DeterministicError
from papers.TSE_SI_2020.utils import preds_continuous_var_plot, turn_preds_into_univariate_pdf_or_cdf_like
from scipy import stats

NN_MODEL_PREDICTION_PATH = project_path_ / r"Data\Results\Forecasting\NN_model_predictions"
COPULA_MODEL_PATH = project_path_ / r'Data\Results\4d_cvine_gmcm_model'
FINAL_COMBINATION_RESULTS_FOLDER = project_path_ / r"Data\Results\Forecasting\Final_Combination_Results"

COPULA_LINSPACE_NUMBER = 500
USE_STAGE_1_SAMPLE_SIZE = 3000
COMBINATION_SAMPLE_SIZE = 5000


def get_stage_1_results(wf_name: str, use_corr_impute: str):
    pred_natural_resources = load_pkl_file(
        NN_MODEL_PREDICTION_PATH / fr"{wf_name}/EveryThing{use_corr_impute}/test_set_predictions.pkl"
    )
    pred_opr = load_pkl_file(
        NN_MODEL_PREDICTION_PATH / fr"{wf_name}/OPR/test_set_predictions.pkl"
    )
    return pred_natural_resources, pred_opr


def combine_copula_outputs(copula_outputs: List[UnivariatePDFOrCDFLike]):
    samples = []
    for ele in copula_outputs:
        samples.append(ele.sample(COMBINATION_SAMPLE_SIZE))
    samples = np.array(samples)
    ans = UnivariatePDFOrCDFLike.init_from_samples_by_ecdf(samples)
    return ans


def cal_final_results_for_one_wf(wf_name: str, idx_s: int = None, idx_e: int = None, *, use_corr_impute: str):
    print("⭐" * 79)
    print(f"Now calculate the final results for {wf_name}{use_corr_impute}")
    print("⭐" * 79)

    pred_natural_resources, pred_opr = get_stage_1_results(wf_name, use_corr_impute)
    vine_gmcm_copula_left, vine_gmcm_copula_right = get_copula_model(wf_name, 'test')
    test_data = stage_2_get_data(wf_name, 'test')[1]
    target_pout = test_data.iloc[test_data.shape[0] // 2:]['active power output'].values

    final_results_path = FINAL_COMBINATION_RESULTS_FOLDER / (wf_name + use_corr_impute)
    try_to_find_folder_path_otherwise_make_one(final_results_path)

    idx_s = idx_s or 0
    idx_e = idx_e or len(target_pout)
    for i in range(idx_s, idx_e):
        if try_to_find_file(final_results_path / 'actual_opr' / f"i_eq_{i}_results.pkl"):
            continue

        _t1 = datetime.datetime.now()
        copula_inputs = pred_natural_resources['prediction_results_inv'][:USE_STAGE_1_SAMPLE_SIZE, i, 0, :]
        copula_inputs = pd.DataFrame(copula_inputs,
                                     columns=['wind speed', 'air density', 'wind direction'])
        copula_outputs = get_copula_outputs(
            wind_farm_name=wf_name,
            copula_model_left=vine_gmcm_copula_left,
            copula_model_right=vine_gmcm_copula_right,
            copula_inputs=copula_inputs,
            linspace_number=COPULA_LINSPACE_NUMBER
        )
        now_copula_outputs = transform_copula_conditional_pdfs(
            wind_farm_name=wf_name,
            copula_outputs=copula_outputs,
            pred_normally_operating_number=pred_opr['prediction_results_inv'][:USE_STAGE_1_SAMPLE_SIZE, i, 0, 0],
            actual_normally_operating_number=np.broadcast_to(pred_opr['test_samples_y_inv'][i, 0, 0],
                                                             (USE_STAGE_1_SAMPLE_SIZE,)),
        )  # type:dict
        print(f"time stamp = {i} finished!")
        print(f"Actual = {target_pout[i]}")

        for key, val in now_copula_outputs.items():
            now_ans = combine_copula_outputs(val)
            print(f"if...using... {key}")
            print(f"Forecast mean = {now_ans.mean_}")
            print(f"Forecast 2.5 - 97.5 percentiles = {now_ans.find_nearest_inverse_cdf([0.025, 0.975])}")
            print(f"Forecast 5 - 95 percentiles = {now_ans.find_nearest_inverse_cdf([0.05, 0.95])}")
            print(f"Forecast 10 - 90 percentiles = {now_ans.find_nearest_inverse_cdf([0.10, 0.9])}")

            # save
            try_to_find_folder_path_otherwise_make_one(final_results_path / key)
            save_pkl_file(final_results_path / key / f"i_eq_{i}_results.pkl", now_ans.cdf_like_ndarray)
        print('\n')
        _t2 = datetime.datetime.now()
        print(f"Time usage = {_t2 - _t1}")


def plot_stage_1_results(wf_name: str, *, use_corr_impute: str):
    pred_natural_resources, pred_opr = get_stage_1_results(wf_name, use_corr_impute=use_corr_impute)

    def make_source_codes(variable='pred_natural_resources'):
        func_name = 'series' if variable == 'pred_natural_resources' else 'step'
        source_code = f"""\
        for j in range({variable}['test_samples_y_inv'].shape[-1]):
            ax = {func_name}({variable}['test_samples_y_inv'][:, 0, j], color='red', figure_size=(10, 2.4))
            ax = {func_name}(np.mean({variable}['prediction_results_inv'], axis=0)[:, 0, j], ax=ax,
                        color='royalblue')
            {func_name}({variable}['prediction_results_inv'][:600, :, 0, j].T, color='grey',
                   linewidth=0.8, ax=ax, alpha=0.1, zorder=-1)
    
            # ax = {func_name}({variable}['test_samples_y_inv'][:, 0, j], color='red')
            # ax = {func_name}(np.mean({variable}['prediction_results_inv'], axis=0)[:, 0, j], ax=ax,
            #             color='royalblue')
            # ax = {func_name}(np.percentile({variable}['prediction_results_inv'], 2.5, axis=0)[:, 0, j], ax=ax,
            #             color='green', linestyle='--')
            # ax = {func_name}(np.percentile({variable}['prediction_results_inv'], 97.5, axis=0)[:, 0, j], ax=ax,
            #             color='green', linestyle='--')\
        """
        indent = re.search("for", source_code.split("\n")[0]).regs[0][0]
        source_code = "\n".join([x[indent:] for x in source_code.split("\n")])
        return source_code

    exec(make_source_codes())
    exec(make_source_codes('pred_opr'))


def get_final_results_for_one_wf(wf_name: str, also_plot: bool = True, *,
                                 use_corr_impute: str,
                                 opr: str = 'pred_opr'):
    assert opr in {'pred_opr', 'actual_opr', 'full_opr'}

    pred_natural_resources, pred_opr = get_stage_1_results(wf_name, use_corr_impute=use_corr_impute)
    test_data = stage_2_get_data(wf_name, 'test')[1]
    target_pout = test_data.iloc[test_data.shape[0] // 2:]['active power output'].values

    final_results_path = FINAL_COMBINATION_RESULTS_FOLDER / (wf_name + use_corr_impute)
    preds = np.empty(len(target_pout), dtype=object)
    preds_mean = np.full(len(target_pout), np.nan)
    preds_2p5 = np.full(len(target_pout), np.nan)
    preds_97p5 = np.full(len(target_pout), np.nan)
    preds_samples = np.full((len(target_pout), 3000), np.nan)
    for i in range(len(target_pout)):
        now_file_path = final_results_path / opr / f"i_eq_{i}_results.pkl"
        try_find = load_pkl_file(now_file_path)
        if try_find is not None:
            now_preds = UnivariatePDFOrCDFLike(cdf_like_ndarray=try_find)
            preds[i] = now_preds
            preds_mean[i] = now_preds.mean_ / WF_RATED_POUT_MAPPER[wf_name]
            preds_2p5[i] = now_preds.find_nearest_inverse_cdf(0.025) / WF_RATED_POUT_MAPPER[wf_name]
            preds_97p5[i] = now_preds.find_nearest_inverse_cdf(0.975) / WF_RATED_POUT_MAPPER[wf_name]
            preds_samples[i] = now_preds.sample(3000) / WF_RATED_POUT_MAPPER[wf_name]

    tt = 1
    if also_plot:
        ax = preds_continuous_var_plot(wf_name=wf_name,
                                       preds_samples=preds_samples,
                                       target_pout=target_pout,
                                       name='Pout')

    return target_pout, preds


def get_final_errors_for_one_wf(wf_name: str, *, use_corr_impute: str, opr: str = 'pred_opr'):
    target_pout, preds = get_final_results_for_one_wf(wf_name, False, use_corr_impute=use_corr_impute, opr=opr)
    preds = turn_preds_into_univariate_pdf_or_cdf_like(wf_name, preds, per_unitise=True)

    deter_error = DeterministicError(target=target_pout / WF_RATED_POUT_MAPPER[wf_name],
                                     model_output=np.array([x.mean_ if x is not None else np.nan for x in preds]))
    mae = deter_error.cal_mean_absolute_error()
    rmse = deter_error.cal_root_mean_square_error()

    prob_error = ProbabilisticError(target=target_pout / WF_RATED_POUT_MAPPER[wf_name],
                                    model_output=[x.cdf_estimate if x is not None else None for x in preds],
                                    reduce_method='median')
    crps = prob_error.cal_continuous_ranked_probability_score([-float_eps, 1 + float_eps])
    pinball_loss = prob_error.cal_pinball_loss()
    winker_score = prob_error.cal_winker_score(0.1)
    tt = 1
    print(f"mae = {mae:.3f}")
    print(f"rmse = {rmse:.3f}")
    print(f"crps = {crps:.3f}")
    print(f"pinball_loss = {pinball_loss:.3f}")
    print(f"winker_score = {winker_score:.3f}")


def get_final_results_all(*, use_corr_impute):
    for wf_name in AVAILABLE_WF_NAMES:
        get_final_results_for_one_wf(wf_name, also_plot=True, use_corr_impute=use_corr_impute)


if __name__ == '__main__':
    # get_final_results_for_one_wf('Zelengrad', also_plot=True, use_corr_impute='')
    # get_final_results_all(use_corr_impute='')
    pass
    # get_final_results_for_one_wf('Jelinak', use_corr_impute='')
    # get_final_errors_for_one_wf('Jelinak', use_corr_impute='')
    # print("*" * 79)
    # get_final_errors_for_one_wf('Jelinak', use_corr_impute='_cluster_')
    # print("*" * 79)
    # get_final_errors_for_one_wf('Bruska', use_corr_impute='')
    # print("*" * 79)
    # get_final_errors_for_one_wf('Bruska', use_corr_impute='_cluster_')

    # get_final_errors_for_one_wf('Bruska', use_corr_impute='_cluster_', opr='full_opr')
    # print("*" * 79)
    # get_final_errors_for_one_wf('Bruska', use_corr_impute='_cluster_', opr='actual_opr')
    # print("*" * 79)
    # get_final_errors_for_one_wf('Bruska', use_corr_impute='_cluster_', opr='pred_opr')
    # print("*" * 79)

    # get_final_results_for_one_wf('Jelinak', use_corr_impute='', opr='pred_opr')

    # get_final_results_for_one_wf('Bruska', use_corr_impute='_cluster_', opr='pred_opr')
    # get_final_results_for_one_wf('Bruska', use_corr_impute='_cluster_', opr='actual_opr')
    # get_final_results_for_one_wf('Bruska', use_corr_impute='_cluster_', opr='full_opr')

    # plot_stage_1_results('Bruska', use_corr_impute='')
    plot_stage_1_results('Katuni', use_corr_impute='_cluster_')
