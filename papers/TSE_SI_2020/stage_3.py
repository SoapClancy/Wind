from papers.TSE_SI_2020.stage_2 import get_model as get_copula_model
from papers.TSE_SI_2020.stage_2 import get_copula_outputs, transform_copula_conditional_pdfs
from papers.TSE_SI_2020.stage_2 import get_data as stage_2_get_data
from Ploting.fast_plot_Func import *
from project_utils import *
from WT_WF_Class import WF
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, CLUSTER_TO_WF_MAPPER, WF_TO_CLUSTER_MAPPER, \
    Croatia_WF_LOCATION_MAPPER
import pandas as pd
from UnivariateAnalysis_Class import Univariate, UnivariateProbabilisticModel, UnivariatePDFOrCDFLike, ECDF
from typing import List
import datetime
from ConvenientDataType import StrOneDimensionNdarray, UncertaintyDataFrame
from Ploting.uncertainty_plot_Func import plot_from_uncertainty_like_dataframe
from Ploting.adjust_Func import adjust_legend_in_ax
from ErrorEvaluation_Class import ProbabilisticError, DeterministicError

NN_MODEL_PREDICTION_PATH = project_path_ / r"Data\Results\Forecasting\NN_model_predictions"
COPULA_MODEL_PATH = project_path_ / r'Data\Results\4d_cvine_gmcm_model'
FINAL_COMBINATION_RESULTS_FOLDER = project_path_ / r"Data\Results\Forecasting\Final_Combination_Results"

COPULA_LINSPACE_NUMBER = 500
USE_STAGE_1_SAMPLE_SIZE = 3000
COMBINATION_SAMPLE_SIZE = 5000


def get_stage_1_results(wf_name: str):
    pred_natural_resources = load_pkl_file(NN_MODEL_PREDICTION_PATH / fr"{wf_name}/EveryThing/test_set_predictions.pkl")
    pred_opr = load_pkl_file(NN_MODEL_PREDICTION_PATH / fr"{wf_name}/OPR/test_set_predictions.pkl")
    return pred_natural_resources, pred_opr


def combine_copula_outputs(copula_outputs: List[UnivariatePDFOrCDFLike]):
    samples = []
    for ele in copula_outputs:
        samples.append(ele.sample(COMBINATION_SAMPLE_SIZE))
    samples = np.array(samples)
    ecdf_obj = ECDF(samples.flatten())
    x = np.linspace(np.min(samples.flatten()), np.max(samples.flatten()), 10000)
    y = ecdf_obj(x)
    ans = UnivariatePDFOrCDFLike(cdf_like_ndarray=np.stack([y, x], axis=1))
    return ans


def cal_final_results_for_one_wf(wf_name: str, idx_s: int = None, idx_e: int = None):
    print("⭐" * 79)
    print(f"Now calculate the final results for {wf_name}")
    print("⭐" * 79)

    pred_natural_resources, pred_opr = get_stage_1_results(wf_name)
    vine_gmcm_copula_left, vine_gmcm_copula_right = get_copula_model(wf_name, 'test')
    test_data = stage_2_get_data(wf_name, 'test')[1]
    target_pout = test_data.iloc[test_data.shape[0] // 2:]['active power output'].values

    final_results_path = FINAL_COMBINATION_RESULTS_FOLDER / wf_name
    try_to_find_folder_path_otherwise_make_one(final_results_path)

    idx_s = idx_s or 0
    idx_e = idx_e or len(target_pout)
    for i in range(idx_s, idx_e):
        now_file_path = final_results_path / f"i_eq_{i}_results.pkl"
        if try_to_find_file(now_file_path):
            continue

        _t1 = datetime.datetime.now()
        copula_inputs = pred_natural_resources['prediction_results_inv'][:USE_STAGE_1_SAMPLE_SIZE, i, 0, :]
        copula_inputs = pd.DataFrame(copula_inputs,
                                     columns=['wind speed', 'air density', 'wind direction'])
        now_copula_outputs = get_copula_outputs(
            wind_farm_name=wf_name,
            copula_model_left=vine_gmcm_copula_left,
            copula_model_right=vine_gmcm_copula_right,
            copula_inputs=copula_inputs,
            linspace_number=COPULA_LINSPACE_NUMBER
        )
        now_copula_outputs = transform_copula_conditional_pdfs(
            wind_farm_name=wf_name,
            copula_outputs=now_copula_outputs,
            normally_operating_number=pred_opr['prediction_results_inv'][:USE_STAGE_1_SAMPLE_SIZE, i, 0, 0]
        )

        now_ans = combine_copula_outputs(now_copula_outputs)
        _t2 = datetime.datetime.now()
        print(f"time stamp = {i} finished!")
        print(f"Actual = {target_pout[i]}")
        print(f"Forecast mean = {now_ans.mean_}")
        print(f"Forecast 2.5 - 97.5 percentiles = {now_ans.find_nearest_inverse_cdf([0.025, 0.975])}")
        print(f"Forecast 5 - 95 percentiles = {now_ans.find_nearest_inverse_cdf([0.05, 0.95])}")
        print(f"Forecast 10 - 90 percentiles = {now_ans.find_nearest_inverse_cdf([0.10, 0.9])}")
        print(f"Time usage = {_t2 - _t1}")
        print('\n')
        # save
        save_pkl_file(final_results_path / f"i_eq_{i}_results.pkl", now_ans.cdf_like_ndarray)


def plot_stage_1_results(wf_name: str):
    pred_natural_resources, pred_opr = get_stage_1_results(wf_name)

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


def get_final_results_for_one_wf(wf_name: str, also_plot: bool = False):
    pred_natural_resources, pred_opr = get_stage_1_results(wf_name)
    test_data = stage_2_get_data(wf_name, 'test')[1]
    target_pout = test_data.iloc[test_data.shape[0] // 2:]['active power output'].values

    final_results_path = FINAL_COMBINATION_RESULTS_FOLDER / wf_name
    preds = np.empty(len(target_pout), dtype=object)
    preds_mean = np.full(len(target_pout), np.nan)
    preds_2p5 = np.full(len(target_pout), np.nan)
    preds_97p5 = np.full(len(target_pout), np.nan)
    preds_samples = np.full((len(target_pout), 3000), np.nan)
    for i in range(len(target_pout)):
        now_file_path = final_results_path / f"i_eq_{i}_results.pkl"
        try_find = load_pkl_file(now_file_path)
        if try_find is not None:
            now_preds = UnivariatePDFOrCDFLike(cdf_like_ndarray=try_find)
            preds[i] = now_preds
            preds_mean[i] = now_preds.mean_ / WF_RATED_POUT_MAPPER[wf_name]
            preds_2p5[i] = now_preds.find_nearest_inverse_cdf(0.025) / WF_RATED_POUT_MAPPER[wf_name]
            preds_97p5[i] = now_preds.find_nearest_inverse_cdf(0.975) / WF_RATED_POUT_MAPPER[wf_name]
            preds_samples[i] = now_preds.sample(3000) / WF_RATED_POUT_MAPPER[wf_name]

    if also_plot:
        # construct uncertainty df
        uct_df = UncertaintyDataFrame.init_from_template(columns_number=preds_samples.shape[0],
                                                         percentiles=np.arange(0, 100 + 1 / 3000 * 100, 1 / 3000 * 100))
        for i in range(preds_samples.shape[0]):
            uct_df.update_one_column(i, data=preds_samples[i])

        ax = plot_from_uncertainty_like_dataframe(
            x=np.arange(preds_samples.shape[0]),
            uncertainty_like_dataframe=uct_df,
            lower_half_percentiles=StrOneDimensionNdarray(np.arange(5, 50, 5).astype(str)),
            cmap_name='gray',
            facecolor_shift_factor=-1.5,
            show_coverage_labels=StrOneDimensionNdarray(np.arange(5, 50, 5).astype(str)),
            figure_size=(6, 5 * 0.551),
            x_label='Time [Hour]',
            x_ticks=(np.arange(0, 168 + 1, 24), np.arange(0, 168 + 1, 24)),
            x_lim=(-1, 169),
            y_label=f'{wf_name} WF Power Output [p.u.]',
            y_lim=WS_POUT_2D_PLOT_KWARGS['y_lim']
        )
        ax = series(target_pout / WF_RATED_POUT_MAPPER[wf_name], color='red', ax=ax,
                    linestyle='-.', linewidth=1.2, alpha=0.95, label='Actual')
        ax = series(preds_mean, ax=ax, color='royalblue', linewidth=1.2, alpha=0.95, label='Mean')

        ax = adjust_legend_in_ax(ax, protocol='Outside center right')

    return target_pout, preds


def _per_unitise_preds(wf_name: str, preds):
    ans = np.empty(preds.shape, dtype=object)
    for i, ele in enumerate(preds):
        if ele is not None:
            cdf_like_ndarray = ele.cdf_like_ndarray
            cdf_like_ndarray[:, 1] /= WF_RATED_POUT_MAPPER[wf_name]
            ans[i] = (UnivariatePDFOrCDFLike(cdf_like_ndarray=cdf_like_ndarray))
    return ans


def get_final_errors_for_one_wf(wf_name: str, also_plot: bool = False):
    target_pout, preds = get_final_results_for_one_wf(wf_name, False)
    preds = _per_unitise_preds(wf_name, preds)

    deter_error = DeterministicError(target=target_pout / WF_RATED_POUT_MAPPER[wf_name],
                                     model_output=np.array([x.mean_ if x is not None else np.nan for x in preds]))
    mae = deter_error.cal_mean_absolute_error()
    rmse = deter_error.cal_root_mean_square_error()

    prob_error = ProbabilisticError(target=target_pout / WF_RATED_POUT_MAPPER[wf_name],
                                    model_output=[x.cdf_estimate if x is not None else None for x in preds])
    crps = prob_error.cal_continuous_ranked_probability_score([-float_eps, 1 + float_eps])
    pinball_loss = prob_error.cal_pinball_loss()
    winker_score = prob_error.cal_winker_score(0.1)


if __name__ == '__main__':
    pass
    get_final_results_for_one_wf('Glunca', also_plot=True)
    # get_final_errors_for_one_wf('Glunca', also_plot=True)

    # get_final_results_for_one_wf('Jelinak', also_plot=True)
    # get_final_errors_for_one_wf('Jelinak', also_plot=True)
