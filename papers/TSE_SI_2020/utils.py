from Ploting.uncertainty_plot_Func import plot_from_uncertainty_like_dataframe
from Ploting.adjust_Func import adjust_legend_in_ax
import pandas as pd
from ErrorEvaluation_Class import ProbabilisticError, DeterministicError
from Ploting.fast_plot_Func import *
from typing import List
import datetime
from ConvenientDataType import StrOneDimensionNdarray, UncertaintyDataFrame
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, CLUSTER_TO_WF_MAPPER, WF_TO_CLUSTER_MAPPER, \
    Croatia_WF_LOCATION_MAPPER, AVAILABLE_WF_NAMES
from project_utils import *
import copy
from pycircstat.descriptive import mean as circle_mean
from pycircstat.descriptive import median as circle_median
from pycircstat.descriptive import percentile as circle_pct
from UnivariateAnalysis_Class import UnivariatePDFOrCDFLike

PRED_BY = "mean"
assert PRED_BY in {"mean", "median"}


def turn_preds_into_univariate_pdf_or_cdf_like(wf_name: str, preds, *, per_unitise: bool):
    ans = np.empty(preds.shape, dtype=object)
    for i, ele in enumerate(preds):
        if ele is not None:
            cdf_like_ndarray = ele.cdf_like_ndarray
            if per_unitise:
                cdf_like_ndarray[:, 1] /= WF_RATED_POUT_MAPPER[wf_name]
            ans[i] = (UnivariatePDFOrCDFLike(cdf_like_ndarray=cdf_like_ndarray))
    return ans


def preds_continuous_var_plot(wf_name: str, *,
                              preds_samples: ndarray = None,
                              uct_df: UncertaintyDataFrame = None,
                              target_pout: ndarray,
                              name: str):
    if preds_samples is not None:
        assert preds_samples.ndim == 2
    assert target_pout.ndim == 1
    assert name in {'WS', 'AD', 'WD', 'Pout'}

    if name == 'WS':
        y_label = f'{wf_name} WF Wind Speed [m/s]'
        y_lim = (-0.05, 31.05)
    elif name == 'AD':
        y_label = f'{wf_name} WF Air Density [kg/m' + '$^3$' + ']'
        y_lim = (1.16, 1.29)
    elif name == 'WD':
        y_label = f'{wf_name} WF Wind Direction [deg]'
        y_lim = (-1, 361)
    else:
        target_pout = copy.deepcopy(target_pout / WF_RATED_POUT_MAPPER[wf_name])
        y_label = f'{wf_name} WF Power Output [p.u.]'
        y_lim = WS_POUT_2D_PLOT_KWARGS['y_lim']

    # construct uncertainty df
    if uct_df is None:
        uct_df = UncertaintyDataFrame.init_from_template(columns_number=preds_samples.shape[0],
                                                         percentiles=np.arange(0, 100 + 1 / 3000 * 100, 1 / 3000 * 100))
        for i in range(preds_samples.shape[0]):
            uct_df.update_one_column(i, data=preds_samples[i])

    if name == 'WD':
        tt = 1
        for j in range(uct_df.shape[1]):
            uct_df.iloc[:uct_df.last_nan_index + 1, j] = np.rad2deg(
                circle_pct(np.deg2rad(preds_samples[j]),
                           q=np.arange(0, 100 + 1 / 3000 * 100, 1 / 3000 * 100),
                           q0=0)
            )
            uct_df.iloc[-2, j] = np.rad2deg(circle_mean(np.deg2rad(preds_samples[j])))

    ax = plot_from_uncertainty_like_dataframe(
        x=np.arange(uct_df.shape[1]),
        uncertainty_like_dataframe=uct_df,
        lower_half_percentiles=StrOneDimensionNdarray(np.arange(5, 50, 5).astype(str)),
        cmap_name='gray',
        facecolor_shift_factor=-1.,
        show_coverage_labels=StrOneDimensionNdarray(np.arange(5, 50, 5).astype(str)),
        figure_size=(6, 5 * 0.551),
        x_label='Time [Hour]',
        x_ticks=(np.arange(0, 168 + 1, 24), np.arange(0, 168 + 1, 24)),
        x_lim=(-1, 169),
        y_label=y_label,
        y_lim=y_lim
    )
    ax = series(target_pout, color='red', ax=ax,
                linestyle='-.', linewidth=1.2, alpha=0.95, label='Actual')
    if PRED_BY == 'mean':
        ax = series(uct_df.loc['mean'].values.flatten(), ax=ax, color='royalblue', linewidth=1.2, alpha=0.95,
                    label='Pred.')
    else:
        ax = series(uct_df(by_percentile=50.).flatten(), ax=ax, color='royalblue', linewidth=1.2, alpha=0.95,
                    label='Pred.')

    ax = adjust_legend_in_ax(ax, protocol='Outside center right')
    return ax


def cal_continuous_var_error(target, model_output: List[UnivariatePDFOrCDFLike], *, name: str):
    assert len(target) == len(model_output)
    assert name in {'WS', 'AD', 'WD', 'Pout'}
    if PRED_BY == 'mean':
        deter_error = DeterministicError(target=target,
                                         model_output=[x.mean_ for x in model_output])
    else:
        deter_error = DeterministicError(target=target,
                                         model_output=[x.cal_median_val() for x in model_output])
    mae = deter_error.cal_mean_absolute_error()
    rmse = deter_error.cal_root_mean_square_error()

    if name == 'WS':
        integral_boundary = [-float_eps, 35 + float_eps]
    elif name == 'AD':
        integral_boundary = [0.9, 1.6]
    elif name == 'WD':  # Actually the sin or cos outputs
        integral_boundary = [-1 - float_eps, 1 + float_eps]
    else:
        integral_boundary = [-float_eps, 1 + float_eps]

    prob_error = ProbabilisticError(target=target,
                                    model_output=model_output)
    crps = prob_error.cal_continuous_ranked_probability_score(integral_boundary=integral_boundary)
    pinball_loss = prob_error.cal_pinball_loss()

    return {
        'mae': mae,
        'rmse': rmse,
        'pinball_loss': pinball_loss,
        'crps': crps,

    }
