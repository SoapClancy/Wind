from Ploting.fast_plot_Func import *
import pandas as pd
from papers.TSE_SI_2020.stage_3 import get_stage_1_results
from prepare_datasets import *
from ConvenientDataType import UncertaintyDataFrame
from pycircstat.descriptive import percentile as circle_pct
from pycircstat.descriptive import mean as circle_mean
from pycircstat.descriptive import std as circle_std

for wf_name in AVAILABLE_WF_NAMES:
    for j in ['', '_cluster_']:
        df = get_stage_1_results(wf_name, j)[0]
        preds = df['prediction_results_inv']
        tt = 1
        for i, col_name in enumerate(['wind speed', 'air density', 'wind direction']):
            preds_samples = preds[:, :, 0, i].T
            uct_df = UncertaintyDataFrame.init_from_template(columns_number=preds_samples.shape[0],
                                                             percentiles=np.arange(0, 100 + 1 / 20 * 100, 1 / 20 * 100))
            for k in range(preds_samples.shape[0]):
                uct_df.update_one_column(k, data=preds_samples[k])
                if col_name == 'wind direction':
                    uct_df.iloc[:uct_df.last_nan_index + 1, k] = np.rad2deg(
                        circle_pct(np.deg2rad(preds_samples[k]),
                                   q=np.arange(0, 100 + 1 / 20 * 100, 1 / 20 * 100),
                                   q0=0)
                    )
                    uct_df.iloc[-2, k] = np.rad2deg(circle_mean(np.deg2rad(preds_samples[k])))
                    uct_df.iloc[-1, k] = np.rad2deg(circle_std(np.deg2rad(preds_samples[k])))

            uct_df.to_csv(f"./{'own' if j == '' else j}/{wf_name} {col_name} forecasting.csv")
