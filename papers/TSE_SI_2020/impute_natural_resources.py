from Ploting.fast_plot_Func import *
from project_utils import *
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, CLUSTER_TO_WF_MAPPER, WF_TO_CLUSTER_MAPPER, \
    Croatia_WF_LOCATION_MAPPER, AVAILABLE_WF_NAMES
import numpy as np
import datetime
from WT_WF_Class import WF
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import copy

tfd = eval("tfp.distributions")
tfpl = eval("tfp.layers")
tfb = eval("tfp.bijectors")
tfp_util = eval("tfp.util")
tfp_math = eval("tfp.math")
tfp_sts = eval("tfp.sts")

tf.keras.backend.set_floatx('float32')

SHARED_DIR_PATH = project_path_ / r"Data\Raw_measurements\TSE_SI_2020_WF\shared_data"
IMPUTE_DATA_PATH = project_path_ / r"Data\Results\Forecasting\impute_data"
try_to_find_folder_path_otherwise_make_one(IMPUTE_DATA_PATH)


def transform_data(original_time_series: pd.DataFrame):
    assert set(original_time_series.columns) == {'wind speed', 'air density', 'wind direction'}, \
        "Can only impute 'wind speed', 'air density', 'wind direction' safely"

    transformer = QuantileTransformer(n_quantiles=10000, output_distribution='normal')
    transformer.fit(X=original_time_series.values)

    ans = transformer.transform(original_time_series)
    ans = pd.DataFrame(
        data=ans,
        columns=original_time_series.columns,
        index=original_time_series.index
    )
    return transformer, ans


def impute_one_wf_core(transformed_time_series):
    observed_time_series = tfp_sts.MaskedTimeSeries(
        time_series=transformed_time_series.values,
        is_missing=np.isnan(transformed_time_series.values)
    )
    # %% Build model using observed time series to set heuristic priors.
    linear_trend_model = tfp_sts.LocalLinearTrend(
        observed_time_series=observed_time_series)
    model = tfp_sts.Sum([linear_trend_model],
                        observed_time_series=observed_time_series)
    # %% Fit model to data
    parameter_samples, _ = tfp_sts.fit_with_hmc(model,
                                                observed_time_series,
                                                num_results=300)
    # # Forecast
    # forecast_dist = tfp_sts.forecast(
    #     model, observed_time_series, parameter_samples, num_steps_forecast=6)

    # %% Impute missing values
    observations_dist = tfp_sts.impute_missing_values(model, observed_time_series, parameter_samples=parameter_samples)

    observations_dist_mean_numpy = observations_dist.mean().numpy()
    imputed_transformed_time_series = copy.deepcopy(transformed_time_series)
    for j, col_name in enumerate(imputed_transformed_time_series.columns):
        mask = np.isnan(transformed_time_series.values[:, j])
        imputed_transformed_time_series.loc[mask, col_name] = observations_dist_mean_numpy[mask, j]
    return imputed_transformed_time_series


def impute_one_wf(wf_name: str):
    cluster_name = WF_TO_CLUSTER_MAPPER[wf_name]
    original_time_series = pd.read_csv(SHARED_DIR_PATH / fr"all/{cluster_name} cluster {wf_name} WF.csv",
                                       index_col='time stamp')
    original_time_series.index = pd.DatetimeIndex(original_time_series.index)
    original_time_series = original_time_series[['wind speed', 'wind direction', 'air density']]

    test_periods = load_pkl_file(SHARED_DIR_PATH / rf"{cluster_name} cluster test_periods.pkl")
    original_time_series = original_time_series[
        original_time_series.index < (test_periods[wf_name][0] + datetime.timedelta(days=7))
        ]

    transformer, transformed_time_series = transform_data(original_time_series)

    imputed_transformed_time_series = impute_one_wf_core(transformed_time_series)

    imputed_inverse_transformed_time_series = transformer.inverse_transform(imputed_transformed_time_series.values)
    imputed_inverse_transformed_time_series = pd.DataFrame(
        data=imputed_inverse_transformed_time_series,
        index=imputed_transformed_time_series.index,
        columns=imputed_transformed_time_series.columns
    )
    temp = imputed_inverse_transformed_time_series.interpolate(method='spline', order=5, limit=6,
                                                               limit_direction='both')
    for j, col_name in enumerate(temp.columns):
        _min = np.nanmin(imputed_inverse_transformed_time_series.loc[:, col_name].values)
        _max = np.nanmax(imputed_inverse_transformed_time_series.loc[:, col_name].values)
        temp.loc[temp.loc[:, col_name].values < _min, col_name] = _min
        temp.loc[temp.loc[:, col_name].values > _max, col_name] = _max

    temp.to_csv(IMPUTE_DATA_PATH / fr"{cluster_name} cluster {wf_name} WF imputed natural resources.csv")


def impute_main():
    for wf_name in AVAILABLE_WF_NAMES:
        if wf_name in {'Glunca', 'Jelinak', 'Katuni', 'Lukovac'}:
            continue
        print(f"NOW {wf_name}")

        impute_one_wf(wf_name)


if __name__ == '__main__':
    impute_main()
