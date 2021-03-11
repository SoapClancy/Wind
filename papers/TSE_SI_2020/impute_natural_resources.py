import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
IMPUTE_INDIVIDUAL_DATA_PATH = project_path_ / r"Data\Results\Forecasting\impute_data\individual"
IMPUTE_CLUSTER_DATA_PATH = project_path_ / r"Data\Results\Forecasting\impute_data\cluster"
IMPUTE_ALL_DATA_PATH = project_path_ / r"Data\Results\Forecasting\impute_data\all"
try_to_find_folder_path_otherwise_make_one(IMPUTE_INDIVIDUAL_DATA_PATH)
try_to_find_folder_path_otherwise_make_one(IMPUTE_CLUSTER_DATA_PATH)
try_to_find_folder_path_otherwise_make_one(IMPUTE_ALL_DATA_PATH)


def transform_data(original_time_series: pd.DataFrame):
    # assert set(original_time_series.columns) == {'wind speed', 'air density', 'wind direction'}, \
    #     "Can only impute 'wind speed', 'air density', 'wind direction' safely"

    transformer = QuantileTransformer(n_quantiles=10000, output_distribution='normal')
    transformer.fit(X=original_time_series.values)

    ans = transformer.transform(original_time_series)
    ans = pd.DataFrame(
        data=ans,
        columns=original_time_series.columns,
        index=original_time_series.index
    )
    return transformer, ans


def impute_core(transformed_time_series):
    observed_time_series = tfp_sts.MaskedTimeSeries(
        time_series=transformed_time_series.values.astype(np.float32),
        is_missing=np.isnan(transformed_time_series.values.astype(np.float32))
    )
    # %% Build model using observed time series to set heuristic priors.
    linear_trend_model = tfp_sts.LocalLinearTrend(
        observed_time_series=observed_time_series)
    model = tfp_sts.Sum([linear_trend_model],
                        observed_time_series=observed_time_series)
    # %% Fit model to data
    parameter_samples, _ = tfp_sts.fit_with_hmc(model,
                                                observed_time_series,
                                                num_results=300)  # 300

    # %% Impute missing values
    # save_pkl_file(Path('./parameter_samples.pkl'), parameter_samples)
    # parameter_samples = load_pkl_file(Path('./parameter_samples.pkl'))
    # parameter_samples = [tf.cast(x[:200], tf.float64) for x in parameter_samples]
    # parameter_samples = [tf.cast(x[:180], tf.float32) for x in parameter_samples]
    # for i in range(3):
    #     parameter_samples[i] = tf.cast(parameter_samples[i], tf.float32)

    observations_dist = tfp_sts.impute_missing_values(model, observed_time_series, parameter_samples=parameter_samples)

    observations_dist_mean_numpy = observations_dist.mean().numpy()
    imputed_transformed_time_series = copy.deepcopy(transformed_time_series)
    for j in range(len(imputed_transformed_time_series.columns)):
        mask = np.isnan(transformed_time_series.values[:, j])
        imputed_transformed_time_series.iloc[mask, j] = observations_dist_mean_numpy[mask, j]
    return imputed_transformed_time_series


def impute_core_helper(transformer, transformed_time_series):
    imputed_transformed_time_series = impute_core(transformed_time_series)

    imputed_inverse_transformed_time_series = transformer.inverse_transform(imputed_transformed_time_series.values)
    imputed_inverse_transformed_time_series = pd.DataFrame(
        data=imputed_inverse_transformed_time_series,
        index=imputed_transformed_time_series.index,
        columns=imputed_transformed_time_series.columns
    )
    temp = imputed_inverse_transformed_time_series.interpolate(method='spline', order=5, limit=6,
                                                               limit_direction='both')
    for j in range(len(temp.columns)):
        _min = np.nanmin(imputed_inverse_transformed_time_series.iloc[:, j].values)
        _max = np.nanmax(imputed_inverse_transformed_time_series.iloc[:, j].values)
        temp.iloc[temp.iloc[:, j].values < _min, j] = _min
        temp.iloc[temp.iloc[:, j].values > _max, j] = _max

    return temp


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
    temp = impute_core_helper(transformer, transformed_time_series)

    temp.to_csv(IMPUTE_INDIVIDUAL_DATA_PATH / fr"{cluster_name} cluster {wf_name} WF imputed natural resources.csv")


def impute_individual_main():
    for wf_name in AVAILABLE_WF_NAMES:
        print(f"NOW {wf_name}")

        impute_one_wf(wf_name)


def impute_one_cluster(cluster_name: str):
    print(f"Now {cluster_name}")
    # merge
    merged_df = pd.DataFrame()
    original_df_index = dict()
    for wf_name in CLUSTER_TO_WF_MAPPER[cluster_name]:
        original_time_series = pd.read_csv(
            SHARED_DIR_PATH.parent / fr"pre_final/{cluster_name} cluster {wf_name} WF_pre_FINAL.csv",
            index_col=0)
        original_time_series.index = pd.DatetimeIndex(original_time_series.index)
        reading = original_time_series[['wind speed', 'wind direction', 'air density']]
        reading.index.name = 'time stamp'

        # temp = IMPUTE_INDIVIDUAL_DATA_PATH / fr"{cluster_name} cluster {wf_name} WF imputed natural resources.csv"
        # reading = pd.read_csv(temp, index_col='time stamp')
        # reading.index = pd.DatetimeIndex(reading.index)

        reading.columns = pd.MultiIndex.from_product([[wf_name], reading.columns],
                                                     names=['wf_name', 'natural_resources_name'])
        original_df_index.setdefault(wf_name, reading.index.values)
        merged_df = pd.merge(merged_df, reading, left_index=True, right_index=True, how='outer')

    merged_df = merged_df.astype('float32')

    transformer, transformed_time_series = transform_data(merged_df)

    temp = impute_core_helper(transformer, transformed_time_series)
    temp.columns = pd.MultiIndex.from_tuples(temp)

    for wf_name in CLUSTER_TO_WF_MAPPER[cluster_name]:
        temp.loc[original_df_index[wf_name]].to_csv(
            IMPUTE_CLUSTER_DATA_PATH / fr"{cluster_name} cluster {wf_name} WF imputed natural resources.csv"
        )


def impute_cluster_main():
    for cluster_name in ['Sibenik', 'Split', 'Zadar']:
        # if cluster_name != 'Sibenik':
        #     continue
        try:
            impute_one_cluster(cluster_name)
        except:
            pass


def impute_all_main():
    # merge
    merged_df = pd.DataFrame()
    original_df_index = dict()
    for wf_name in AVAILABLE_WF_NAMES:
        original_time_series = pd.read_csv(
            SHARED_DIR_PATH.parent / fr"pre_final/{WF_TO_CLUSTER_MAPPER[wf_name]} cluster {wf_name} WF_pre_FINAL.csv",
            index_col=0)
        original_time_series.index = pd.DatetimeIndex(original_time_series.index)
        reading = original_time_series[['wind speed', 'wind direction', 'air density']]
        reading.index.name = 'time stamp'

        # temp = IMPUTE_INDIVIDUAL_DATA_PATH / (fr"{WF_TO_CLUSTER_MAPPER[wf_name]} cluster {wf_name}" +
        #                                      " WF imputed natural resources.csv")
        # reading = pd.read_csv(temp, index_col='time stamp')
        # reading.index = pd.to_datetime(reading.index)

        reading.columns = pd.MultiIndex.from_product([[wf_name], reading.columns],
                                                     names=['wf_name', 'natural_resources_name'])
        original_df_index.setdefault(wf_name, reading.index.values)
        merged_df = pd.merge(merged_df, reading, left_index=True, right_index=True, how='outer')

    merged_df = merged_df.astype('float')

    transformer, transformed_time_series = transform_data(merged_df)

    temp = impute_core_helper(transformer, transformed_time_series)
    temp.columns = pd.MultiIndex.from_tuples(temp)

    for wf_name in AVAILABLE_WF_NAMES:
        temp.loc[original_df_index[wf_name]].to_csv(
            IMPUTE_ALL_DATA_PATH / (fr'{WF_TO_CLUSTER_MAPPER[wf_name]} cluster {wf_name} WF ' +
                                    'imputed natural resources.csv')
        )


if __name__ == '__main__':
    # impute_individual_main()
    # impute_cluster_main()
    # impute_all_main()

    # impute_one_cluster('Zadar')
    impute_all_main()
    pass
