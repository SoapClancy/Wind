import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from Ploting.fast_plot_Func import *
from project_utils import *
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, CLUSTER_TO_WF_MAPPER, WF_TO_CLUSTER_MAPPER, \
    Croatia_WF_LOCATION_MAPPER, AVAILABLE_WF_NAMES
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
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from Regression_Analysis.DeepLearning_Class import BayesianConv1DBiLSTM
from prepare_datasets import load_raw_wt_from_txt_file_and_temperature_from_csv
from Ploting.fast_plot_Func import *
import re
from papers.TSE_SI_2020.utils import preds_continuous_var_plot, cal_continuous_var_error, \
    turn_preds_into_univariate_pdf_or_cdf_like
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Union, Sequence
import datetime
from locale import setlocale, LC_ALL
from BivariateAnalysis_Class import MethodOfBins
from matplotlib import cm, colors
from Ploting.adjust_Func import *
from scipy import stats
from UnivariateAnalysis_Class import ECDF, UnivariatePDFOrCDFLike
import json
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.io.matlab import savemat, loadmat

setlocale(LC_ALL, "en_US")
sns.set()

tfd = eval("tfp.distributions")
tfpl = eval("tfp.layers")
tfb = eval("tfp.bijectors")
tfp_util = eval("tfp.util")
tfp_math = eval("tfp.math")

tf.keras.backend.set_floatx('float32')

PRED_BY = "mean"
assert PRED_BY in {"mean", "median"}

BATCH_SIZE = 25000
SHARED_DIR_PATH = project_path_ / r"Data\Raw_measurements\TSE_SI_2020_WF\shared_data"
IMPUTE_INDIVIDUAL_DATA_PATH = project_path_ / r"Data\Results\Forecasting\impute_data\individual"
IMPUTE_CLUSTER_DATA_PATH = project_path_ / r"Data\Results\Forecasting\impute_data\cluster"
IMPUTE_ALL_DATA_PATH = project_path_ / r"Data\Results\Forecasting\impute_data\all"

NN_MODEL_PATH = project_path_ / r"Data\Results\Forecasting\NN_model"
NN_MODEL_PREDICTION_PATH = project_path_ / r"Data\Results\Forecasting\NN_model_predictions"


# Prepare data for NN
class EveryThingDataSet(DeepLearningDataSet):
    def __init__(self, *args, data: pd.DataFrame, geo_loc: str, use_corr_impute: str = '', **kwargs):
        assert type(data) == pd.DataFrame

        self.geo_loc = geo_loc
        if use_corr_impute == '':
            predictor_cols = ('wind speed', 'air density', 'wind direction',)
            dependant_cols = ('wind speed', 'air density', 'wind direction',)
            quantile_transformed_col = ('wind speed', 'air density', 'wind direction',)
        else:
            predictor_cols = list(data.columns)
            # N.B. WD is before AD, not good for Copula, but be transposed in test phase
            dependant_cols = [x for x in data.columns if geo_loc in x]
            quantile_transformed_col = tuple(list(data.columns))
        super().__init__(
            *args,
            original_data_set=data,

            quantile_transformed_col=quantile_transformed_col,

            predictor_cols=tuple(predictor_cols),
            dependant_cols=tuple(dependant_cols),

            name=self.geo_loc + 'EveryThing' + use_corr_impute + '_training',
            transformation_args_folder_path=project_path_ / ''.join(
                ['Data/Results/Forecasting/NN_model_DataSet_transformation_args/',
                 self.geo_loc, '/EveryThing', use_corr_impute, '/transformation_args/']
            ),
            **kwargs
        )


class OPRDataSet(DeepLearningDataSet):
    def __init__(self, *args, data: pd.DataFrame, geo_loc: str, **kwargs):
        assert type(data) == pd.DataFrame

        self.geo_loc = geo_loc
        super().__init__(
            *args,
            original_data_set=data,

            quantile_transformed_col=('wind speed', 'air density', 'wind direction', 'active power output'),
            one_hot_transformed_col=('normally operating number',),

            predictor_cols=('wind speed', 'air density', 'wind direction', 'active power output',
                            'normally operating number'),
            dependant_cols=('normally operating number',),

            name=self.geo_loc + 'OPR' + '_training',
            transformation_args_folder_path=project_path_ / ''.join(
                ['Data/Results/Forecasting/NN_model_DataSet_transformation_args/',
                 self.geo_loc, '/OPR/transformation_args/']
            ),
            **kwargs
        )


def get_natural_resources_or_opr_or_copula_data(geo_loc: str, task: str, only_return_for_nn: bool = True, *,
                                                use_corr_impute: str,
                                                res_name: str):
    assert task in {"training", "test"}
    assert res_name in {'EveryThing', 'OPR', 'Copula'}
    if use_corr_impute != '':
        assert use_corr_impute in {'_cluster_', '_all_'}

    file_path = SHARED_DIR_PATH / fr"all/{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster {geo_loc} WF.csv"
    data = pd.read_csv(file_path, index_col="time stamp")
    data.index = pd.DatetimeIndex(data.index)

    file_path = IMPUTE_INDIVIDUAL_DATA_PATH / fr"{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster {geo_loc}" \
                                              " WF imputed natural resources.csv"
    data_impute = pd.read_csv(file_path, index_col='time stamp')
    data_impute.index = pd.DatetimeIndex(data_impute.index)

    test_periods = load_pkl_file(SHARED_DIR_PATH / rf"{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster test_periods.pkl")
    training_mask = data.index < (test_periods[geo_loc][0] + datetime.timedelta(days=7))
    test_mask = data.index >= test_periods[geo_loc][0]

    if res_name == 'EveryThing':
        data.drop(labels=['active power output', 'normally operating number'], axis=1, inplace=True)
        data = data[data_impute.columns]
        if use_corr_impute == '':
            data.iloc[:data_impute.shape[0]] = data_impute.values
        else:
            if use_corr_impute == '_cluster_':
                file_path = IMPUTE_CLUSTER_DATA_PATH / fr"{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster" \
                                                       " imputed natural resources.csv"
            else:
                file_path = IMPUTE_ALL_DATA_PATH / "all imputed natural resources.csv"

            all_impute_data = pd.read_csv(
                file_path,
                index_col=0,
                header=0,
            )

            all_impute_data.index = pd.DatetimeIndex(all_impute_data.index)

            # get extra cols
            extra_col_names = []
            for i, col_name in enumerate(all_impute_data.columns):
                if geo_loc not in col_name:
                    extra_col_names.append(col_name)

            data.columns = [geo_loc + '_' + x for x in data.columns]

            data.loc[training_mask] = all_impute_data.loc[data.index[training_mask], data.columns]
            data = pd.merge(data, all_impute_data.loc[:, extra_col_names],
                            left_index=True, right_index=True, how='left')

    elif res_name == 'OPR':
        data = data[[*data_impute.columns, 'active power output', 'normally operating number']]
        data.loc[:data_impute.index[-1], data_impute.columns] = data_impute.values
    else:
        to_delete_mask = np.isnan(data['wind speed'].values)
        data_impute.loc[to_delete_mask[:data_impute.shape[0]], 'wind speed'] = np.nan
        data = data[[*data_impute.columns, 'active power output', 'normally operating number']]
        data.loc[:data_impute.index[-1], data_impute.columns] = data_impute.values

    # data[training_mask].to_csv(
    #     "./training/" +
    #     f"{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster {geo_loc} WF imputed natural resources {use_corr_impute}.csv"
    # )
    # data[test_mask].to_csv(
    #     f"./test/{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster {geo_loc} WF imputed natural resources {use_corr_impute}.csv"
    # )

    mask = training_mask if task == "training" else test_mask
    # scatter(data.iloc[:, 0], data.iloc[:, 3], title='all', x_lim=(-0.5, 29.5), y_lim=(-0.05, None))
    # scatter(data[training_mask].iloc[:, 0], data[training_mask].iloc[:, 3], title='training',
    #         x_lim=(-0.5, 29.5), y_lim=(-0.05, None))
    # scatter(data[test_mask].iloc[:, 0], data[test_mask].iloc[:, 3], title='test',
    #         x_lim=(-0.5, 29.5), y_lim=(-0.05, None))
    # scatter(data[np.bitwise_and(test_mask,
    #                             data['normally operating number'] == NUMBER_OF_WT_MAPPER[geo_loc])].iloc[:, 0],
    #         data[np.bitwise_and(test_mask,
    #                             data['normally operating number'] == NUMBER_OF_WT_MAPPER[geo_loc])].iloc[:, 3],
    #         title='test full',
    #         x_lim=(-0.5, 29.5), y_lim=(-0.05, None))
    # series(data[test_mask]['normally operating number'])

    data = data[mask]

    if res_name == 'Copula':
        wf_obj = WF(
            data=data,
            obj_name=f'{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster {geo_loc} WF',
            rated_active_power_output=WF_RATED_POUT_MAPPER[geo_loc],
            predictor_names=('wind speed', 'wind direction', 'air density'),
            dependant_names=('active power output',),
            number_of_wind_turbine=NUMBER_OF_WT_MAPPER[geo_loc],
        )
        return wf_obj

    if res_name == 'EveryThing':
        data_set = EveryThingDataSet(data=data, geo_loc=geo_loc, use_corr_impute=use_corr_impute)
    else:
        data_set = OPRDataSet(data=data, geo_loc=geo_loc)

    data_set_windowed = data_set.windowed_dataset(
        x_window_length=datetime.timedelta(days=7),
        y_window_length=datetime.timedelta(hours=1),
        x_y_start_index_diff=datetime.timedelta(days=7),
        window_shift=datetime.timedelta(hours=1),
        batch_size=BATCH_SIZE
    )
    if only_return_for_nn:
        return data_set_windowed[0]
    else:
        return data_set, data_set_windowed


class TSE2020SIBayesianConv1DBiLSTM(BayesianConv1DBiLSTM):
    def __init__(self, input_shape, output_shape, *, dense_hypers_units):
        super().__init__(
            input_shape=input_shape, output_shape=output_shape, batch_size=BATCH_SIZE,
            conv1d_hypers_filters=9, conv1d_hypers_padding="same", conv1d_hypers_kernel_size=3,
            maxpool1d_hypers_padding="valid", maxpool1d_hypers_pool_size=2,
            bilstm_hypers_units=42,
            use_encoder_decoder=False,
            dense_hypers_units=dense_hypers_units
        )

    def get_distribution_layer(self, dtype=tf.float32):
        pass


class NaturalResourcesBayesianConv1DBiLSTM(TSE2020SIBayesianConv1DBiLSTM):
    def __init__(self, input_shape, output_shape):
        super(NaturalResourcesBayesianConv1DBiLSTM, self).__init__(
            input_shape,
            output_shape,
            dense_hypers_units=tfpl.MixtureSameFamily.params_size(
                num_components=3,
                component_params_size=tfpl.MultivariateNormalTriL.params_size(3)
            )
        )

    def get_distribution_layer(self, dtype=tf.float32):
        dist_layer = tfpl.MixtureSameFamily(num_components=3,
                                            component_layer=tfpl.MultivariateNormalTriL(3),
                                            convert_to_tensor_fn=tfd.Distribution.sample)
        return dist_layer


class OPRBayesianConv1DBiLSTM(TSE2020SIBayesianConv1DBiLSTM):
    def __init__(self, input_shape, output_shape, category_number):
        self.category_number = category_number
        super(OPRBayesianConv1DBiLSTM, self).__init__(
            input_shape,
            output_shape,
            dense_hypers_units=tfpl.OneHotCategorical.params_size(self.category_number)
        )

    def get_distribution_layer(self, dtype=tf.float32):
        dist_layer = tfpl.OneHotCategorical(self.category_number)
        return dist_layer


def get_nn_model(input_shape, output_shape, *, res_name, category_number: int = None):
    if res_name == 'EveryThing':
        model = NaturalResourcesBayesianConv1DBiLSTM(input_shape, output_shape)
    else:
        assert category_number is not None
        model = OPRBayesianConv1DBiLSTM(input_shape, output_shape, category_number)
    return model


def train_nn_model(geo_loc: str, res_name, *, use_corr_impute: str, continue_training: bool):
    print("★" * 79)
    print(f"Train {geo_loc} WF {res_name}{use_corr_impute}")
    print("★" * 79)
    # Get data
    training_data_for_nn = get_natural_resources_or_opr_or_copula_data(geo_loc, "training",
                                                                       res_name=res_name,
                                                                       use_corr_impute=use_corr_impute)
    test_data_for_nn = get_natural_resources_or_opr_or_copula_data(geo_loc, "test",
                                                                   res_name=res_name,
                                                                   use_corr_impute=use_corr_impute)

    # Define NLL. NB, KL part and its weight/scale factor has been passed through divergence_fn or regularizer
    def nll(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    # Build model
    model = get_nn_model(input_shape=training_data_for_nn.element_spec[0].shape[1:],
                         output_shape=training_data_for_nn.element_spec[1].shape[1:],
                         res_name=res_name,
                         category_number=NUMBER_OF_WT_MAPPER[geo_loc] + 1)
    model = model.build()

    try_to_find_folder_path_otherwise_make_one(NN_MODEL_PATH / f'{geo_loc}/{res_name}{use_corr_impute}')

    # Define Callbacks
    class SaveCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 150 == 0:
                model.save_weights(NN_MODEL_PATH / f'{geo_loc}/{res_name}{use_corr_impute}/epoch_{epoch}.h5')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50000)

    # Compile model
    if res_name == 'EveryThing':
        metrics = ['mse']
    else:
        metrics = ['accuracy']

    model.compile(loss=nll, optimizer=tf.keras.optimizers.Adam(0.00001), metrics=metrics,
                  experimental_run_tf_function=False)
    model.summary()
    _debug_training_data_for_nn = list(training_data_for_nn.as_numpy_iterator())
    _debug_training_data_for_nn_x = _debug_training_data_for_nn[0][0][[0]]
    _debug_training_data_for_nn_y = _debug_training_data_for_nn[0][1][[0]]

    if continue_training:
        model.load_weights(NN_MODEL_PATH / f'{geo_loc}/{res_name}{use_corr_impute}/to_continue.h5')

    model.fit(
        training_data_for_nn, verbose=1, epochs=100_005,
        validation_data=test_data_for_nn, validation_freq=50, validation_batch_size=BATCH_SIZE // 2,
        callbacks=[SaveCallback(), early_stopping]
    )
    model.save_weights(NN_MODEL_PATH / f'{geo_loc}/{res_name}{use_corr_impute}/final.h5')
    return model


def test_nn_model(geo_loc: str, res_name: str, ensemble_size: int = 3000, *,
                  use_corr_impute: str, use_training_set: bool = False):
    if use_training_set:
        assert ensemble_size == 1
    # Get data
    test_data_set, test_data_windowed = get_natural_resources_or_opr_or_copula_data(
        geo_loc,
        "training" if use_training_set else "test",
        False,
        res_name=res_name,
        use_corr_impute=use_corr_impute
    )

    # Build and get model
    tester = get_nn_model(input_shape=test_data_windowed[0].element_spec[0].shape[1:],
                          output_shape=test_data_windowed[0].element_spec[1].shape[1:],
                          res_name=res_name,
                          category_number=NUMBER_OF_WT_MAPPER[geo_loc] + 1)
    tester = tester.build()
    tester.load_weights(NN_MODEL_PATH / f'{geo_loc}/{res_name}{use_corr_impute}/final.h5')

    # Select samples to test
    test_samples_x, test_samples_y = [], []
    gen = test_data_windowed[1]
    for i, sample in enumerate(gen(True)):
        test_samples_x.append(sample[0].numpy())
        test_samples_y.append(sample[1].numpy())
    test_samples_x = np.array(test_samples_x)
    test_samples_y = np.array(test_samples_y)

    # Formal test
    prediction_results = np.full((ensemble_size, *test_samples_y.shape), np.nan)
    for i in range(ensemble_size):
        prediction_results[i] = tester(test_samples_x).sample().numpy()

    # Inverse transform
    if res_name == 'EveryThing':
        # N.B. WD is before AD, not good for Copula, but be transposed in test phase (see EveryThingDataSet)
        if use_corr_impute != '':
            inverse_transform_names = [(f'{geo_loc}_wind speed', 'quantile'),
                                       (f'{geo_loc}_wind direction', 'quantile'),
                                       (f'{geo_loc}_air density', 'quantile')]
        else:
            inverse_transform_names = [('wind speed', 'quantile'),
                                       ('air density', 'quantile'),
                                       ('wind direction', 'quantile')]
    else:
        inverse_transform_names = [('normally operating number', 'one_hot')
                                   for _ in range(NUMBER_OF_WT_MAPPER[geo_loc] + 1)]

    test_samples_y_inv = test_data_set.inverse_transform(test_samples_y, inverse_transform_names)
    prediction_results_inv = np.full((ensemble_size, *test_samples_y_inv.shape), np.nan)
    for i in range(prediction_results_inv.shape[0]):
        prediction_results_inv[i] = test_data_set.inverse_transform(prediction_results[i], inverse_transform_names)

    if res_name == 'EveryThing':
        # N.B. WD is before AD, not good for Copula, but be transposed in test phase (see EveryThingDataSet)
        if use_corr_impute != '':
            temp = copy.deepcopy(test_samples_y_inv[:, :, 1])
            test_samples_y_inv[:, :, 1] = copy.deepcopy(test_samples_y_inv[:, :, 2])
            test_samples_y_inv[:, :, 2] = temp

            temp = copy.deepcopy(prediction_results_inv[:, :, :, 1])
            prediction_results_inv[:, :, :, 1] = copy.deepcopy(prediction_results_inv[:, :, :, 2])
            prediction_results_inv[:, :, :, 2] = temp

    # Save
    save_path = NN_MODEL_PREDICTION_PATH / fr"{geo_loc}/{res_name}{use_corr_impute}"
    try_to_find_folder_path_otherwise_make_one(save_path)
    for j in range(2):
        error = test_samples_y_inv[:, 0, j] - prediction_results_inv[0, :, 0, j]
        if j == 0:
            savemat(save_path / "training_set_ws_err.mat", {"error": error})
        else:
            savemat(save_path / "training_set_ad_err.mat", {"error": error})

    save_pkl_file(save_path / "training_set_predictions.pkl" if use_training_set else "test_set_predictions.pkl",
                  {
                      "test_samples_y_inv": test_samples_y_inv,
                      "prediction_results_inv": prediction_results_inv
                  })

    temp = load_pkl_file(save_path / "test_set_predictions.pkl")
    for j in range(temp['test_samples_y_inv'].shape[-1]):
        ax = series(temp['test_samples_y_inv'][:, 0, j], color='red')
        if PRED_BY == 'mean':
            ax = series(np.mean(temp['prediction_results_inv'], axis=0)[:, 0, j], ax=ax, color='royalblue')
        else:
            ax = series(np.median(temp['prediction_results_inv'], axis=0)[:, 0, j], ax=ax, color='royalblue')
        series(temp['prediction_results_inv'][:300, :, 0, j].T, color='grey', linewidth=0.5, ax=ax, alpha=0.1,
               zorder=-1)

        ax = series(temp['test_samples_y_inv'][:, 0, j], color='red')
        ax = series(np.percentile(temp['prediction_results_inv'], 2.5, axis=0)[:, 0, j], ax=ax,
                    color='green', linestyle='--')
        series(np.percentile(temp['prediction_results_inv'], 97.5, axis=0)[:, 0, j], ax=ax,
               color='green', linestyle='--')

        # hist(temp['prediction_results_inv'][:, 0, 0, 0])


def get_natural_resources_results(wf_name: str, use_corr_impute: str):
    pred_natural_resources = load_pkl_file(
        NN_MODEL_PREDICTION_PATH / fr"{wf_name}/EveryThing{use_corr_impute}/test_set_predictions.pkl"
    )

    return pred_natural_resources


def get_opr_results(wf_name: str):
    pred_opr = load_pkl_file(
        NN_MODEL_PREDICTION_PATH / fr"{wf_name}/OPR/test_set_predictions.pkl"
    )
    return pred_opr


def plot_natural_resources_results(wf_name: str, use_corr_impute: str):
    pred_natural_resources = get_natural_resources_results(wf_name, use_corr_impute)
    actual = pred_natural_resources['test_samples_y_inv']
    preds = pred_natural_resources['prediction_results_inv']

    preds_continuous_var_plot(wf_name=wf_name,
                              preds_samples=preds[:, :, 0, 0].T,
                              target_pout=actual[:, 0, 0],
                              name='WS')

    preds_continuous_var_plot(wf_name=wf_name,
                              preds_samples=preds[:, :, 0, 1].T,
                              target_pout=actual[:, 0, 1],
                              name='AD')
    preds_continuous_var_plot(wf_name=wf_name,
                              preds_samples=preds[:, :, 0, 2].T,
                              target_pout=actual[:, 0, 2],
                              name='WD')


def plot_opr_results(wf_name: str):
    pred_opr = get_opr_results(wf_name)
    actual = pred_opr['test_samples_y_inv']
    preds = pred_opr['prediction_results_inv']
    z = np.full((NUMBER_OF_WT_MAPPER[wf_name] + 1, actual.shape[0],), 0, dtype=float)
    cmap = cm.get_cmap('binary')
    for i in range(actual.shape[0]):
        temp = np.histogram(preds[:, i, 0, 0], np.arange(-0.5, NUMBER_OF_WT_MAPPER[wf_name] + 1))
        z[:, i] = temp[0] / np.sum(temp[0])
    z = (np.round(z, 1) * 10).astype(int)
    x = np.arange(-0.5, actual.shape[0], 1)
    y = np.arange(-0.5, NUMBER_OF_WT_MAPPER[wf_name] + 1)
    plt.figure(figsize=(6, 5 * 0.551), constrained_layout=True)
    ax = plt.gca()
    norm = colors.Normalize(vmin=- 0.1, vmax=10)
    for i in range(9, 0, -1):
        ax.fill_between(np.arange(1000, 1010),
                        np.arange(1000, 1010),
                        np.arange(1010, 1020),
                        facecolor=cmap(norm(i)),
                        edgecolor='none',
                        label=f"{i * 10}%")
    ax.pcolormesh(x, y, z,
                  cmap=cmap,
                  norm=norm,
                  edgecolor='none',
                  zorder=-1),
    plt.xlabel('Time [Hour]', fontsize=10)
    plt.xlim(-1, 169)
    plt.xticks(np.arange(0, 168 + 1, 24), np.arange(0, 168 + 1, 24), fontsize=10)
    plt.ylabel(f'{wf_name} WF Operating Regime', fontsize=10)
    plt.ylim(-0.5, NUMBER_OF_WT_MAPPER[wf_name] + 0.5)
    plt.yticks(np.arange(0, NUMBER_OF_WT_MAPPER[wf_name] + 1, 4),
               np.arange(0, NUMBER_OF_WT_MAPPER[wf_name] + 1, 4), fontsize=10)
    step(np.arange(0, 168), actual.flatten(), color='red', ax=ax,
         linestyle='-.', linewidth=1.2, alpha=0.95, label='Actual')
    step(np.arange(0, 168), stats.mode(np.squeeze(preds)).mode.flatten(), ax=ax,
         color='royalblue', linewidth=1.2, alpha=0.95, label='Pred.')
    plt.grid(True, color='gold', alpha=0.25)
    ax = adjust_legend_in_ax(ax, protocol='Outside center right')


def cal_natural_resources_errors(wf_name: str):
    file_path = project_path_ / fr"Data\Results\Forecasting\errors\stage1"
    try_to_find_folder_path_otherwise_make_one(file_path)

    use_corr_impute = ("", "_cluster_")

    ans_dict = {"own": dict(),
                "cluster": dict()}

    cols = ['WS', 'AD', 'WD_cos', 'WD_sin']

    for i, now_use_corr_impute in enumerate(use_corr_impute):

        pred_natural_resources = get_natural_resources_results(wf_name, now_use_corr_impute)
        actual = pred_natural_resources['test_samples_y_inv']
        preds = pred_natural_resources['prediction_results_inv']

        for j, now_col in enumerate(cols):
            if now_col == 'WD_cos':
                dist_objs = [UnivariatePDFOrCDFLike.init_from_samples_by_ecdf(x)
                             for x in np.cos(np.deg2rad(preds[:, :, 0, 2].T))]
                target = np.cos(np.deg2rad(actual[:, 0, 2]))
                name = 'WD'
            elif now_col == 'WD_sin':
                dist_objs = [UnivariatePDFOrCDFLike.init_from_samples_by_ecdf(x)
                             for x in np.sin(np.deg2rad(preds[:, :, 0, 2].T))]
                target = np.sin(np.deg2rad(actual[:, 0, 2]))
                name = 'WD'
            else:
                dist_objs = [UnivariatePDFOrCDFLike.init_from_samples_by_ecdf(x) for x in preds[:, :, 0, j].T]
                target = actual[:, 0, j]
                name = now_col

            temp = cal_continuous_var_error(target=target,
                                            model_output=dist_objs,
                                            name=name)
            if now_use_corr_impute == '':
                ans_dict["own"][now_col] = temp
            else:
                ans_dict["cluster"][now_col] = temp

    for val in ans_dict.values():
        val['WD'] = {}
        for ele in ('mae', 'rmse', 'pinball_loss', 'crps'):
            val['WD'][ele] = (val['WD_cos'][ele] + val['WD_sin'][ele]) / 2

    def recursion_round(node):
        for node_key, node_val in node.items():
            if isinstance(node_val, dict):
                recursion_round(node_val)
            else:
                node[node_key] = f"{np.round(node_val, 3):.3f}"

    ans_dict_round = copy.deepcopy(ans_dict)
    recursion_round(ans_dict_round)
    with open(file_path / f"{wf_name}_natural_resources.json", 'w') as json_file:
        json.dump(ans_dict_round, json_file)

    df = pd.DataFrame(columns=['mae', 'rmse', 'pinball_loss', 'crps'],
                      index=pd.MultiIndex.from_tuples([('own', 'WS'),
                                                       ('cluster', 'WS'),
                                                       ('own', 'AD'),
                                                       ('cluster', 'AD'),
                                                       ('own', 'WD'),
                                                       ('cluster', 'WD')]))
    for i in ('own', 'cluster'):
        for j in ('WS', 'AD', 'WD'):
            for k in ['mae', 'rmse', 'pinball_loss', 'crps']:
                df.loc[(i, j), k] = ans_dict[i][j][k]
    df.to_csv(file_path / f"{wf_name}_natural_resources.csv")


def cal_opr_errors(wf_name: str):
    file_path = project_path_ / fr"Data\Results\Forecasting\errors\stage1"
    try_to_find_folder_path_otherwise_make_one(file_path)

    pred_natural_resources = get_opr_results(wf_name)
    actual = pred_natural_resources['test_samples_y_inv']
    preds = pred_natural_resources['prediction_results_inv']
    z = np.full((NUMBER_OF_WT_MAPPER[wf_name] + 1, actual.shape[0],), 0, dtype=float)
    for i in range(actual.shape[0]):
        z[:, i] = np.histogram(preds[:, i, 0, 0], np.arange(-0.5, NUMBER_OF_WT_MAPPER[wf_name] + 1))[0] / preds.shape[0]

    error = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    err_val = error(list(actual.flatten().astype(int)),
                    list(z.T),
                    ).numpy()

    acc = 0
    for i in range(actual.shape[0]):
        now_true = int(actual[i, 0, 0])
        now_pred = preds[:, i, 0, 0].astype(int)
        # acc += np.sum(now_pred + 1 == now_true)
        acc += np.sum(np.any([now_pred + x == now_true for x in range(-4, 5, 1)], axis=0))

    return np.mean(err_val), acc / (preds.shape[0] * preds.shape[1])


def cal_corr():
    ans = pd.DataFrame()
    for name in ['Bruska', 'Jelinak', 'Lukovac']:
        data, _ = get_natural_resources_or_opr_or_copula_data(name, "training", False,
                                                              res_name='EveryThing',
                                                              use_corr_impute='_cluster_')
        cluster_df = pd.DataFrame()
        for i in range(3):
            arr_a = data.data.iloc[:, i].values
            arr_b = data.data.iloc[:, i + 3].values
            if i != 1:
                corr = spearmanr(arr_a, arr_b)
                coef = corr[0]
                p_val = corr[1]
                now_ans = pd.DataFrame(
                    data=[coef, p_val],
                    index=[WF_TO_CLUSTER_MAPPER[name]],
                    columns=[data.data.columns[i] + 'coefficient', data.data.columns[i] + 'p-value']
                )
            else:
                corr = spearmanr(np.cos(np.deg2rad(arr_a)), np.cos(np.deg2rad(arr_b)))
                coef = corr[0]
                p_val = corr[1]
                now_ans = pd.DataFrame(
                    data=[coef, p_val],
                    index=[WF_TO_CLUSTER_MAPPER[name]],
                    columns=['wind direction cos']
                )
                corr = spearmanr(np.sin(np.deg2rad(arr_a)), np.sin(np.deg2rad(arr_b)))
                coef = corr[0]
                p_val = corr[1]
                now_ans = pd.DataFrame(
                    data=[coef, p_val],
                    index=[WF_TO_CLUSTER_MAPPER[name]],
                    columns=['wind direction sin']
                )


if __name__ == "__main__":
    # train_nn_model("Glunca", 'EveryThing', continue_training=True, use_corr_impute='')
    # train_nn_model('Glunca', 'OPR', continue_training=False)

    # train_nn_model("Jelinak", 'EveryThing', continue_training=True)
    # train_nn_model('Jelinak', 'OPR', continue_training=True, use_corr_impute='')

    # train_nn_model("Zelengrad", 'EveryThing', continue_training=False)
    # train_nn_model("Zelengrad", 'OPR', continue_training=True)

    # train_nn_model("Bruska", 'EveryThing', continue_training=False)
    # train_nn_model("Bruska", 'OPR', continue_training=False)

    # train_nn_model("Lukovac", 'EveryThing', continue_training=False, use_corr_impute='')
    # train_nn_model("Lukovac", 'OPR', continue_training=True)

    # train_nn_model("Katuni", 'EveryThing', continue_training=False, use_corr_impute='')
    # train_nn_model("Katuni", 'OPR', continue_training=True, use_corr_impute='')

    pass
    # train_nn_model("Glunca", 'EveryThing', continue_training=True, use_corr_impute='_cluster_')
    # train_nn_model("Jelinak", 'EveryThing', continue_training=True, use_corr_impute='_cluster_')

    # train_nn_model("Zelengrad", 'EveryThing', continue_training=True, use_corr_impute='_cluster_')
    # train_nn_model("Bruska", 'EveryThing', continue_training=True, use_corr_impute='_cluster_')

    # train_nn_model("Lukovac", 'EveryThing', continue_training=True, use_corr_impute='_cluster_')
    # train_nn_model("Katuni", 'EveryThing', continue_training=True, use_corr_impute='_cluster_')

    pass
    # get_natural_resources_or_opr_or_copula_data('Katuni', 'training', use_corr_impute='', res_name='EveryThing')
    test_nn_model('Katuni', 'EveryThing', ensemble_size=1, use_corr_impute='_cluster_', use_training_set=True)
    # for final_name in AVAILABLE_WF_NAMES:
    #     # plot_natural_resources_results(final_name, '')
    #     # plot_natural_resources_results(final_name, '_cluster_')
    #     # plot_opr_results(final_name)
    #     cal_natural_resources_errors(final_name)

    # plot_natural_resources_results('Bruska', '')
    # plot_natural_resources_results('Bruska', '_cluster_')
    # cal_natural_resources_errors('Bruska')
    # print(cal_opr_errors('Bruska'))
    # print(cal_opr_errors('Jelinak'))
    # cal_natural_resources_errors('Lukovac')
    # plot_opr_results('Jelinak')
    # print(cal_opr_errors('Jelinak'))
    # for now_wf in AVAILABLE_WF_NAMES:
    #     plot_opr_results(now_wf)
    #     print(f"{now_wf} cross_entropy = {cal_opr_errors(now_wf):.3f}")

    # cal_corr()
