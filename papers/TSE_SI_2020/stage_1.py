from Ploting.fast_plot_Func import *
from project_utils import *
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, CLUSTER_TO_WF_MAPPER, WF_TO_CLUSTER_MAPPER, \
    Croatia_WF_LOCATION_MAPPER
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

tfd = eval("tfp.distributions")
tfpl = eval("tfp.layers")
tfb = eval("tfp.bijectors")
tfp_util = eval("tfp.util")
tfp_math = eval("tfp.math")

tf.keras.backend.set_floatx('float32')

BATCH_SIZE = 25000
SHARED_DIR_PATH = project_path_ / r"Data\Raw_measurements\TSE_SI_2020_WF\shared_data"
IMPUTE_DATA_PATH = project_path_ / r"Data\Results\Forecasting\impute_data"

NN_MODEL_PATH = project_path_ / r"Data\Results\Forecasting\NN_model"
NN_MODEL_PREDICTION_PATH = project_path_ / r"Data\Results\Forecasting\NN_model_predictions"


# Prepare data for NN
class EveryThingDataSet(DeepLearningDataSet):
    def __init__(self, *args, data: pd.DataFrame, geo_loc: str, **kwargs):
        assert type(data) == pd.DataFrame

        self.geo_loc = geo_loc
        super().__init__(
            *args,
            original_data_set=data,

            quantile_transformed_col=('wind speed', 'air density', 'wind direction',),

            predictor_cols=('wind speed', 'air density', 'wind direction',),
            dependant_cols=('wind speed', 'air density', 'wind direction',),

            name=self.geo_loc + 'EveryThing' + '_training',
            transformation_args_folder_path=project_path_ / ''.join(
                ['Data/Results/Forecasting/NN_model_DataSet_transformation_args/',
                 self.geo_loc, '/EveryThing/transformation_args/']
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
                                                res_name: str):
    assert task in {"training", "test"}
    assert res_name in {'EveryThing', 'OPR', 'Copula'}

    file_path = SHARED_DIR_PATH / fr"all/{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster {geo_loc} WF.csv"
    data = pd.read_csv(file_path, index_col="time stamp")
    data.index = pd.DatetimeIndex(data.index)

    file_path = IMPUTE_DATA_PATH / fr"{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster {geo_loc}" \
                                   " WF imputed natural resources.csv"
    data_impute = pd.read_csv(file_path, index_col='time stamp')
    data_impute.index = pd.DatetimeIndex(data_impute.index)

    if res_name == 'EveryThing':
        data.drop(labels=['active power output', 'normally operating number'], axis=1, inplace=True)

        data = data[data_impute.columns]
        data.iloc[:data_impute.shape[0]] = data_impute.values
    elif res_name == 'OPR':
        data = data[[*data_impute.columns, 'active power output', 'normally operating number']]
        data.loc[:data_impute.index[-1], data_impute.columns] = data_impute.values
    else:
        to_delete_mask = np.isnan(data['wind speed'].values)
        data_impute.loc[to_delete_mask[:data_impute.shape[0]], 'wind speed'] = np.nan
        data = data[[*data_impute.columns, 'active power output', 'normally operating number']]
        data.loc[:data_impute.index[-1], data_impute.columns] = data_impute.values

    test_periods = load_pkl_file(SHARED_DIR_PATH / rf"{WF_TO_CLUSTER_MAPPER[geo_loc]} cluster test_periods.pkl")
    training_mask = data.index < (test_periods[geo_loc][0] + datetime.timedelta(days=7))
    test_mask = data.index >= test_periods[geo_loc][0]

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
        data_set = EveryThingDataSet(data=data, geo_loc=geo_loc)
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


def train_nn_model(geo_loc: str, res_name, *, continue_training: bool):
    print("★" * 79)
    print(f"Train {geo_loc} WF {res_name}")
    print("★" * 79)
    # Get data
    training_data_for_nn = get_natural_resources_or_opr_or_copula_data(geo_loc, "training", res_name=res_name)
    test_data_for_nn = get_natural_resources_or_opr_or_copula_data(geo_loc, "test", res_name=res_name)

    # Define NLL. NB, KL part and its weight/scale factor has been passed through divergence_fn or regularizer
    def nll(y_true, y_pred):
        return -y_pred.log_prob(y_true)

    # Build model
    model = get_nn_model(input_shape=training_data_for_nn.element_spec[0].shape[1:],
                         output_shape=training_data_for_nn.element_spec[1].shape[1:],
                         res_name=res_name,
                         category_number=NUMBER_OF_WT_MAPPER[geo_loc] + 1)
    model = model.build()

    # Define Callbacks
    try_to_find_folder_path_otherwise_make_one(NN_MODEL_PATH / f'{geo_loc}/{res_name}')

    class SaveCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 150 == 0:
                model.save_weights(NN_MODEL_PATH / f'{geo_loc}/{res_name}/epoch_{epoch}.h5')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000)

    # Compile model
    if res_name == 'EveryThing':
        metrics = ['mse']
    else:
        metrics = ['accuracy']

    model.compile(loss=nll, optimizer=tf.keras.optimizers.RMSprop(), metrics=metrics,
                  experimental_run_tf_function=False)
    model.summary()
    _debug_training_data_for_nn = list(training_data_for_nn.as_numpy_iterator())
    _debug_training_data_for_nn_x = _debug_training_data_for_nn[0][0][[0]]
    _debug_training_data_for_nn_y = _debug_training_data_for_nn[0][1][[0]]

    if continue_training:
        model.load_weights(NN_MODEL_PATH / f'{geo_loc}/{res_name}/to_continue.h5')

    model.fit(
        training_data_for_nn, verbose=1, epochs=100_005,
        validation_data=test_data_for_nn, validation_freq=50, validation_batch_size=BATCH_SIZE // 2,
        callbacks=[SaveCallback(), early_stopping]
    )
    model.save_weights(NN_MODEL_PATH / f'{geo_loc}/{res_name}/final.h5')
    return model


def test_nn_model(geo_loc: str, res_name: str, ensemble_size: int = 3000, *,
                  test_sample_index: Sequence[int] = None):
    # Get data
    test_data_set, test_data_windowed = get_natural_resources_or_opr_or_copula_data(geo_loc, "test", False,
                                                                                    res_name=res_name)

    # Build and get model
    tester = get_nn_model(input_shape=test_data_windowed[0].element_spec[0].shape[1:],
                          output_shape=test_data_windowed[0].element_spec[1].shape[1:],
                          res_name=res_name,
                          category_number=NUMBER_OF_WT_MAPPER[geo_loc] + 1)
    tester = tester.build()
    tester.load_weights(NN_MODEL_PATH / f'{geo_loc}/{res_name}/final.h5')

    # Select samples to test
    test_samples_x, test_samples_y = [], []
    gen = test_data_windowed[1]
    for i, sample in enumerate(gen(True)):
        if test_sample_index is None or i in test_sample_index:
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

    # Save
    save_path = NN_MODEL_PREDICTION_PATH / fr"{geo_loc}/{res_name}"
    try_to_find_folder_path_otherwise_make_one(save_path)
    save_pkl_file(save_path / "test_set_predictions.pkl",
                  {
                      "test_samples_y_inv": test_samples_y_inv,
                      "prediction_results_inv": prediction_results_inv
                  })

    temp = load_pkl_file(save_path / "test_set_predictions.pkl")
    for j in range(temp['test_samples_y_inv'].shape[-1]):
        ax = series(temp['test_samples_y_inv'][:, 0, j], color='red')
        ax = series(np.mean(temp['prediction_results_inv'], axis=0)[:, 0, j], ax=ax, color='royalblue')
        series(temp['prediction_results_inv'][:300, :, 0, j].T, color='grey', linewidth=0.5, ax=ax, alpha=0.1,
               zorder=-1)

        ax = series(temp['test_samples_y_inv'][:, 0, j], color='red')
        ax = series(np.mean(temp['prediction_results_inv'], axis=0)[:, 0, j], ax=ax, color='royalblue')
        ax = series(np.percentile(temp['prediction_results_inv'], 2.5, axis=0)[:, 0, j], ax=ax,
                    color='green', linestyle='--')
        ax = series(np.percentile(temp['prediction_results_inv'], 97.5, axis=0)[:, 0, j], ax=ax,
                    color='green', linestyle='--')

        # hist(temp['prediction_results_inv'][:, 0, 0, 0])


if __name__ == "__main__":
    # train_nn_model("Glunca", 'EveryThing', continue_training=False)
    # train_nn_model('Glunca', 'OPR', continue_training=False)

    # train_nn_model("Jelinak", 'EveryThing', continue_training=True)
    # train_nn_model('Jelinak', 'OPR', continue_training=True)

    # train_nn_model("Zelengrad", 'EveryThing', continue_training=False)
    train_nn_model("Zelengrad", 'OPR', continue_training=True)

    train_nn_model("Bruska", 'EveryThing', continue_training=False)
    train_nn_model("Bruska", 'OPR', continue_training=False)

    train_nn_model("Lukovac", 'EveryThing', continue_training=False)
    train_nn_model("Lukovac", 'OPR', continue_training=False)

    train_nn_model("Katuni", 'EveryThing', continue_training=False)
    train_nn_model("Katuni", 'OPR', continue_training=False)

    pass
