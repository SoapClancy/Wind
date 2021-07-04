import numpy as np
import pandas as pd
import math
from Ploting.fast_plot_Func import *
from project_utils import *
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from prepare_gefcom_datasets import GEFCom2014SingleZoneDataSet, get_data_for_a_task
from Regression_Analysis.DeepLearning_Class import BayesianConv1DBiLSTM
from locale import setlocale, LC_ALL
import tensorflow as tf
import tensorflow_probability as tfp
from functools import reduce
from tqdm import tqdm
from ErrorEvaluation_Class import DeterministicError, ProbabilisticError
from UnivariateAnalysis_Class import UnivariatePDFOrCDFLike

setlocale(LC_ALL, "en_US")
tf.keras.backend.set_floatx('float32')

tfd = eval("tfp.distributions")
tfpl = eval("tfp.layers")
tfb = eval("tfp.bijectors")
tfp_util = eval("tfp.util")
tfp_math = eval("tfp.math")

BATCH_SIZE = 5000
EPOCHS = 1_000_000 + 1


def get_data_for_nn(*, task_num: int, zone_num: int, task: str, only_return_for_nn=True):
    assert task_num in range(1, 16)
    assert zone_num in range(1, 11)
    assert task in {"training", "test"}

    dataset = get_data_for_a_task(task_num)[f"{task} set"]
    gefcom_2014_dataset = GEFCom2014SingleZoneDataSet(dataset=dataset, task_num=task_num, zone_num=zone_num)
    gefcom_2014_dataset_windowed = gefcom_2014_dataset.windowed_dataset(
        x_window_length=datetime.timedelta(days=1),
        y_window_length=datetime.timedelta(days=1),
        x_y_start_index_diff=datetime.timedelta(0),
        window_shift=datetime.timedelta(hours=12 if task == "training" else 24),
        batch_size=BATCH_SIZE
    )

    if not only_return_for_nn:
        return dataset, gefcom_2014_dataset_windowed
    else:
        return gefcom_2014_dataset_windowed[0]


class TSE2020SIRevisionBayesianConv1DBiLSTM(BayesianConv1DBiLSTM):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        super().__init__(
            input_shape=input_shape, output_shape=output_shape, batch_size=BATCH_SIZE,
            conv1d_hypers_filters=16, conv1d_hypers_padding="same", conv1d_hypers_kernel_size=3,
            bilstm_hypers_units=32,
            use_encoder_decoder=False,
            dense_hypers_units=9
        )

    def build(self):
        model = tf.keras.Sequential(
            [
                # tf.keras.layers.Conv1D(9, 3, 1, "same", input_shape=self.input_shape),
                # tf.keras.layers.Dropout(0.2),
                self.get_convolution1d_reparameterization_layer(),

                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
                tf.keras.layers.Dropout(0.5),
                # self.get_bilstm_reparameterization_layer(True),

                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
                tf.keras.layers.Dropout(0.5),
                # self.get_bilstm_reparameterization_layer(True),

                # tf.keras.layers.Dense(1),
                self.get_dense_variational_layer(),
                self.get_distribution_layer()
            ]
        )
        return model

    def get_distribution_layer(self, dtype=tf.float32):
        # dist_layer = tfpl.MixtureSameFamily(
        #     num_components=3,
        #     component_layer=tfpl.DistributionLambda(
        #         make_distribution_fn=lambda t: tfd.Independent(
        #             tfd.LogitNormal(
        #                 loc=tfb.Sigmoid().forward(tf.expand_dims(t[..., 0], -1)),
        #                 scale=tfb.Softplus().forward(tf.expand_dims(t[..., 1], -1)),
        #             ),
        #             reinterpreted_batch_ndims=1
        #         )
        #     )
        # )

        # dist_layer = tfpl.MixtureSameFamily(
        #     num_components=3,
        #     component_layer=tfpl.DistributionLambda(
        #         make_distribution_fn=lambda t: tfd.Independent(
        #             tfd.TruncatedNormal(
        #                 loc=tfb.Sigmoid().forward(tf.expand_dims(t[..., 0], -1)),
        #                 scale=tfb.Softplus().forward(tf.expand_dims(t[..., 1], -1)),
        #                 low=-1e6,
        #                 high=1 + 1e-6
        #             ),
        #             reinterpreted_batch_ndims=1
        #         )
        #     )
        # )

        # dist_layer = tfpl.DistributionLambda(
        #     make_distribution_fn=lambda t: tfd.Independent(
        #         tfd.TruncatedNormal(
        #             loc=tfb.Sigmoid().forward(tf.expand_dims(t[..., 0], -1)),
        #             scale=tfb.Softplus().forward(tf.expand_dims(t[..., 1], -1)),
        #             low=-1e-3,
        #             high=1 + 1e-3
        #         ),
        #         reinterpreted_batch_ndims=1
        #     )
        # )

        # dist_layer = tfpl.IndependentNormal(
        #     event_shape=[1]
        # )

        # dist_layer = tfpl.MixtureNormal(
        #     3, event_shape=[1]
        # )

        dist_layer = tfpl.MixtureSameFamily(
            num_components=3,
            component_layer=tfpl.DistributionLambda(
                make_distribution_fn=lambda t: tfd.Independent(
                    tfd.TransformedDistribution(
                        distribution=tfd.Normal(
                            loc=tf.expand_dims(t[..., 0], -1),
                            scale=1e-3 + tf.math.softplus(tf.expand_dims(t[..., 1], -1)),
                        ),
                        bijector=tfb.SoftClip(low=-1e-3, high=1 + 1e-3)
                    ),
                    reinterpreted_batch_ndims=1
                )
            )
        )
        return dist_layer


def train_nn_model(*, task_num: int, zone_num: int, continue_training: bool = False, fine_tuning: bool = False):
    print("★" * 79)
    print(f"{'Continue ' if continue_training else ''}Train Task {task_num} for Zone {zone_num}")
    print("★" * 79)
    # Get data
    training_data_for_nn = get_data_for_nn(task_num=task_num, zone_num=zone_num, task='training')
    training_data_for_nn = list(training_data_for_nn.as_numpy_iterator())
    training_x = reduce(lambda a, b: np.concatenate([a, b], axis=0), [x[0] for x in training_data_for_nn])
    training_y = reduce(lambda a, b: np.concatenate([a, b], axis=0), [x[1] for x in training_data_for_nn])
    del training_data_for_nn

    test_data_for_nn = list(get_data_for_nn(task_num=task_num, zone_num=zone_num, task='test').as_numpy_iterator())
    print()
    test_x = reduce(lambda a, b: np.concatenate([a, b], axis=0), [x[0] for x in test_data_for_nn])
    test_y = reduce(lambda a, b: np.concatenate([a, b], axis=0), [x[1] for x in test_data_for_nn])
    del test_data_for_nn

    # Define NLL. NB, KL part and its weight/scale factor has been passed through divergence_fn or regularizer
    def nll(y_true, y_pred):
        return -y_pred.log_prob(y_true)
        # return tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred)) + tf.reduce_sum(model.losses)

    # Build model
    model = TSE2020SIRevisionBayesianConv1DBiLSTM(
        input_shape=training_x.shape[1:],
        output_shape=training_y.shape[1:]
    )
    model = model.build()

    model_folder_path = Path(f"./data_gefcom2014/results/task_{task_num}_zone_{zone_num}")
    try_to_find_folder_path_otherwise_make_one(model_folder_path)
    history = dict()

    # Define Callbacks
    class SaveCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % int(EPOCHS * 0.01) == 0:
                model.save_weights(model_folder_path / f"epoch_{epoch}.h5")

            if epoch % 500 == 0:
                print(f"Epoch {epoch}/{EPOCHS}\n"
                      f"loss = {logs.get('loss'):.4f}, mae = {logs.get('mae'):.4f}, "
                      f"rmse = {logs.get('root_mean_squared_error'):.4f} |",
                      f"val_loss = {logs.get('val_loss'):.4f}, val_mae = {logs.get('val_mae'):.4f}, "
                      f"val_rmse = {logs.get('val_root_mean_squared_error'):.4f}")
                history[epoch] = {
                    'loss': logs.get('loss'),
                    'mae': logs.get('mae'),
                    'rmse': logs.get('root_mean_squared_error'),
                    'val_loss': logs.get('val_loss'),
                    'val_mae': logs.get('val_mae'),
                    'val_rmse': logs.get('val_root_mean_squared_error')
                }

            if epoch % int(EPOCHS * 0.02) == 0:
                if epoch == 0:
                    return
                model_y = np.array([model(test_x, training=True).sample().numpy() for _ in range(300)])
                model_output_pdf = []
                print("=" * 79)
                for day in range(model_y.shape[1]):
                    for hour in range(model_y.shape[2]):
                        model_output_pdf.append(
                            UnivariatePDFOrCDFLike.init_from_samples_by_ecdf(model_y[:, day, hour, 0]))
                print(f"Current under-training model performance on test set, task = {task_num}, zone = {zone_num}")
                deter_err = DeterministicError(target=test_y.flatten(), model_output=np.mean(model_y, axis=0).flatten())
                test_mae = deter_err.cal_mean_absolute_error()
                test_rmse = deter_err.cal_root_mean_square_error()
                print(f"Test MAE = {test_mae:.4f}")
                print(f"Test RMSE = {test_rmse:.4f}")

                prob_err = ProbabilisticError(target=test_y.flatten(), model_output=model_output_pdf)
                test_pinball_loss = prob_err.cal_pinball_loss(quantiles=np.arange(0.01, 1., 0.01))
                print(f"Pinball Loss = {test_pinball_loss:.4f}")
                print("=" * 79)

                if epoch not in history:
                    history[epoch] = dict()
                history[epoch]['Test MAE'] = test_mae
                history[epoch]['Test RMSE'] = test_rmse
                history[epoch]['Test Pinball Loss'] = test_pinball_loss

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(EPOCHS * 0.3))
    model.compile(loss=nll,
                  optimizer=tf.keras.optimizers.Adam(0.0001 if fine_tuning else 0.001),
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError()],
                  experimental_run_tf_function=False)
    model.summary()

    if continue_training:
        model.load_weights(model_folder_path / "to_continue.h5")

    model.fit(
        x=training_x, y=training_y, batch_size=BATCH_SIZE, verbose=0, epochs=EPOCHS,
        validation_split=0.05, validation_freq=1, validation_batch_size=BATCH_SIZE // 2, shuffle=False,
        callbacks=[SaveCallback(), early_stopping]
    )
    model.save_weights(model_folder_path / "final.h5")
    save_pkl_file(model_folder_path / "history.pkl", history)

    return model


def test_nn_model(*, task_num: int, zone_num: int):
    # Get data
    test_data_for_nn = list(get_data_for_nn(task_num=task_num, zone_num=zone_num, task='test').as_numpy_iterator())
    test_x = reduce(lambda a, b: np.concatenate([a, b], axis=0), [x[0] for x in test_data_for_nn])
    test_y = reduce(lambda a, b: np.concatenate([a, b], axis=0), [x[1] for x in test_data_for_nn])
    del test_data_for_nn

    # Build model
    model = TSE2020SIRevisionBayesianConv1DBiLSTM(
        input_shape=test_x.shape[1:],
        output_shape=test_y.shape[1:]
    )
    model = model.build()
    model.summary()

    model_folder_path = Path(f"./data_gefcom2014/results/task_{task_num}_zone_{zone_num}")
    model.load_weights(model_folder_path / "final.h5")

    model_y = np.array([model(test_x, training=True).sample().numpy() for _ in tqdm(range(300))])
    model_y_mean = np.mean(model_y, axis=0)

    # model_y = np.array([model.predict(test_x) for _ in tqdm(range(300))])

    yy_05 = np.percentile(model_y, 5, axis=0)
    yy_95 = np.percentile(model_y, 95, axis=0)
    ax = series(test_y[:3].flatten(), label="Actual")
    ax = series(yy_05[:3].flatten(), ax=ax, label="5 %")
    ax = series(model_y_mean[:3].flatten(), ax=ax, label="Mean")
    ax = series(yy_95[:3].flatten(), ax=ax, label="95 %", y_lim=(-.05, 1.05))

    deter_err = DeterministicError(target=test_y.flatten(), model_output=model_y_mean.flatten())
    print(deter_err.cal_root_mean_square_error())
    model_output_pdf = []
    for day in range(model_y.shape[1]):
        for hour in range(model_y.shape[2]):
            model_output_pdf.append(UnivariatePDFOrCDFLike.init_from_samples_by_ecdf(model_y[:, day, hour, 0]))
    prob_err = ProbabilisticError(target=test_y.flatten(), model_output=model_output_pdf)
    print(prob_err.cal_pinball_loss(quantiles=np.arange(0.01, 1., 0.01)))


if __name__ == "__main__":
    pass
    # test_nn_model(task_num=4, zone_num=5)
    # train_nn_model(task_num=4, zone_num=5)
