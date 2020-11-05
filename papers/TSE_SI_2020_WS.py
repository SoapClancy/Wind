from Ploting.fast_plot_Func import *
from project_utils import *
from prepare_datasets import load_croatia_data
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
from tensorflow.keras.layers import Bidirectional, LSTM, Conv1D
import edward2 as ed

exec(r"from TSE2020 import darly_wind_farm_operating_regime")
darly_wind_farm_operating_regime = globals()['darly_wind_farm_operating_regime']  # type: Callable
DARLY_WIND_FARM_RAW, _, DARLY_WIND_FARM_RAW_OPERATING_REGIME = darly_wind_farm_operating_regime()


# Prepare data for NN
class WindFarmDataSet(DeepLearningDataSet):
    def __init__(self, *args, wind_farm: WF, operating_regime: DataCategoryData, **kwargs):
        assert isinstance(wind_farm, WF)
        assert isinstance(operating_regime, DataCategoryData)
        wind_farm_extended = self._expand_feature_dim_for_month_and_operating_regime(wind_farm, operating_regime)
        wind_farm_extended = self._shift_power_output_one_day_ahead(wind_farm_extended)
        super().__init__(*args,
                         original_data_set=wind_farm_extended,
                         **kwargs)

    @staticmethod
    def _expand_feature_dim_for_month_and_operating_regime(wind_farm, operating_regime):
        wind_farm_extended = copy.deepcopy(wind_farm.pd_view()).astype(float)
        wind_farm_extended['month'] = wind_farm_extended.index.month
        return wind_farm_extended

    @staticmethod
    def _shift_power_output_one_day_ahead(wind_farm_extended):
        """
        This is to shift the power output dim one day ahead, as it is a forecasting task!
        :param wind_farm_extended:
        :return:
        """
        shift = wind_farm_extended['wind speed'].shift(periods=-1, freq=datetime.timedelta(1))
        wind_farm_extended = wind_farm_extended.drop(['wind speed'], axis=1)
        wind_farm_extended = pd.merge(wind_farm_extended, shift, how='left', left_index=True, right_index=True)
        return wind_farm_extended


if __name__ == '__main__':
    # this_wind_farm = RESAMPLED_ZELENGRAD_WIND_FARM
    # this_wind_farm_operating_regime = RESAMPLED_ZELENGRAD_OPERATING_REGIME

    this_wind_farm = DARLY_WIND_FARM_RAW  # type: WF
    this_wind_farm = this_wind_farm.drop('active power output', axis=1)

    this_wind_farm_operating_regime = DARLY_WIND_FARM_RAW_OPERATING_REGIME  # type: DataCategoryData
    _mask = this_wind_farm.index < datetime.datetime(2009, 1, 1)
    # Training
    this_wind_farm_training = copy.deepcopy(this_wind_farm)
    this_wind_farm_operating_regime_training = copy.deepcopy(this_wind_farm_operating_regime)
    this_wind_farm_training = this_wind_farm_training[_mask]
    this_wind_farm_operating_regime_training.index = this_wind_farm_operating_regime_training.index[_mask]
    this_wind_farm_operating_regime_training.abbreviation = this_wind_farm_operating_regime_training.abbreviation[_mask]
    # Test
    this_wind_farm_test = copy.deepcopy(this_wind_farm)
    this_wind_farm_operating_regime_test = copy.deepcopy(this_wind_farm_operating_regime)
    this_wind_farm_test = this_wind_farm_test[~_mask]
    this_wind_farm_operating_regime_test.index = this_wind_farm_operating_regime_test.index[~_mask]
    this_wind_farm_operating_regime_test.abbreviation = this_wind_farm_operating_regime_test.abbreviation[~_mask]


    def get_training_or_test_data_set(this_wind_farm_what, this_wind_farm_operating_regime_what):
        what_data_set = WindFarmDataSet(
            wind_farm=this_wind_farm_what,
            operating_regime=this_wind_farm_operating_regime_what,
            cos_sin_transformed_col=('month',),
            min_max_transformed_col=('wind speed',),
            dependant_cols=('wind speed',),
            name=this_wind_farm_what.obj_name + '_training',
            transformation_args_folder_path=project_path_ / f'Data/Results/Forecasting_WS/'
                                                            f'{this_wind_farm_what.obj_name}/transformation_args/',
            stacked_shift_col=('wind speed',),
            stacked_shift_size=(datetime.timedelta(days=1),),
            how_many_stacked=(28,)
        )
        return what_data_set


    training_data_set = get_training_or_test_data_set(this_wind_farm_training,
                                                      this_wind_farm_operating_regime_training)
    training_data_set_for_nn = training_data_set.windowed_dataset(datetime.timedelta(days=1), batch_size=int(730 / 4),
                                                                  shift=int(144 / 3),
                                                                  drop_remainder=True)

    test_data_set = get_training_or_test_data_set(this_wind_farm_test,
                                                  this_wind_farm_operating_regime_test)
    test_data_set_for_nn = test_data_set.windowed_dataset(datetime.timedelta(days=1), batch_size=int(730 / 4),
                                                          shift=int(144 / 3),
                                                          drop_remainder=True)

    # ####################1号模型#############################
    model = tf.keras.models.Sequential([
        tf.keras.layers.Bidirectional(tf.keras.layers.RNN(ed.layers.LSTMCellReparameterization(1024),
                                                          return_sequences=True),
                                      input_shape=training_data_set_for_nn.element_spec[0].shape[1:]),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(training_data_set_for_nn.element_spec[1].shape[-1]),
    ])
    # ####################2号模型#############################
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Bidirectional(tf.keras.layers.RNN(ed.layers.LSTMCellReparameterization(1500),
    #                                                       return_sequences=True),
    #                                   input_shape=training_data_set_for_nn.element_spec[0].shape[1:],),
    #     tf.keras.layers.Dense(500, activation="relu"),
    #     tf.keras.layers.Dense(training_data_set_for_nn.element_spec[1].shape[-1]),
    # ])
    # ####################3号模型#############################
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=128, kernel_size=3,
                               strides=1, padding="causal",
                               activation="relu",
                               input_shape=training_data_set_for_nn.element_spec[0].shape[1:]),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(1024),
                                                          return_sequences=True), ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(1024),
                                                          return_sequences=True)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(training_data_set_for_nn.element_spec[1].shape[-1]),
    ])
    # ####################载入模型#############################
    # model.load_weights('E:\mymodel_epoch_500.h5')
    pass

    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 25 == 0:
                model.save(f'E:\mymodel_epoch_{epoch}.h5')


    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=keras.losses.mae, metrics=['mse'])
    history = model.fit(training_data_set_for_nn, verbose=2, epochs=10000, validation_data=test_data_set_for_nn,
                        callbacks=[CustomCallback()], )

    model.save('mymodel.h5')

    training_data_set_for_nn = training_data_set.windowed_dataset(datetime.timedelta(hours=1), batch_size=500 * 24,
                                                                  drop_remainder=True)

    test_data_set_for_nn = test_data_set.windowed_dataset(datetime.timedelta(hours=1), batch_size=500 * 24,
                                                          drop_remainder=True)

    test_list = list(test_data_set_for_nn.as_numpy_iterator())[0]
    training_list = list(training_data_set_for_nn.as_numpy_iterator())[0]


    def plot_plot(day: int):
        day_x = test_list[0][day]
        day_y = test_list[1][day].flatten()
        pre = model.predict(day_x[np.newaxis, ...]).flatten()
        ax = series(day_y, label='True')
        ax = series(pre, label='Model', ax=ax, y_lim=(-0.05, 1.05))


    def plot_plot_true(day: int, test=True):
        if test:
            day_y = test_list[1][day].flatten()
        else:
            day_y = training_list[1][day].flatten()
        ax = series(day_y, color='r', label='True')


    def plot_plot_uct(day: int, test=True):
        if test:
            day_x = test_list[0][day]
            day_y = test_list[1][day].flatten()
        else:
            day_x = training_list[0][day]
            day_y = training_list[1][day].flatten()
        pre = np.full((50, 144), np.nan)
        for i in range(50):
            pre[i] = model(day_x[np.newaxis, ...], training=True).numpy().flatten()
        ax = series(pre.T, color='g')
        ax = series(day_y, color='r', label='True', ax=ax)


    # series((test_data_set.transformed_data.values[:, 2]))
    series(training_list[1][1].flatten())
    series((training_data_set.transformed_data.values[28 * 144:29 * 144, 2]))
