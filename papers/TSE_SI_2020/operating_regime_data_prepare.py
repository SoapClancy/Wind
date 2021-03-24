from project_utils import *
from Ploting.fast_plot_Func import *
from prepare_datasets import load_croatia_data_tse_si_2020, WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, load_croatia_data
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
import datetime
from WT_WF_Class import WF
from PhysicalInstance_Class import *
import pandas as pd


def _print_basic_info(wf_obj):
    print(wf_obj)
    print(f"Rated power = {wf_obj.rated_active_power_output}")
    print(f"Number of WF = {wf_obj.number_of_wind_turbine}")
    print(f"Row count if any missing = {np.sum(np.any(np.isnan(wf_obj.values), axis=1))}")


def fit_or_load(wf_obj, task: str, load_with_new_params: dict = None, ts_freq_minutes=60):
    assert task in ('fit', 'load')
    not_nan_mask = np.bitwise_and(~np.isnan(wf_obj.iloc[:, 0]), ~np.isnan(wf_obj.iloc[:, 1]))
    wf_obj = wf_obj[not_nan_mask]
    if task == 'fit':
        wf_obj.operating_regime_detector(task='fit', ts_freq_minutes=ts_freq_minutes)
    else:
        wf_pc_obj, operating_regime = wf_obj.operating_regime_detector(task='load', ts_freq_minutes=ts_freq_minutes,
                                                                       load_with_new_params=load_with_new_params)
        # wf_pc_obj.assess_fit_2d_scatters(operating_regime=operating_regime, total_curtailment_amount=[0],
        #                                  original_scatters_pc=wf_pc_obj.corresponding_mob_pc_obj)
        ax = wf_pc_obj.corresponding_8p_pc_obj.plot(color='g')
        # wf_obj.plot(operating_regime=operating_regime, not_show_color_bar=True, ax=ax)

        # wf_obj.init_assuming_all_fully_operating(ts_freq_minutes=ts_freq_minutes).plot()
        # wf_obj[operating_regime('S1')].plot()
        # wf_obj[operating_regime('S2')].plot()
        # wf_obj[operating_regime('S3')].plot()
        print(f"fully operation percentage = {wf_obj[operating_regime('S1')].shape[0] / wf_obj.shape[0]}")

        # operating_regime = wf_obj.operating_regime_detector(task='load raw', ts_freq_minutes=ts_freq_minutes,
        #                                                     load_with_new_params=load_with_new_params)
        # to_save = wf_obj.pd_view()
        # to_save['normally operating number'] = operating_regime[0]['normally_operating_number']
        # scatter(to_save.loc[to_save.iloc[:, 2] == 15].iloc[:, 0].values,
        #         to_save.loc[to_save.iloc[:, 2] == 15].iloc[:, 1].values)
        # series(to_save.iloc[:, 2])
        # to_save.to_csv(f"{wf_obj.obj_name}_with_OPR.csv")
        return operating_regime


def get_bruska_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Zadar', 'Bruska')
    wf_obj.plot()
    _print_basic_info(wf_obj)
    # 手动赋初值
    """
    DEBUG_VAR.b_2 = 28.
    DEBUG_VAR.c_2 = 21.8
    DEBUG_VAR.g_2 = 1.08
    """
    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


def get_bruska_hi_operating_regime_estimation(task='fit'):
    wf_obj = load_croatia_data('Bruska')['Bruska']
    wf_obj.obj_name = 'Bruska hi'
    wf_obj = wf_obj.loc[np.bitwise_and(datetime.datetime(2018, 1, 1) <= wf_obj.index,
                                       wf_obj.index < datetime.datetime(2019, 12, 25))]
    wf_obj = wf_obj.resample('2T', resampler_obj_func_source_code='mean()')
    wf_obj.loc[wf_obj['wind speed'] < 3.] = np.nan
    wf_obj.number_of_wind_turbine = NUMBER_OF_WT_MAPPER['Bruska']
    not_nan_mask = np.bitwise_and(~np.isnan(wf_obj.iloc[:, 0]), ~np.isnan(wf_obj.iloc[:, 1]))
    wf_obj = wf_obj[not_nan_mask]

    wf_obj.plot()
    _print_basic_info(wf_obj)
    """
                [
                ('a', [1 - float_eps, 1 + float_eps]),
                ('d', [-float_eps, float_eps]),
                ('b_1', [-15., -6.]),
                ('c_1', [8., 15.]),
                ('g_1', [float_eps, 0.6]),
                ('b_2', [0., 60.]),
                ('c_2', [18., 30.]),
                ('g_2', [float_eps, 4.]),
            ]
    """
    tt = 1
    operating_regime = fit_or_load(wf_obj, task,
                                   load_with_new_params={
                                       'a': 1.,
                                       'd': 0.,
                                       'b_1': -10.9,
                                       'c_1': 10.6,
                                       'g_1': 0.21,
                                       'b_2': 39.2,
                                       'c_2': 27,
                                       'g_2': 1.021,
                                   },
                                   ts_freq_minutes=2)

    operating_regime.name_mapper.loc[:, 'long name'] = [str(17 - int(x[1:])) for x in
                                                        operating_regime.name_mapper.loc[:, 'abbreviation'].values]

    wf_obj.plot(operating_regime=operating_regime, not_show_color_bar=False, ax=None,
                y_label='Bruska WF Power Output [p.u.]', x_lim=(-0.5, 32.5),)


def get_benkovac_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Zadar', 'Benkovac')
    wf_obj.plot()
    _print_basic_info(wf_obj)

    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


def get_zelengrad_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Zadar', 'Zelengrad')
    wf_obj.plot()
    _print_basic_info(wf_obj)

    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


def get_katuni_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Split', 'Katuni')
    wf_obj.plot()
    _print_basic_info(wf_obj)

    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


def get_lukovac_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Split', 'Lukovac')
    wf_obj.plot()
    _print_basic_info(wf_obj)

    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


def get_vostane_kamensko_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Split', 'Vostane_Kamensko')
    wf_obj.plot()
    _print_basic_info(wf_obj)

    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


def get_glunca_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Sibenik', 'Glunca')
    wf_obj.plot()
    _print_basic_info(wf_obj)

    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


def get_jelinak_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Sibenik', 'Jelinak')
    wf_obj.plot()
    _print_basic_info(wf_obj)

    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


def get_velika_glava_operating_regime_estimation(task='fit', *, load_with_new_params: dict = None):
    wf_obj = load_croatia_data_tse_si_2020('Sibenik', 'VelikaGlava')
    wf_obj.plot()
    _print_basic_info(wf_obj)

    fit_or_load(wf_obj, task, load_with_new_params=load_with_new_params)


if __name__ == '__main__':
    get_bruska_hi_operating_regime_estimation(
        'load'
    )
    # get_bruska_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -8.6,
    #         'c_1': 10.4,
    #         'g_1': 0.33
    #     }
    # )
    # get_benkovac_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -9.,
    #         'c_1': 9.4,
    #         'g_1': 0.22
    #     }
    # )
    # get_zelengrad_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -9.,
    #         'c_1': 13.1,
    #         'g_1': 0.25
    #     }
    # )

    """"""
    # get_katuni_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -9.,
    #         'c_1': 10.6,
    #         'g_1': 0.33
    #     }
    # )
    # get_lukovac_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -9.,
    #         'c_1': 10.9,
    #         'g_1': 0.32
    #     }
    # )
    # get_vostane_kamensko_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -6.5,
    #         'c_1': 10.6,
    #         'g_1': 0.3
    #     }
    # )
    """"""
    # get_glunca_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -8.5,
    #         'c_1': 9.3,
    #         'g_1': 0.34
    #     }
    # )
    # get_jelinak_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -8.1,
    #         'c_1': 12.1,
    #         'g_1': 0.31
    #     }
    # )
    # get_velika_glava_operating_regime_estimation(
    #     'load',
    #     load_with_new_params={
    #         'b_1': -8.6,
    #         'c_1': 11.1,
    #         'g_1': 0.31,
    #         'b_2': 39.,
    #         'c_2': 25.,
    #         'g_2': 3.1,
    #     }
    # )

    pass
    """ 检查为什么hourly没有clear的operating regime pattern
    kat = load_croatia_data('Katuni')
    kat = kat['Katuni']  # type: WF
    kat_resample = kat.resample('60T', resampler_obj_func_source_code='mean()')

    kat_df = kat.pd_view()
    kat_resample_df = kat_resample.pd_view()
    # kat.plot()
    # kat.resample('60T').mean().plot()
    # scatter(*kat.resample('60T').mean().values.T)

    curtail_mask = kat.data_category_inside_boundary(
        {
            'wind speed': [15, 20],
            'active power output': np.array([0.55, 0.58]) * kat.rated_active_power_output
        }
    )
    curtail_index = kat.index[curtail_mask]
    curtail_df = kat.loc[curtail_index].pd_view()

    kat_df_certain_days = kat_df.loc[np.bitwise_and(kat_df.index >= datetime.datetime(2017, 3, 8),
                                                    kat_df.index < datetime.datetime(2017, 3, 9))]

    # kat_df_certain_days_reindex = kat_df_certain_days.reindex(
    #     pd.period_range(start='2017-03-08', end='2017-03-08 23:59:00', freq='T').to_timestamp())

    kat_df_certain_days_reindex = kat_df.reindex(
        pd.period_range(start='2017-02-01', end='2020-02-25 23:59:00', freq='T').to_timestamp())

    pout = np.reshape(kat_df_certain_days_reindex.values.T[1], (-1, 60))
    pout_mean = np.nanmean(pout, axis=1)

    ws = np.reshape(kat_df_certain_days_reindex.values.T[0], (-1, 60))
    ws_mean = np.nanmean(ws, axis=1)
    scatter(ws_mean, pout_mean)
    """
    # wf_obj = load_croatia_data_tse_si_2020('Zadar', 'Bruska')
    #
    # wf_obj.init_assuming_all_fully_operating(60).plot()
