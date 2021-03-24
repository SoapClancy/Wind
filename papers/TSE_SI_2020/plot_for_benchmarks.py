from papers.TSE_SI_2020.utils import *
from ErrorEvaluation_Class import EnergyBasedError

BENCHMARK_PATH = project_path_ / r"Data\Results\Forecasting\benchmarks"


def get_univariate_pdf_or_cdf_like(uct):
    ans = []
    for i in range(uct.shape[1]):
        now_pred = UnivariatePDFOrCDFLike(cdf_like_ndarray=np.stack(
            [uct.index[:uct.last_nan_index + 1].values.astype(float) / 100,
             uct.iloc[:uct.last_nan_index + 1, i].values, ],
            axis=1)
        )
        ans.append(now_pred)
    return ans


def plot_for_lstm():
    # temp_ans = plot_for_mc()
    for name in ('Bruska', 'Jelinak'):

        df_1 = pd.read_csv(BENCHMARK_PATH / f"lstm/{name}.csv", header=None)
        df_2 = pd.read_csv(BENCHMARK_PATH / f"lstm/{name}_percentiles.csv")
        actual_pu = df_1.iloc[:, 1].values
        forecast_pu = df_1.iloc[:, 0].values
        uct = UncertaintyDataFrame.init_from_template(columns_number=168,
                                                      percentiles=np.concatenate([np.arange(0, 50, 5.),
                                                                                  np.arange(55, 105, 5.)]))
        for j in range(168):
            uct.iloc[1:uct.last_nan_index, j] = df_2.iloc[j, 0:18].values[::-1]
            uct.iloc[-2, j] = forecast_pu[j]
        uct.iloc[0] = 0.
        uct.iloc[uct.last_nan_index] = 1.
        uct = uct.astype(float)

        ax = series(
            np.arange(uct.shape[1]),
            actual_pu,
            color='red',
            linestyle='-.', linewidth=1.2, alpha=0.95, label='Actual',
            figure_size=(6, 5 * 0.551),
            x_label='Time [Hour]',
            x_ticks=(np.arange(0, 168 + 1, 24), np.arange(0, 168 + 1, 24)),
            x_lim=(-1, 169),
            y_label=f'{name} WF Power Output [p.u.]',
            y_lim=WS_POUT_2D_PLOT_KWARGS['y_lim'],
            legend_loc='lower center',
        )

        ax = series(forecast_pu, ax=ax, color='royalblue', linewidth=1.2, alpha=0.95,
                    label='Pred.', legend_loc='lower center',)

        # preds_continuous_var_plot(wf_name=name,
        #                           uct_df=uct,
        #                           target_pout=actual_pu * WF_RATED_POUT_MAPPER[name],
        #                           name='Pout')

        deter_error = DeterministicError(target=actual_pu,
                                         model_output=forecast_pu)
        mae = deter_error.cal_mean_absolute_error()
        rmse = deter_error.cal_root_mean_square_error()
        print(f"MAE = {np.round(mae, 3)}")
        print(f"RMSE = {np.round(rmse, 3)}")

        probs = get_univariate_pdf_or_cdf_like(uct)
        prob_error = ProbabilisticError(target=actual_pu,
                                        model_output=probs)
        crps = prob_error.cal_continuous_ranked_probability_score(integral_boundary=[0., 1.])
        pinball_loss = prob_error.cal_pinball_loss()
        print(f"pinball_loss = {np.round(pinball_loss, 3)}")
        print(f"crps = {np.round(crps, 3)}")

        energy_error = EnergyBasedError(target=actual_pu * WF_RATED_POUT_MAPPER[name],
                                        model_output=forecast_pu * WF_RATED_POUT_MAPPER[name],
                                        time_step=1.)
        energy_error_val = energy_error.do_calculation(
            drop_keys=('target_total_when_over', 'target_total_when_under', 'target_total')
        )
        temp = {key: np.round(val, 3) for key, val in energy_error_val.items()}
        print(f"{temp}")


def plot_for_mc():
    temp_ans = []
    for name in ('Bruska', 'Jelinak'):

        print("☆" * 79)
        print(name)
        file_path = BENCHMARK_PATH / f"mc/{name}.xlsx"
        df = pd.read_excel(file_path)
        uct = UncertaintyDataFrame.init_from_template(columns_number=168,
                                                      percentiles=np.concatenate([np.arange(0, 50, 5.),
                                                                                  np.arange(55, 105, 5.)]))
        for j in range(168):
            uct.iloc[1:uct.last_nan_index, j] = df.iloc[j, :2:-1].values
            uct.iloc[-2, j] = df.iloc[j, 2]

        uct.iloc[0] = 0.
        uct.iloc[uct.last_nan_index] = 1.
        uct = uct.astype(float)
        preds_continuous_var_plot(wf_name=name,
                                  uct_df=uct,
                                  target_pout=df.iloc[:, 1] * WF_RATED_POUT_MAPPER[name],
                                  name='Pout')
        temp_ans.append(df.iloc[:, 1])
        deter_error = DeterministicError(target=df.iloc[:, 1],
                                         model_output=df.iloc[:, 2].values)
        mae = deter_error.cal_mean_absolute_error()
        rmse = deter_error.cal_root_mean_square_error()
        print(f"MAE = {np.round(mae, 3)}")
        print(f"RMSE = {np.round(rmse, 3)}")

        probs = get_univariate_pdf_or_cdf_like(uct)
        prob_error = ProbabilisticError(target=df.iloc[:, 1].values,
                                        model_output=probs)
        crps = prob_error.cal_continuous_ranked_probability_score(integral_boundary=[0., 1.])
        pinball_loss = prob_error.cal_pinball_loss()
        print(f"pinball_loss = {np.round(pinball_loss, 3)}")
        print(f"crps = {np.round(crps, 3)}")

        energy_error = EnergyBasedError(target=df.iloc[:, 1] * WF_RATED_POUT_MAPPER[name],
                                        model_output=df.iloc[:, 2].values * WF_RATED_POUT_MAPPER[name],
                                        time_step=1.)
        energy_error_val = energy_error.do_calculation(
            drop_keys=('target_total_when_over', 'target_total_when_under', 'target_total')
        )
        temp = {key: np.round(val, 3) for key, val in energy_error_val.items()}
        print(f"{temp}")
    return temp_ans


if __name__ == '__main__':
    # plot_for_mc()
    plot_for_lstm()
