from Ploting.fast_plot_Func import *
import pandas as pd
from MarkovChain.MarkovChain_Class import OneDimMarkovChain
from prepare_datasets import load_dalry_wind_farm_met_mast
from Ploting.adjust_Func import *


def fit_mc(*, ws_resol=None, wd_resol=None):
    assert (ws_resol or wd_resol) is not None
    met_mast_raw = load_dalry_wind_farm_met_mast()

    mask = np.full(met_mast_raw.__len__(), fill_value=np.False_)
    mask[:len(met_mast_raw) // 3] = np.True_

    ws_raw = met_mast_raw.loc[mask, "wind speed_2"]
    wd_raw = met_mast_raw.loc[mask, "wind direction"]

    analysis_obj = dict()
    if ws_resol is not None:
        analysis_obj.setdefault("Wind speed", [ws_raw, ws_resol])
    if wd_resol is not None:
        analysis_obj.setdefault("Wind direction", [wd_raw, wd_resol])

    test_slice = slice(0, 144)
    ws_test = ws_raw.values[test_slice]
    wd_test = wd_raw.values[test_slice]

    for key, value in analysis_obj.items():
        now_name = key  # type: str
        now_data = value[0]  # type: pd.Series
        now_resol = value[1]  # type: Union[float, int]
        mc_obj = OneDimMarkovChain.init_from_one_dim_ndarray(now_data.values, now_resol)

        now_test = ws_test if now_name == "Wind speed" else wd_test
        now_mc_range = []
        now_mc_mean = []
        range_obj = range(test_slice.start, test_slice.stop, test_slice.step or 1)
        for i in range_obj:
            now_mc_range.append(mc_obj.get_next_digitize_range_from_current_raw(now_test[i], (5, 95),
                                                                                method="sampling",
                                                                                # method="interpolation",
                                                                                ))
            now_mc_mean.append(mc_obj.get_next_digitize_mean_from_current_raw(now_test[i],
                                                                              method="sampling",
                                                                              # method="weighted average"
                                                                              ))
        now_mc_range = np.array(now_mc_range)
        now_mc_mean = np.array(now_mc_mean)

        # Plot
        y_label = now_name + ("\n[m/s]" if "speed" in now_name else "\n[degrees]")
        y_lim = (-0.05, 35.5) if "speed" in now_name else (-1, 361)
        ax = series(x=range_obj[1:], y=now_test[1:], color="royalblue", figure_size=(10, 2.4), label="Actual recording",
                    x_label="Recording index [every 10 min]", y_label=y_label, y_lim=y_lim,
                    x_lim=(range_obj[1:][0] - 1, range_obj[1:][-1] + 1))
        ax = series(x=range_obj[1:], y=now_mc_range[:-1, 0], ax=ax, color="green", linestyle="--",
                    label="MC 5-95 percentiles")
        ax = series(x=range_obj[1:], y=now_mc_mean[:-1], ax=ax, color="red", linestyle="-.", label="MC average")
        adjust_legend_in_ax(ax, ncol=3, loc="best")
        ax = series(x=range_obj[1:], y=now_mc_range[:-1, 1], ax=ax, color="green", linestyle="--",
                    save_file_=f"{now_name}_{now_resol}", save_format="svg")


if __name__ == "__main__":
    ws_resols = [0.5, 1, 2, 5]
    wd_resols = [10, 30, 60, 90]

    for r_ws in ws_resols:
        if r_ws == 0.5:
            fit_mc(ws_resol=r_ws)

    # for r_wd in wd_resols:
    #     if r_wd == 30:
    #         fit_mc(wd_resol=r_wd)
