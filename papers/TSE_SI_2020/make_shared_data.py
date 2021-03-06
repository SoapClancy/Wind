from project_utils import *
from Ploting.fast_plot_Func import *
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
import datetime
from WT_WF_Class import WF
from PhysicalInstance_Class import *
import pandas as pd
from operating_regime_data_prepare import _print_basic_info
from Data_Preprocessing.TruncatedOrCircularToLinear_Class import CircularToLinear
from prepare_datasets import WF_RATED_POUT_MAPPER, NUMBER_OF_WT_MAPPER, WF_TO_CLUSTER_MAPPER, CLUSTER_TO_WF_MAPPER, \
    Croatia_WF_LOCATION_MAPPER
from functools import reduce
from typing import Dict
import json

PRE_FINAL_DIR = project_path_ / Path(r'./Data/Raw_measurements/TSE_SI_2020_WF/pre_final')
TWO_WEEK = 24 * 14


def get_test_period_all_ok(cluster_data_like: pd.DataFrame) -> Union[Tuple[datetime.datetime], None]:
    i = cluster_data_like.shape[0] - TWO_WEEK
    ans = None
    while i >= 0:
        cluster_data_like_i = cluster_data_like.iloc[i:i + TWO_WEEK]
        nan_mask = np.any(np.isnan(cluster_data_like_i).values, axis=1)

        if not np.any(nan_mask):
            ans = [cluster_data_like.index[i], cluster_data_like.index[i] + datetime.timedelta(hours=TWO_WEEK)]
            break
        else:
            i -= 1

    # if ans is None:
    #     print("No data found... =_=!")
    return ans


def get_test_period_own_ok(cluster_full_data: pd.DataFrame) -> Dict[str, Union[Tuple[datetime.datetime], None]]:
    ans = {key: None for key in cluster_full_data.columns.__getattribute__('levels')[0]}
    for wf_name in cluster_full_data.columns.__getattribute__('levels')[0]:
        ans[wf_name] = get_test_period_all_ok(cluster_full_data[wf_name])
        print(fr"{wf_name} test_period = {ans[wf_name]}")
    return ans


def count_training_set_sample_available(cluster_data_like: pd.DataFrame, test_period_start: datetime.datetime) -> int:
    i = np.argwhere(cluster_data_like.index == test_period_start - datetime.timedelta(days=7))[0][0]
    count = 0
    while i >= 0:
        cluster_data_like_i = cluster_data_like.iloc[i:i + 24 * 7]
        nan_mask = np.any(np.isnan(cluster_data_like_i).values)
        if not nan_mask:
            count += 1
        i -= 1
    return count


def make_one_cluster(cluster_name):
    shared_dir = PRE_FINAL_DIR.parent / Path(r"./shared_data")
    readings = []
    for wf_name in CLUSTER_TO_WF_MAPPER[cluster_name]:
        if wf_name == 'Vostane_Kamensko':
            continue
        one_reading = pd.read_csv(PRE_FINAL_DIR / Path(fr"./{cluster_name} cluster {wf_name} WF_pre_FINAL.csv"))
        one_reading.index = pd.to_datetime(one_reading.iloc[:, 0])
        one_reading.drop(labels=one_reading.columns[0], axis=1, inplace=True)
        one_reading.columns = pd.MultiIndex.from_product([[wf_name], one_reading.columns])
        one_reading.index.name = "time stamp"
        readings.append(one_reading)
    assert np.all([np.all(readings[i].index == readings[i + 1].index) for i in range(readings.__len__() - 1)])

    # get test period
    cluster_full_data = reduce(lambda a, b: pd.merge(a, b, how='left', left_index=True, right_index=True), readings)
    test_periods = get_test_period_own_ok(cluster_full_data)
    save_pkl_file(shared_dir / Path(f"./{cluster_name} cluster test_periods.pkl"), test_periods)

    # get training sets sample count
    training_sets_sample_available = {wf_name: 0 for wf_name in test_periods.keys()}
    for wf_name, test_period in test_periods.items():
        if test_period is None:
            continue
        training_sets_sample_available[wf_name] = count_training_set_sample_available(
            cluster_full_data[wf_name], test_period[0]
        )
        print(f"{wf_name} training_set_sample_available = {training_sets_sample_available[wf_name]}")
    print("=" * 64 + "\n")
    save_pkl_file(shared_dir / Path(f"./{cluster_name} cluster training_sets_sample_available.pkl"),
                  training_sets_sample_available)

    # only the case when both training and test ok are considered
    for pro in ("all", 'training', 'test'):
        try_to_find_folder_path_otherwise_make_one(shared_dir / pro)
    for wf_name, test_period in test_periods.items():
        if test_period is None or training_sets_sample_available[wf_name] < 100:
            continue
        # all
        df = cluster_full_data[wf_name].loc[: test_periods[wf_name][1] - datetime.timedelta(hours=1)]
        df.to_csv(shared_dir / "all" / f"{cluster_name} cluster {wf_name} WF.csv")
        # training
        df = cluster_full_data[wf_name].loc[: test_periods[wf_name][0] - datetime.timedelta(hours=1)]
        df.to_csv(shared_dir / "training" / f"{cluster_name} cluster {wf_name} WF.csv")
        # test
        df = cluster_full_data[wf_name].loc[test_periods[wf_name][0]:
                                            test_periods[wf_name][1] - datetime.timedelta(hours=1)]
        df.to_csv(shared_dir / "test" / f"{cluster_name} cluster {wf_name} WF.csv")

    # make meta
    json_obj = {
        "WF_TO_CLUSTER_MAPPER": WF_TO_CLUSTER_MAPPER,
        "CLUSTER_TO_WF_MAPPER": CLUSTER_TO_WF_MAPPER,
        "WF_RATED_POUT_MAPPER": WF_RATED_POUT_MAPPER,
        "NUMBER_OF_WT_MAPPER": NUMBER_OF_WT_MAPPER,
        "LOCATION_MAPPER": Croatia_WF_LOCATION_MAPPER,
    }
    with open(shared_dir / "meta_data.json", 'w') as json_file:
        json.dump(json_obj, json_file)


def make_shared_data():
    for cluster in CLUSTER_TO_WF_MAPPER.keys():
        make_one_cluster(cluster)


if __name__ == '__main__':
    make_shared_data()
