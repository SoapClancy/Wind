import pandas as pd
import math
from Ploting.fast_plot_Func import *
from project_utils import *
from File_Management.path_and_file_management_Func import *
from File_Management.load_save_Func import *
from Data_Preprocessing.TruncatedOrCircularToLinear_Class import CircularToLinear
import datetime
from dateutil.relativedelta import relativedelta
from Regression_Analysis.DataSet_Class import DeepLearningDataSet
import copy

_AVAILABLE_START_ALL = datetime.datetime(2012, 1, 1, 1)
available_end_1 = datetime.datetime(2012, 10, 1, 0)
_GEFCOM2014_AVAILABLE_DATA_PERIOD = {
    i + 1: (_AVAILABLE_START_ALL, available_end_1 + relativedelta(months=i)) for i in range(15)
}  # inclusive
del available_end_1

benchmark_start_1 = datetime.datetime(2012, 10, 1, 1)
benchmark_end_1 = datetime.datetime(2012, 11, 1, 0)
# add the additional last 3 days from the training set
_GEFCOM2014_BENCHMARK_PERIOD = {
    i + 1: (benchmark_start_1 + relativedelta(months=i) - relativedelta(days=3),
            benchmark_end_1 + relativedelta(months=i)) for i in range(15)
}  # inclusive


@load_exist_pkl_file_otherwise_run_and_save(Path("./data_gefcom2014/gefcom2014_pkl.pkl"))
def get_full_raw_file_in_pkl():
    full_reading = pd.DataFrame()
    for i in range(1, 11):
        one_reading = pd.read_csv(Path(f"./data_gefcom2014/Task15_W_Zone1_10/Task15_W_Zone{i}.csv"))
        # set index and rename
        one_reading.index = pd.to_datetime(one_reading["TIMESTAMP"].values)
        one_reading.drop(["ZONEID", 'TIMESTAMP'], axis=1, inplace=True)
        one_reading.rename({"TARGETVAR": "active power output"}, axis=1, inplace=True)
        # add wind speed
        one_reading["wind speed 10"] = np.sqrt(one_reading["U10"] ** 2 + one_reading["V10"] ** 2)
        one_reading["wind speed 100"] = np.sqrt(one_reading["U100"] ** 2 + one_reading["V100"] ** 2)
        # add wind direction
        transformer = CircularToLinear(math.tau, -1., 1.)
        one_reading["wind direction 10"] = transformer.inverse_transform(
            one_reading["U10"] / one_reading["wind speed 10"],
            one_reading["V10"] / one_reading["wind speed 10"],
        ) / math.pi * 180.
        one_reading["wind direction 100"] = transformer.inverse_transform(
            one_reading["U100"] / one_reading["wind speed 100"],
            one_reading["V100"] / one_reading["wind speed 100"],
        ) / math.pi * 180.
        # make multi-level column
        one_reading.columns = pd.MultiIndex.from_product([[f"Zone {i}"], one_reading.columns],
                                                         names=["#Zone", "Feature"])
        # merge
        full_reading = pd.merge(full_reading, one_reading, left_index=True, right_index=True, how="outer")
    full_reading.columns = pd.MultiIndex.from_tuples(full_reading.columns)
    # full_reading.loc[:, (slice(None), 'active power output')]
    # fill missing
    full_reading = full_reading.interpolate(method='spline', order=5, limit=24, limit_direction='both')
    for col_name in full_reading.columns:
        if "active power output" in col_name:
            full_reading.loc[full_reading[col_name] < 0, col_name] = 0
            full_reading.loc[full_reading[col_name] > 1, col_name] = 1
    return full_reading


def get_full_provided_benchmark_in_pkl():
    pass


def get_solution_15():
    pass


def get_data_for_a_task(task_num: int):
    ans = {key: None for key in ["training set", "test set", "benchmark"]}
    full_raw_file = get_full_raw_file_in_pkl()
    # training set
    ans["training set"] = full_raw_file.loc[
        np.bitwise_and(full_raw_file.index >= _GEFCOM2014_AVAILABLE_DATA_PERIOD[task_num][0],
                       full_raw_file.index <= _GEFCOM2014_AVAILABLE_DATA_PERIOD[task_num][1])
    ]
    # test set
    if task_num == 15:
        raise NotImplementedError
    ans["test set"] = full_raw_file.loc[
        np.bitwise_and(full_raw_file.index >= _GEFCOM2014_BENCHMARK_PERIOD[task_num][0],
                       full_raw_file.index <= _GEFCOM2014_BENCHMARK_PERIOD[task_num][1])
    ]
    # benchmark
    # TODO
    return ans


class GEFCom2014DataSet(DeepLearningDataSet):
    __slots__ = ("task_num",)

    @staticmethod
    def make_data_set(dataset: pd.DataFrame):
        """
        no more MultiIndex
        add date time feature
        """
        dataset = copy.deepcopy(dataset)
        # flatten the index
        dataset.columns = [f"{a}-{b}" for a, b in dataset.columns.to_flat_index()]
        # dataset.columns = dataset.columns.to_flat_index()
        # get annual cycle
        so_far_seconds = (dataset.index - _AVAILABLE_START_ALL).total_seconds().values
        annual_total_seconds = 364 * 24 * 3600
        dataset["annual cos cycle"] = np.cos(math.tau * so_far_seconds / annual_total_seconds)
        dataset["annual sin cycle"] = np.sin(math.tau * so_far_seconds / annual_total_seconds)
        # get seasonal cycle
        seasonal_total_seconds = 91 * 24 * 3600
        dataset["seasonal cos cycle"] = np.cos(math.tau * so_far_seconds / seasonal_total_seconds)
        dataset["seasonal sin cycle"] = np.sin(math.tau * so_far_seconds / seasonal_total_seconds)
        # get monthly cycle
        monthly_total_seconds = 28 * 24 * 3600
        dataset["monthly cos cycle"] = np.cos(math.tau * so_far_seconds / monthly_total_seconds)
        dataset["monthly sin cycle"] = np.sin(math.tau * so_far_seconds / monthly_total_seconds)
        """
        # get weekly cycle
        weekly_total_seconds = 7 * 24 * 3600
        dataset["weekly cos cycle"] = np.cos(math.tau * so_far_seconds / weekly_total_seconds)
        dataset["weekly sin cycle "] = np.sin(math.tau * so_far_seconds / weekly_total_seconds)
        # get daily cycle
        daily_total_seconds = 1 * 24 * 3600
        dataset["daily cos cycle"] = np.cos(math.tau * so_far_seconds / daily_total_seconds)
        dataset["daily sin cycle"] = np.sin(math.tau * so_far_seconds / daily_total_seconds)
        """
        return dataset


class GEFCom2014SingleZoneDataSet(GEFCom2014DataSet):
    __slots__ = ("zone_num",)

    def __init__(self, *args, dataset: pd.DataFrame, task_num: int, zone_num: int, **kwargs):
        assert type(dataset) == pd.DataFrame
        assert task_num in range(1, 16)
        assert zone_num in range(1, 11)

        self.task_num = task_num
        self.zone_num = zone_num
        self.data = self.make_data_set(dataset)

        predictor_cols = self.get_predictor_cols()
        dependant_cols = self.get_dependant_cols()
        quantile_transformed_col = self.get_quantile_transformed_col()
        non_transformed_col = self.get_non_transformed_col()
        stacked_shift_col = self.get_stacked_shift_col()
        self.data = self.data[predictor_cols + dependant_cols]

        super().__init__(
            *args,
            original_data_set=self.data,
            predictor_cols=tuple(predictor_cols),
            dependant_cols=tuple(dependant_cols),
            quantile_transformed_col=tuple(quantile_transformed_col),
            non_transformed_col=non_transformed_col,
            name=f"Task_{self.task_num}_Zone_{self.zone_num}_training",
            transformation_args_folder_path=Path("./data_gefcom2014/results/transformation_args"),
            stacked_shift_col=stacked_shift_col,
            stacked_shift_size=[datetime.timedelta(days=1) for _ in range(len(stacked_shift_col))],
            how_many_stacked=[3 for _ in range(len(stacked_shift_col))],
            **kwargs
        )

    def get_predictor_cols(self):
        return [x for x in self.data.columns
                if "active power output" not in x and
                "U10" not in x and
                "V10" not in x and
                "U100" not in x and
                "V100" not in x]

    # def get_predictor_cols(self):
    #     return [f"Zone {self.zone_num}-wind speed 10", f"Zone {self.zone_num}-wind speed 100",
    #             f"Zone {self.zone_num}-wind direction 10", f"Zone {self.zone_num}-wind direction 100"] + \
    #             [x for x in self.data.columns if "cycle" in x]

    def get_dependant_cols(self):
        return [f"Zone {self.zone_num}-active power output"]

    def get_quantile_transformed_col(self):
        return [x for x in self.data.columns
                if "active power output" not in x and
                "U10" not in x and
                "V10" not in x and
                "U100" not in x and
                "V100" not in x and
                "cycle" not in x]

    # def get_quantile_transformed_col(self):
    #     return [f"Zone {self.zone_num}-wind speed 10", f"Zone {self.zone_num}-wind speed 100",
    #             f"Zone {self.zone_num}-wind direction 10", f"Zone {self.zone_num}-wind direction 100"]

    def get_non_transformed_col(self):
        return [x for x in self.data.columns if "cycle" in x or
                x == f"Zone {self.zone_num}-active power output"]

    def get_stacked_shift_col(self):
        return [x for x in self.get_predictor_cols() if "cycle" not in x]


class GEFCom2014AllZoneDataSet(GEFCom2014DataSet):
    def __init__(self, *args, dataset: pd.DataFrame, task_num: int, **kwargs):
        assert type(dataset) == pd.DataFrame
        assert task_num in range(1, 16)

        self.task_num = task_num
        self.data = self.make_data_set(dataset)

        predictor_cols = self.get_predictor_cols()
        dependant_cols = self.get_dependant_cols()
        quantile_transformed_col = self.get_quantile_transformed_col()
        non_transformed_col = self.get_non_transformed_col()
        stacked_shift_col = self.get_stacked_shift_col()
        self.data = self.data[predictor_cols + dependant_cols]

        super().__init__(
            *args,
            original_data_set=self.data,
            predictor_cols=tuple(predictor_cols),
            dependant_cols=tuple(dependant_cols),
            quantile_transformed_col=tuple(quantile_transformed_col),
            non_transformed_col=non_transformed_col,
            name=f"Task_{self.task_num}_Zone_All_training",
            transformation_args_folder_path=Path("./data_gefcom2014/results/transformation_args"),
            stacked_shift_col=stacked_shift_col,
            stacked_shift_size=[datetime.timedelta(days=1) for _ in range(len(stacked_shift_col))],
            how_many_stacked=[3 for _ in range(len(stacked_shift_col))],
            **kwargs
        )

    def get_predictor_cols(self):
        return [x for x in self.data.columns
                if "active power output" not in x and
                "U10" not in x and
                "V10" not in x and
                "U100" not in x and
                "V100" not in x]

    def get_dependant_cols(self):
        return [x for x in self.data.columns if "active power output" in x]

    def get_quantile_transformed_col(self):
        return [x for x in self.data.columns
                if "active power output" not in x and
                "U10" not in x and
                "V10" not in x and
                "U100" not in x and
                "V100" not in x and
                "cycle" not in x]

    def get_non_transformed_col(self):
        return [x for x in self.data.columns if "cycle" in x or "active power output" in x]

    def get_stacked_shift_col(self):
        return [x for x in self.get_predictor_cols() if "cycle" not in x]


if __name__ == "__main__":
    # dd = GEFCom2014SingleZoneDataSet(task_num=1, zone_num=1, dataset=get_data_for_a_task(1)["training set"])
    dd = GEFCom2014AllZoneDataSet(task_num=1, dataset=get_data_for_a_task(1)["training set"])
