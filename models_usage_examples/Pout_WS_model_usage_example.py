from Ploting.fast_plot_Func import *
from prepare_datasets import load_raw_wt_from_txt_file_and_temperature_from_csv
from WT_WF_Class import WT
from File_Management.path_and_file_management_Func import remove_win10_max_path_limit

"""
This (remove_win10_max_path_limit) is a default call to make the code run on Windows platform 
(and only Windows 10 is supported!)
Because the Windows 10 default path length limitation (MAX_PATH) is 256 characters, many load/save functions in
this project may have errors in reading the path
To restore the path_limit, call File_Management.path_management_Func.restore_win10_max_path_limit yourself
"""
remove_win10_max_path_limit()


def train_and_load_wind_turbine_pout_ws_model():
    """
    train wind turbine models。this function should be called before anything
    """
    wind_turbines = load_raw_wt_from_txt_file_and_temperature_from_csv()
    for this_wind_turbine in wind_turbines:
        this_wind_turbine.fit_2d_conditional_probability_model_by_gmm(bin_step=0.5)
    return wind_turbines


def to_asad_wind_turbine_pout_ws_model_example(considered_wind_turbine_no: int) -> tuple:
    """
    a simple example of calling "estimate_active_power_output_by_2d_conditional_probability_model_by_gmm" of WT instance

    :param considered_wind_turbine_no: Specify the No. of WT to be considered, range from 0 to 5,
    e.g., "considered_wind_turbine_no = 0" is the first WT,
    and "considered_wind_turbine_no = 1" is the second WT,
    and so on

    :return: tuple，
    element 0 is the probabilistic power output distributions, type UnivariateGaussianMixtureModel,
    refer to UnivariateGaussianMixtureModel in UnivariateAnalysis_Class.py (asset of Python_Project_common_package)

    and element 1 is the actual measurements of wind speed in 10 min [m/s], type pandas.DataFrame,
    refer to the official docs of Pandas
    """
    this_wind_turbine = train_and_load_wind_turbine_pout_ws_model()[considered_wind_turbine_no]  # type: WT
    _actual_wind_speed_data = this_wind_turbine.measurements['wind speed'].values
    _probabilistic_power_output_data = \
        this_wind_turbine.estimate_active_power_output_by_2d_conditional_probability_model_by_gmm(
            _actual_wind_speed_data,
            bin_step=0.5
        )
    return _probabilistic_power_output_data, this_wind_turbine.measurements[['time', 'wind speed']]


if __name__ == '__main__':
    train_and_load_wind_turbine_pout_ws_model()
    """
    As'ad, you should choose "considered_wind_turbine_no" for "to_asad_wind_turbine_pout_ws_model_example" call yourself
    
    please read the docstring of "to_asad_wind_turbine_pout_ws_model_example" function. 
    
    type(probabilistic_power_output_data[int]) is UnivariateAnalysis_Class.UnivariateGaussianMixtureModel,
    you could get all implemented statistics from it directly,
    e.g., mean, percentiles
    """
    probabilistic_power_output_data, actual_wind_speed_data = to_asad_wind_turbine_pout_ws_model_example(0)
    # example of getting mean of probabilistic_power_output_data[0]
    print(probabilistic_power_output_data[0].mean_)
    # example of getting 95-th percentile of probabilistic_power_output_data[0]
    print(probabilistic_power_output_data[0].find_nearest_inverse_cdf(0.95))
    # example of getting the PDF of probabilistic_power_output_data[0]
    probabilistic_power_output_data[0].plot_pdf(np.arange(0, 3030),
                                                x_label="Power output (kW)")
    # example of getting the CDF of probabilistic_power_output_data[0]
    probabilistic_power_output_data[0].plot_cdf(np.arange(0, 3030),
                                                x_label="Power output (kW)",
                                                y_lim=(-0.01, 1.02))



