source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd()
spec = spec_from_file_location("TSE2020", cwd / 'TSE2020.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.individual_wind_turbine_outliers_outlier_detector(power_curve_model_task = 'fit',
                                                       wt_index = (0,1))
"""
if __name__ == '__main__':
    exec(source_code)
