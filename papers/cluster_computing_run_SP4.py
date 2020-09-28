source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd()
spec = spec_from_file_location("TSE2020", cwd / 'TSE2020.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.fit_or_analyse_darly_wind_farm_power_curve_model_without_known_wind_turbines(task='fit', ga_max_num_iteration=400)
"""
if __name__ == '__main__':
    exec(source_code)
