source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd().parent
spec = spec_from_file_location("TSE2020", cwd / 'TSE2020.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.individual_wind_turbine_outliers_outlier_detector('load',
                                                      wt_index=tuple(range(1, 6)),
                                                      write_to_a_docx=False)
"""
if __name__ == '__main__':
    exec(source_code)
