source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd()
spec = spec_from_file_location("TSE2020", cwd / 'TSE_SI_2020/operating_regime_data_prepare.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.get_jelinak_operating_regime_estimation()
"""
if __name__ == '__main__':
    exec(source_code)
