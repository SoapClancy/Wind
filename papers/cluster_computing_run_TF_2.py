source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd()
spec = spec_from_file_location("TSE_SI_2020", cwd / 'TSE_SI_2020/stage_3.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
foo.cal_final_results_for_one_wf('Jelinak', idx_s=0, idx_e=100)

"""
if __name__ == '__main__':
    exec(source_code)
