source_code = """
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

cwd = Path().cwd().parent
spec = spec_from_file_location("TSE_SI_2020", cwd / 'TSE_SI_2020/stage_3.py')
foo = module_from_spec(spec)
spec.loader.exec_module(foo)
i = 1
j = 17
m = i * j
foo.cal_final_results_for_one_wf('Bruska', idx_s=m, idx_e=m + j, use_corr_impute='_cluster_')
"""
if __name__ == '__main__':
    exec(source_code)
