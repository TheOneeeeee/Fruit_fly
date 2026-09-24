import sys
from pathlib import Path
sys.path.insert(0, "code/paper-phil-drosophila")

from model import run_exp, default_params
from brian2 import Hz, ms
from stimuli import STIMULI

fear = STIMULI["fear"]

params = default_params.copy()
params['r_poi'] = fear['hz'] * Hz
params['t_run'] = 10000 * ms
params['n_run'] = 1

path_res = Path("data/results")
path_res.mkdir(exist_ok=True)

run_exp(
    exp_name='Fear_150Hz',
    neu_exc=fear['ids'],
    path_res=path_res,
    path_comp="data/2025_Completeness_783.csv",
    path_con="data/2025_Connectivity_783.parquet",
    params=params,
    n_proc=1,
)