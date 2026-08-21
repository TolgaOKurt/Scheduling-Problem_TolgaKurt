import sys, os
sys.path.insert(0, os.path.abspath('.'))
import pulp
import numpy as np
from algorithms.worker_manager import generate_worker_profiles

nw = 24
nd = 7
rd = 4
re = 4
rn = 4
weights = {'circadian': 50, 'night_imb': 25, 'workload_imb': 20, 'exp_mix': 60, 'pref_off': 40, 'posta': 15}
workers = generate_worker_profiles(nw, nd, randomize=False)

# Let's inspect the ILP model vs LP model
# Where does 766 come from?
from algorithms.ilp_pulp_solver import solve_ilp_pulp, compute_lp_relaxation_bound

res = solve_ilp_pulp(nw, nd, rd, re, rn, weights, time_limit=15, custom_workers=workers)
print("ILP solver result:", res['final_score'])
print("ILP solver penalties:", res['penalties'])

bound_val, breakdown, reasons, stage_info = compute_lp_relaxation_bound(nw, nd, rd, re, rn, weights, custom_workers=workers, return_stages=True)
print("\nBound val:", bound_val)
print("Stage info:", stage_info)
print("Breakdown:", breakdown)
