import sys, os
sys.path.insert(0, os.path.abspath('.'))
from algorithms.worker_manager import generate_worker_profiles
from algorithms.cp_sat_solver import solve_cp_sat

nw = 24
nd = 7
rd = 4
re = 4
rn = 4
weights = {'circadian': 50, 'night_imb': 25, 'workload_imb': 20, 'exp_mix': 60, 'pref_off': 40, 'posta': 15}
workers = generate_worker_profiles(nw, nd, randomize=False)

res = solve_cp_sat(nw, nd, rd, re, rn, weights, time_limit=5, custom_workers=workers)
print(f"CP-SAT: final_score = {res['final_score']}")
print(f"CP-SAT: best_bound = {res['meta']['best_bound']}")
print(f"CP-SAT: mip_gap = {res['meta']['mip_gap']}%")
print(f"CP-SAT: is_optimal = {res['meta']['is_optimal']}")
assert res['meta']['best_bound'] <= res['final_score'], f"best_bound {res['meta']['best_bound']} > final_score {res['final_score']}"
print("\n>>> CP-SAT BOUND TEST PASSED PERFECTLY! <<<")
