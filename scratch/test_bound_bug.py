import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import global_state
from algorithms.worker_manager import generate_worker_profiles
from algorithms.ilp_pulp_solver import compute_lp_relaxation_bound, solve_ilp_pulp

# Reproduce user's case: N=24, D=7, 4-4-4 (Fırın Uyutma / Bakım preset or 24 workers)
nw = 24
nd = 7
rd = 4
re = 4
rn = 4
weights = {'circadian': 50, 'night_imb': 25, 'workload_imb': 20, 'exp_mix': 60, 'pref_off': 40, 'posta': 15}
workers = generate_worker_profiles(nw, nd, randomize=False)

print("Solving ILP...")
res_ilp = solve_ilp_pulp(nw, nd, rd, re, rn, weights, time_limit=10, custom_workers=workers)
print(f"ILP Result: final_score = {res_ilp['final_score']}, is_optimal = {res_ilp.get('is_optimal')}")
print(f"ILP Penalties breakdown: {res_ilp['penalties']}")

print("\nComputing LP Relaxation Bound...")
bound_val, breakdown, reasons, stage_info = compute_lp_relaxation_bound(nw, nd, rd, re, rn, weights, custom_workers=workers, return_stages=True)
print(f"Computed bound_val = {bound_val}")
print(f"stage1_analytical = {stage_info['stage1_analytical']}")
print(f"stage2_continuous_lp = {stage_info['stage2_continuous_lp']}")
print(f"Breakdown = {breakdown}")

print(f"\nIS BOUND VALID? (bound_val <= res_ilp['final_score']): {bound_val <= res_ilp['final_score']}")
if bound_val > res_ilp['final_score']:
    print(f"BUG DETECTED: bound_val ({bound_val}) > integer optimal ({res_ilp['final_score']})!")
