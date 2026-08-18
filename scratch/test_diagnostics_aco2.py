import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath("."))

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import get_best_greedy_initial_solution
from algorithms.penalty_calculator import check_swap_feasibility, audit_all_hard_constraints

n_workers = 28
n_days = 28
r_day = 12
r_eve = 8
r_night = 4
shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
weights = {'w_req': 20, 'w_seq': 30, 'w_night': 15, 'w_equity': 10, 'w_rest': 50, 'w_lead': 40}
workers = generate_worker_profiles(n_workers, n_days, randomize=False)

seed = get_best_greedy_initial_solution(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers)
sched = seed['schedule'].copy()

is_feab, _, logs = audit_all_hard_constraints(sched, workers, n_workers, n_days, shift_reqs)
print(f"Seed feasible: {is_feab}")

# Now test swap
d = 3
w1, w2 = 0, 1
s1, s2 = sched[w1, d], sched[w2, d]
print(f"Before swap: w1={s1}, w2={s2}")

sched[w1, d], sched[w2, d] = s2, s1
val = check_swap_feasibility(sched, workers, d, w1, w2, n_workers, n_days, shift_reqs)
print(f"Check swap feasibility returned: {val}")

is_feab, _, logs = audit_all_hard_constraints(sched, workers, n_workers, n_days, shift_reqs)
print(f"After swap feasible according to audit: {is_feab}")
