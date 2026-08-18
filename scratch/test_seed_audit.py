import sys
import os
sys.path.insert(0, os.path.abspath("."))

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import get_best_greedy_initial_solution
from algorithms.penalty_calculator import audit_all_hard_constraints

n_workers = 28
n_days = 28
r_day = 12
r_eve = 8
r_night = 4
weights = {'w_req': 20, 'w_seq': 30, 'w_night': 15, 'w_equity': 10, 'w_rest': 50, 'w_lead': 40}
shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
workers = generate_worker_profiles(n_workers, n_days, randomize=False)

seed = get_best_greedy_initial_solution(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers)
sched = seed['schedule'].copy()

print(f"Seed reported is_feasible: {seed.get('is_feasible')}")
is_feab, count, logs = audit_all_hard_constraints(sched, workers, n_workers, n_days, shift_reqs)
print(f"Audit is_feasible: {is_feab}, count: {count}")
for log in logs:
    print("  LOG:", log.encode('ascii', 'replace').decode('ascii'))
