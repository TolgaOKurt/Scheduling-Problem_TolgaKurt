import sys
import os
sys.path.insert(0, os.path.abspath("."))

from algorithms.worker_manager import generate_worker_profiles
from algorithms.aco_solver import run_ant_colony_optimization

n_workers = 28
n_days = 28
r_day = 12
r_eve = 8
r_night = 4
weights = {'w_req': 20, 'w_seq': 30, 'w_night': 15, 'w_equity': 10, 'w_rest': 50, 'w_lead': 40}
workers = generate_worker_profiles(n_workers, n_days, randomize=False)

res = run_ant_colony_optimization(
    n_workers=n_workers,
    n_days=n_days,
    r_day=r_day,
    r_eve=r_eve,
    r_night=r_night,
    weights=weights,
    n_ants=5,
    max_iterations=10,
    seed=42,
    custom_workers=workers
)

print(f"Feasible: {res['is_feasible']}")
print(f"Violations count: {res['hard_violations_count']}")
for log in res['hard_violation_logs']:
    print(f"  - {log.encode('ascii', 'replace').decode('ascii')}")
