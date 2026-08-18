import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

from algorithms.worker_manager import generate_worker_profiles
from algorithms.aco_solver import run_ant_colony_optimization

print("--- TESTING ACO SOLVER WITH GLOBAL DEFAULT PARAMETERS ---")
n_workers = 28
n_days = 7
r_day = 9
r_eve = 7
r_night = 5
weights = {
    'circadian': 50,
    'night_imb': 25,
    'exp_mix': 30,
    'pref_off': 40,
    'posta': 15
}

workers = generate_worker_profiles(n_workers, n_days, randomize=False)

start_time = time.time()
res = run_ant_colony_optimization(
    n_workers=n_workers,
    n_days=n_days,
    r_day=r_day,
    r_eve=r_eve,
    r_night=r_night,
    weights=weights,
    n_ants=20,
    max_iterations=50,
    evaporation_rate=0.15,
    alpha=1.0,
    beta=2.0,
    seed=42,
    custom_workers=workers
)
elapsed = round(time.time() - start_time, 2)

print(f"ACO Executed in {elapsed}s (Solver reported {res['exec_time_ms']} ms)")
print(f"Feasible: {res['is_feasible']} (Hard violations: {res['hard_violations_count']})")
print(f"Initial Score: {res['initial_score']} -> Final Score: {res['final_score']} (Improvement: {res['improvement_rate']}%)")
print(f"Total Evaluations: {res['eval_count']}, Total Iterations: {res['total_iterations']}")
print(f"Active Pheromone Matrix Shape: {len(res['meta']['active_pheromone_matrix'])}x{len(res['meta']['active_pheromone_matrix'][0])}")

# Check standard contract keys
expected_keys = [
    'schedule', 'workers', 'is_feasible', 'hard_violations_count', 'hard_violation_logs',
    'final_score', 'initial_score', 'improvement_rate', 'exec_time_ms', 'eval_count',
    'total_iterations', 'termination_reason', 'penalties', 'request_details', 'score_history', 'meta'
]
for k in expected_keys:
    assert k in res, f"Missing key: {k}"

assert res['is_feasible'] == True, "Schedule must be feasible!"
assert res['final_score'] <= res['initial_score'], "ACO should not degrade score!"
print("✅ ACO SOLVER VALIDATION WITH GLOBAL DEFAULTS PASSED!")

print("\n--- TESTING 28 DAYS SCENARIO ---")
n_days_28 = 28
workers_28 = generate_worker_profiles(n_workers, n_days_28, randomize=False)
res_28 = run_ant_colony_optimization(
    n_workers=n_workers,
    n_days=n_days_28,
    r_day=r_day,
    r_eve=r_eve,
    r_night=r_night,
    weights=weights,
    n_ants=15,
    max_iterations=30,
    evaporation_rate=0.15,
    alpha=1.0,
    beta=2.0,
    seed=42,
    custom_workers=workers_28
)
print(f"28-Day Scenario - Feasible: {res_28['is_feasible']}, Final Score: {res_28['final_score']}")
assert res_28['is_feasible'] == True, "28-Day schedule must be feasible!"
print("✅ 28-DAY SCENARIO PASSED!")
