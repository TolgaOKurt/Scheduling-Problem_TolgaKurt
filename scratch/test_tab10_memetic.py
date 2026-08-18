import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from algorithms.worker_manager import generate_worker_profiles
from algorithms.memetic_algorithm_solver import run_memetic_algorithm
from algorithms.genetic_algorithm_solver import run_genetic_algorithm
from algorithms.greedy_solver import get_best_greedy_initial_solution

n_workers = 28
n_days = 7
r_day = 9
r_eve = 7
r_night = 5
weights = {'circadian': 50, 'night_imb': 25, 'exp_mix': 30, 'pref_off': 40, 'posta': 15}

workers = generate_worker_profiles(n_workers, n_days, randomize=False)

print("1. Testing Best Greedy Seed:")
seed_res = get_best_greedy_initial_solution(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers)
print(f"Seed feasible: {seed_res['is_feasible']}, Hard viols: {seed_res['hard_violations_count']}")

print("\n2. Testing 50 runs of Memetic Algorithm:")
infeasible_count = 0
for s in range(1, 51):
    res = run_memetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, seed=s, custom_workers=workers)
    if not res['is_feasible']:
        infeasible_count += 1
        print(f"Seed {s} Infeasible ({res['hard_violations_count']} violations): {res['hard_violation_logs']}")

print(f"\nMemetic Algorithm Infeasible runs: {infeasible_count} / 50")
