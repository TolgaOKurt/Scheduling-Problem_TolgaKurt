import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from algorithms.memetic_algorithm_solver import run_memetic_algorithm
from algorithms.genetic_algorithm_solver import run_genetic_algorithm
from algorithms.worker_manager import generate_worker_profiles

weights = {'posta': 35, 'circadian': 50, 'pref_off': 40, 'night_imb': 25, 'exp_mix': 30}

print("Testing 100 seeds for 12 workers, 7 days...")
infeasible_ma = []
for seed in range(1, 101):
    res_ma = run_memetic_algorithm(12, 7, 2, 2, 2, weights, seed=seed)
    if not res_ma['is_feasible']:
        print(f"Memetic Seed {seed} INFEASIBLE: count={res_ma['hard_violations_count']}")
        for log in res_ma['hard_violation_logs']:
            print(f"   -> {log}")
        infeasible_ma.append(seed)

print(f"\nTotal Memetic Infeasible Seeds: {len(infeasible_ma)} / 100")
