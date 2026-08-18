import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm
from algorithms.csp_backtracking_solver import CSPBacktrackingSolver
from algorithms.ilp_pulp_solver import solve_ilp_pulp, compute_lp_relaxation_bound
from algorithms.hill_climbing_solver import run_hill_climbing
from algorithms.simulated_annealing_solver import run_simulated_annealing
from algorithms.genetic_algorithm_solver import run_genetic_algorithm
from algorithms.memetic_algorithm_solver import run_memetic_algorithm
from algorithms.tabu_search_solver import run_tabu_search
from algorithms.vns_solver import run_variable_neighborhood_search
from algorithms.pso_solver import run_particle_swarm_optimization
from algorithms.aco_solver import run_ant_colony_optimization

print("=== RUNNING FULL 13-SOLVER BENCHMARK TEST ===")
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

solvers = [
    ("Greedy 1: Myopic", lambda: run_greedy_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, solver_mode="1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)", custom_workers=workers)),
    ("Greedy 2: Staggered", lambda: run_greedy_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, solver_mode="2. Kademeli / Desen Tabanlı Yapıcı Sezgisel (Staggered Pattern-Based Greedy)", custom_workers=workers)),
    ("Greedy 3: MRV/LCV", lambda: run_greedy_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, solver_mode="3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)", custom_workers=workers)),
    ("CSP Backtracking", lambda: CSPBacktrackingSolver(n_workers, n_days, r_day, r_eve, r_night, max_backtracks=5000, custom_workers=workers, weights=weights).solve()),
    ("ILP / MILP", lambda: solve_ilp_pulp(n_workers, n_days, r_day, r_eve, r_night, weights, time_limit=5, custom_workers=workers)),
    ("Hill Climbing", lambda: run_hill_climbing(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=500, seed=42, custom_workers=workers)),
    ("Simulated Annealing", lambda: run_simulated_annealing(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=500, seed=42, custom_workers=workers)),
    ("Genetic Algorithm", lambda: run_genetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, pop_size=20, generations=20, seed=42, custom_workers=workers)),
    ("Memetic Algorithm", lambda: run_memetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, pop_size=15, generations=15, seed=42, custom_workers=workers)),
    ("Tabu Search", lambda: run_tabu_search(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=200, seed=42, custom_workers=workers)),
    ("VNS", lambda: run_variable_neighborhood_search(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=200, seed=42, custom_workers=workers)),
    ("Discrete PSO", lambda: run_particle_swarm_optimization(n_workers, n_days, r_day, r_eve, r_night, weights, swarm_size=15, max_iterations=20, seed=42, custom_workers=workers)),
    ("Ant Colony (ACO)", lambda: run_ant_colony_optimization(n_workers, n_days, r_day, r_eve, r_night, weights, n_ants=15, max_iterations=25, seed=42, custom_workers=workers)),
]

for name, fn in solvers:
    t0 = time.time()
    res = fn()
    dur = round(time.time() - t0, 3)
    f_score = res.get('final_score')
    feas = res.get('is_feasible')
    print(f"[{name:20}] Score: {f_score:5} | Feasible: {str(feas):5} | Time: {dur}s")
    assert res is not None
    assert 'schedule' in res
    assert 'final_score' in res

print("\nALL 13 SOLVERS TESTED AND COMPLETED SUCCESSFULLY!")
