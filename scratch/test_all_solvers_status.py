import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm
from algorithms.csp_backtracking_solver import CSPBacktrackingSolver
from algorithms.ilp_pulp_solver import solve_ilp_pulp
from algorithms.hill_climbing_solver import run_hill_climbing
from algorithms.simulated_annealing_solver import run_simulated_annealing
from algorithms.genetic_algorithm_solver import run_genetic_algorithm
from algorithms.memetic_algorithm_solver import run_memetic_algorithm
from algorithms.tabu_search_solver import run_tabu_search
from algorithms.vns_solver import run_vns
from algorithms.pso_solver import run_particle_swarm_optimization
from algorithms.aco_solver import run_ant_colony_optimization

n_workers, n_days, r_day, r_eve, r_night = 28, 7, 9, 7, 5
weights = {'circadian': 50, 'night_imb': 25, 'exp_mix': 30, 'pref_off': 40, 'posta': 15}
workers = generate_worker_profiles(n_workers, n_days, randomize=False)

solvers = [
    ('Greedy (Yapıcı)', lambda: run_greedy_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers)),
    ('CSP Backtracking', lambda: CSPBacktrackingSolver(n_workers, n_days, r_day, r_eve, r_night, weights=weights, custom_workers=workers).solve()),
    ('ILP / MILP (PuLP)', lambda: solve_ilp_pulp(n_workers, n_days, r_day, r_eve, r_night, weights, time_limit=5, custom_workers=workers)),
    ('Hill Climbing', lambda: run_hill_climbing(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers)),
    ('Simulated Annealing', lambda: run_simulated_annealing(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers)),
    ('Genetic Algorithm', lambda: run_genetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, pop_size=20, generations=20, custom_workers=workers)),
    ('Memetic Algorithm', lambda: run_memetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, pop_size=20, generations=20, custom_workers=workers)),
    ('Tabu Search', lambda: run_tabu_search(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=20, custom_workers=workers)),
    ('VNS (Değişken Komşuluk)', lambda: run_vns(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=20, custom_workers=workers)),
    ('PSO (Sürü Zekası)', lambda: run_particle_swarm_optimization(n_workers, n_days, r_day, r_eve, r_night, weights, swarm_size=15, max_iterations=20, custom_workers=workers)),
    ('ACO (Karınca Kolonisi)', lambda: run_ant_colony_optimization(n_workers, n_days, r_day, r_eve, r_night, weights, n_ants=15, max_iterations=20, custom_workers=workers))
]

print(f"{'Algoritma':<25} | {'Feasible':<8} | {'İhlal':<5} | {'Toplam Z':<8} | {'Posta':<6} | {'Sirkad.':<7} | {'Gece':<5} | {'Kıdem':<5} | {'İzin':<5}")
print("-" * 90)
for name, fn in solvers:
    res = fn()
    p = res['penalties']
    is_feas = str(res.get('is_feasible', False))
    viols = res.get('hard_violations_count', 0)
    score = res.get('final_score', -1)
    p_posta = p.get("Posta Takım Bütünlüğü İhlali", 0)
    p_circ = p.get("Sirkadiyen Ritim İhlali (Akşam->Gündüz)", 0)
    p_night = p.get("Gece Nöbeti Dengesizliği", 0)
    p_exp = p.get("Kıdem & MYK Sertifika Eksikliği", 0)
    p_pref = p.get("Kişisel İzin İhlali", 0)
    print(f"{name:<25} | {is_feas:<8} | {viols:>5} | {score:>8} | {p_posta:>6} | {p_circ:>7} | {p_night:>5} | {p_exp:>5} | {p_pref:>5}")

print("-" * 90)
