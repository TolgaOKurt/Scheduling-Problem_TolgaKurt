import sys, os
sys.path.insert(0, os.path.abspath('.'))
import global_state
from algorithms.greedy_solver import run_greedy_algorithm
from algorithms.csp_backtracking_solver import CSPBacktrackingSolver
from algorithms.ilp_pulp_solver import solve_ilp_pulp, compute_lp_relaxation_bound
from algorithms.cp_sat_solver import solve_cp_sat
from algorithms.hill_climbing_solver import run_hill_climbing
from algorithms.simulated_annealing_solver import run_simulated_annealing
from algorithms.genetic_algorithm_solver import run_genetic_algorithm
from algorithms.memetic_algorithm_solver import run_memetic_algorithm
from algorithms.tabu_search_solver import run_tabu_search
from algorithms.vns_solver import run_variable_neighborhood_search
from algorithms.pso_solver import run_particle_swarm_optimization
from algorithms.aco_solver import run_ant_colony_optimization

print("=" * 70)
print("TEST: ALL 13 SOLVERS WITH 6 SOFT CONSTRAINTS (INCLUDING WORKLOAD IMBALANCE)")
print("=" * 70)

params = global_state.get_global_params()
nw = params['n_workers']
nd = params['n_days']
rd = params['r_day']
re = params['r_eve']
rn = params['r_night']
weights = params['weights']
workers = params['custom_workers']

print(f"Config: N={nw}, D={nd}, Shifts={rd}-{re}-{rn}, Weights={weights}")

solvers = [
    ("Greedy Myopic", lambda: run_greedy_algorithm(nw, nd, rd, re, rn, weights, solver_mode="1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)", custom_workers=workers)),
    ("Greedy Staggered", lambda: run_greedy_algorithm(nw, nd, rd, re, rn, weights, solver_mode="2. Kademeli / Desen Tabanlı Yapıcı Sezgisel (Staggered Pattern-Based Greedy)", custom_workers=workers)),
    ("Greedy MRV-LCV", lambda: run_greedy_algorithm(nw, nd, rd, re, rn, weights, solver_mode="3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)", custom_workers=workers)),
    ("CSP Backtracking", lambda: CSPBacktrackingSolver(nw, nd, rd, re, rn, max_backtracks=2000, custom_workers=workers, weights=weights).solve()),
    ("ILP / MILP (PuLP)", lambda: solve_ilp_pulp(nw, nd, rd, re, rn, weights, time_limit=5, custom_workers=workers)),
    ("CP-SAT (OR-Tools)", lambda: solve_cp_sat(nw, nd, rd, re, rn, weights, time_limit=3, custom_workers=workers)),
    ("Hill Climbing", lambda: run_hill_climbing(nw, nd, rd, re, rn, weights, max_iterations=200, custom_workers=workers)),
    ("Simulated Annealing", lambda: run_simulated_annealing(nw, nd, rd, re, rn, weights, max_iterations=200, custom_workers=workers)),
    ("Genetic Algorithm", lambda: run_genetic_algorithm(nw, nd, rd, re, rn, weights, pop_size=20, generations=20, custom_workers=workers)),
    ("Memetic Algorithm", lambda: run_memetic_algorithm(nw, nd, rd, re, rn, weights, pop_size=20, generations=20, custom_workers=workers)),
    ("Tabu Search", lambda: run_tabu_search(nw, nd, rd, re, rn, weights, max_iterations=100, custom_workers=workers)),
    ("VNS Solver", lambda: run_variable_neighborhood_search(nw, nd, rd, re, rn, weights, max_iterations=100, custom_workers=workers)),
    ("PSO Solver", lambda: run_particle_swarm_optimization(nw, nd, rd, re, rn, weights, swarm_size=15, max_iterations=30, custom_workers=workers)),
    ("ACO Solver", lambda: run_ant_colony_optimization(nw, nd, rd, re, rn, weights, n_ants=10, max_iterations=20, custom_workers=workers))
]

all_passed = True
for name, solver_fn in solvers:
    try:
        res = solver_fn()
        feas = res.get('is_feasible', False)
        score = res.get('final_score', None)
        pen_dict = res.get('penalties', {})
        workload_p = pen_dict.get('Toplam İş Yükü Dengesizliği', None)
        
        status = "PASSED (Feasible)" if feas else "FAILED (Infeasible)"
        print(f"[{status}] {name:<22} -> Score: {score} | Workload Penalty: {workload_p} | Keys: {len(pen_dict)}")
        if not feas or workload_p is None:
            all_passed = False
    except Exception as e:
        print(f"[ERROR]  {name:<22} -> Exception: {e}")
        all_passed = False

# Also check LP relaxation bound
bound_val, breakdown, reasons = compute_lp_relaxation_bound(nw, nd, rd, re, rn, weights, custom_workers=workers, return_breakdown=True)
print(f"\n[LP Relaxation Best Bound] -> Value: {bound_val} | Breakdown: {breakdown}")

assert all_passed, "Some solvers failed or missed the workload penalty key!"
print("\n>>> ALL 13 SOLVERS PASSED 100% WITH 6 SOFT CONSTRAINTS! <<<")
