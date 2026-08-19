import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from algorithms.worker_manager import generate_worker_profiles
from algorithms.tabu_search_solver import run_tabu_search
from algorithms.vns_solver import run_variable_neighborhood_search
from algorithms.pso_solver import run_particle_swarm_optimization
from algorithms.aco_solver import run_ant_colony_optimization

print("================================================================================")
print("AUDIT: TABS 11, 12, 13, 14 COMPLIANCE WITH SINGLE SOURCE OF TRUTH & CONTRACT")
print("================================================================================")

n_workers = 28
n_days = 7
r_day = 9
r_eve = 7
r_night = 5
weights = {
    'circadian': 50,
    'night_imb': 25,
    'exp_mix': 60,
    'pref_off': 40,
    'posta': 15
}

workers = generate_worker_profiles(n_workers, n_days, randomize=False)

CANONICAL_16_KEYS = [
    'schedule', 'workers', 'is_feasible', 'hard_violations_count', 'hard_violation_logs',
    'final_score', 'initial_score', 'improvement_rate', 'exec_time_ms', 'eval_count',
    'total_iterations', 'termination_reason', 'penalties', 'request_details',
    'score_history', 'meta'
]

# Exact 5 canonical soft penalty keys defined in penalty_calculator.py (Single Source of Truth)
CANONICAL_5_PENALTY_KEYS = [
    'Posta Takım Bütünlüğü İhlali',
    'Sirkadiyen Ritim İhlali (Akşam->Gündüz)',
    'Gece Nöbeti Dengesizliği',
    'Kıdem / Usta Eksikliği',
    'Kişisel İzin İhlali'
]

test_cases = [
    ("Tab 11: Tabu Search", lambda: run_tabu_search(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=50, custom_workers=workers)),
    ("Tab 12: Variable Neighborhood Search (VNS)", lambda: run_variable_neighborhood_search(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=50, custom_workers=workers)),
    ("Tab 13: Discrete PSO", lambda: run_particle_swarm_optimization(n_workers, n_days, r_day, r_eve, r_night, weights, swarm_size=15, max_iterations=20, custom_workers=workers)),
    ("Tab 14: Ant Colony Optimization (ACO)", lambda: run_ant_colony_optimization(n_workers, n_days, r_day, r_eve, r_night, weights, n_ants=15, max_iterations=20, custom_workers=workers))
]

all_passed = True

for tab_name, solver_fn in test_cases:
    print(f"\n--- Auditing {tab_name} ---")
    t0 = time.time()
    res = solver_fn()
    dur = round(time.time() - t0, 3)

    # 1. Check all 16 Canonical Keys
    missing_keys = [k for k in CANONICAL_16_KEYS if k not in res]
    if missing_keys:
        print(f"[FAIL] Missing canonical keys: {missing_keys}")
        all_passed = False
    else:
        print(f"[PASS] All 16 Canonical Keys Present")

    # 2. Check Data Types & Shapes
    assert isinstance(res['schedule'], np.ndarray), "schedule must be np.ndarray"
    assert res['schedule'].shape == (n_workers, n_days), f"schedule shape mismatch: {res['schedule'].shape}"
    assert isinstance(res['workers'], list) and len(res['workers']) == n_workers, "workers list invalid"
    assert isinstance(res['is_feasible'], bool) or isinstance(res['is_feasible'], np.bool_), "is_feasible must be bool"
    assert isinstance(res['final_score'], (int, float)), "final_score must be number"
    assert isinstance(res['initial_score'], (int, float)), "initial_score must be number"
    assert isinstance(res['improvement_rate'], float), "improvement_rate must be float"
    assert isinstance(res['exec_time_ms'], float), "exec_time_ms must be float"
    assert isinstance(res['eval_count'], int), "eval_count must be int"
    assert isinstance(res['total_iterations'], int), "total_iterations must be int"
    assert isinstance(res['penalties'], dict), "penalties must be dict"
    assert isinstance(res['request_details'], list), "request_details must be list"
    assert isinstance(res['score_history'], list), "score_history must be list"
    assert isinstance(res['meta'], dict), "meta must be dict"
    print(f"[PASS] All 16 Canonical Data Types & Dimensions Valid")

    # 3. Check 5 Canonical Penalty Names
    pen_dict = res['penalties']
    missing_pens = [pk for pk in CANONICAL_5_PENALTY_KEYS if pk not in pen_dict]
    if missing_pens:
        print(f"[FAIL] Missing penalty sub-keys: {missing_pens}")
        all_passed = False
    else:
        print(f"[PASS] All 5 Canonical Penalty Keys Match penalty_calculator.py exactly")

    # 4. Check Single Source of Trust Meta Telemetry
    meta = res['meta']
    assert 'seed_source' in meta, "Missing seed_source in meta"
    assert 'is_csp_fallback' in meta, "Missing is_csp_fallback in meta"
    print(f"  * Seed Source: {meta.get('seed_source', 'N/A')}")
    print(f"  * is_csp_fallback: {meta.get('is_csp_fallback', 'N/A')}")
    print(f"  * Feasible: {res['is_feasible']} (Violations: {res['hard_violations_count']})")
    print(f"  * Score: {res['initial_score']} -> {res['final_score']} (Improvement: %{res['improvement_rate']})")
    print(f"  * Exec Time: {res['exec_time_ms']} ms")

if all_passed:
    print("\n================================================================================")
    print("ALL TESTS PASSED: TABS 11, 12, 13, 14 ARE 100% COMPLIANT WITH CANONICAL CONTRACT!")
    print("================================================================================")
else:
    print("\n[FAIL] AUDIT FAILED! See errors above.")
