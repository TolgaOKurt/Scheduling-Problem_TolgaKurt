import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
from algorithms.worker_manager import generate_worker_profiles
from algorithms.penalty_calculator import calculate_full_penalties, build_fast_evaluator

print("=" * 60)
print("TEST: WORKLOAD IMBALANCE PENALTY CALCULATION")
print("=" * 60)

n_workers = 4
n_days = 2
workers = generate_worker_profiles(n_workers, n_days, randomize=False)
weights = {'circadian': 50, 'night_imb': 25, 'workload_imb': 20, 'exp_mix': 60, 'pref_off': 40, 'posta': 15}

# Case 1: Perfectly balanced schedule (each worker works exactly 1 shift)
# Worker 0: Day 1 (1), Day 2 (0) -> 1
# Worker 1: Day 1 (2), Day 2 (0) -> 1
# Worker 2: Day 1 (0), Day 2 (1) -> 1
# Worker 3: Day 1 (0), Day 2 (2) -> 1
sched_balanced = np.array([
    [1, 0],
    [2, 0],
    [0, 1],
    [0, 2]
])
pen_bal, tot_bal = calculate_full_penalties(sched_balanced, workers, n_workers, n_days, weights)
print("Balanced Workload Penalty:", pen_bal.get('Toplam İş Yükü Dengesizliği'))
assert pen_bal.get('Toplam İş Yükü Dengesizliği') == 0, f"Expected 0, got {pen_bal.get('Toplam İş Yükü Dengesizliği')}"

# Case 2: Highly imbalanced schedule
# Worker 0 works both days (2 shifts)
# Worker 1 works both days (2 shifts)
# Worker 2 works 0 shifts
# Worker 3 works 0 shifts
# Total shifts = 4. Average = 1.0.
# Deviations: |2-1| + |2-1| + |0-1| + |0-1| = 1 + 1 + 1 + 1 = 4.
# Penalty = 4 * 20 = 80.
sched_imbal = np.array([
    [1, 1],
    [2, 2],
    [0, 0],
    [0, 0]
])
pen_imbal, tot_imbal = calculate_full_penalties(sched_imbal, workers, n_workers, n_days, weights)
print("Imbalanced Workload Penalty:", pen_imbal.get('Toplam İş Yükü Dengesizliği'))
assert pen_imbal.get('Toplam İş Yükü Dengesizliği') == 80, f"Expected 80, got {pen_imbal.get('Toplam İş Yükü Dengesizliği')}"

# Fast evaluator check
fast_eval = build_fast_evaluator(workers, n_days, weights)
fast_score_bal = fast_eval(sched_balanced)
fast_score_imbal = fast_eval(sched_imbal)
print("Fast Evaluator Scores -> Balanced:", fast_score_bal, "| Imbalanced:", fast_score_imbal)
assert fast_score_bal == tot_bal, f"Fast evaluator score {fast_score_bal} != full penalty score {tot_bal}"
assert fast_score_imbal == tot_imbal, f"Fast evaluator score {fast_score_imbal} != full penalty score {tot_imbal}"

print("\n>>> ALL WORKLOAD PENALTY UNIT TESTS PASSED! <<<")
