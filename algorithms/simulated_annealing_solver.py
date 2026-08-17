"""
================================================================================
  ALGORITHMS/SIMULATED_ANNEALING_SOLVER.PY - SIMULATED ANNEALING (TAVLAMA BENZETİMİ)
================================================================================
  Bu modül, Simulated Annealing (Tavlama Benzetimi) stokastik metasezgisel algoritmasını
  içerir. Algoritma sonlandığında 'termination_reason' (Sonlandırma Nedeni) bilgisini tam
  ve net olarak raporlar.
================================================================================
"""

import time
import math
import random
import numpy as np
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm, get_best_greedy_initial_solution
from algorithms.penalty_calculator import calculate_full_penalties, check_swap_feasibility, build_worker_request_details

def run_simulated_annealing(n_workers, n_days, r_day, r_eve, r_night, weights, 
                            t_start=1000.0, t_min=0.01, cooling_rate=0.980, max_iterations=3000, 
                            seed=42, custom_workers=None):
    """
    Simulated Annealing (Tavlama Benzetimi) Vardiya Optimizasyonu.
    Sonlandırma nedeni (termination_reason) eklenmiştir.
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    init_sched, workers, _, _, _, _, _ = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=custom_workers
    )

    curr_schedule = init_sched.copy()
    _, curr_score = calculate_full_penalties(curr_schedule, workers, n_workers, n_days, weights)
    eval_count = 1
    
    best_schedule = curr_schedule.copy()
    best_score = curr_score
    initial_score = curr_score

    temp_history = []
    curr_score_history = []
    best_score_history = []
    acceptance_probs = []
    
    accepted_moves = 0
    worse_accepted_moves = 0
    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    T = float(t_start)
    k = 0
    termination_reason = ""

    while k < max_iterations and T > t_min:
        d = random.randint(0, n_days - 1)
        w1_idx = random.randint(0, n_workers - 1)
        w2_idx = random.randint(0, n_workers - 1)

        if w1_idx == w2_idx:
            k += 1
            continue

        s1 = curr_schedule[w1_idx, d]
        s2 = curr_schedule[w2_idx, d]

        if s1 == s2:
            k += 1
            continue

        curr_schedule[w1_idx, d] = s2
        curr_schedule[w2_idx, d] = s1

        if check_swap_feasibility(curr_schedule, workers, d, w1_idx, w2_idx, n_workers, n_days, shift_reqs):
            _, cand_score = calculate_full_penalties(curr_schedule, workers, n_workers, n_days, weights)
            eval_count += 1
            delta_z = cand_score - curr_score

            if delta_z < 0:
                curr_score = cand_score
                accepted_moves += 1
                prob = 1.0
                if cand_score < best_score:
                    best_score = cand_score
                    best_schedule = curr_schedule.copy()
            else:
                prob = math.exp(-delta_z / T)
                if random.random() < prob:
                    curr_score = cand_score
                    accepted_moves += 1
                    worse_accepted_moves += 1
                else:
                    curr_schedule[w1_idx, d] = s1
                    curr_schedule[w2_idx, d] = s2

            acceptance_probs.append(prob)
        else:
            curr_schedule[w1_idx, d] = s1
            curr_schedule[w2_idx, d] = s2
            acceptance_probs.append(0.0)

        temp_history.append(T)
        curr_score_history.append(curr_score)
        best_score_history.append(best_score)

        T *= cooling_rate
        k += 1

    # SONLANDIRMA NEDENİ TESPİTİ
    if T <= t_min:
        termination_reason = f"❄️ SOGUMA TAMAMLANDI (Sıcaklık Hedef Eşiğe Ulaştı: T = {round(T, 4)} <= {t_min})"
    elif k >= max_iterations:
        termination_reason = f"🏁 İTERASYON LİMİTİNE ULAŞILDI (Maksimum Hamle Sınırı Doldu: K = {max_iterations})"
    else:
        termination_reason = "🏁 ARAMA TAMAMLANDI"

    exec_time = round((time.time() - start_time) * 1000, 2)
    final_penalties, final_total = calculate_full_penalties(best_schedule, workers, n_workers, n_days, weights)
    eval_count += 1
    improvement_rate = round(((initial_score - best_score) / max(1, initial_score)) * 100, 1)

    return {
        'initial_score': initial_score,
        'final_score': best_score,
        'improvement_rate': improvement_rate,
        'accepted_moves': accepted_moves,
        'worse_accepted_moves': worse_accepted_moves,
        'total_iterations': k,
        'eval_count': eval_count,
        'final_temp': round(T, 4),
        'termination_reason': termination_reason,
        'schedule': best_schedule,
        'workers': workers,
        'exec_time_ms': exec_time,
        'penalties': final_penalties,
        'curr_score_history': curr_score_history,
        'best_score_history': best_score_history,
        'temp_history': temp_history,
        'acceptance_probs': acceptance_probs,
        'request_details': build_worker_request_details(best_schedule, workers, n_days, weights)
    }
