"""
================================================================================
  ALGORITHMS/SIMULATED_ANNEALING_SOLVER.PY - SIMULATED ANNEALING (TAVLAMA BENZETİMİ)
================================================================================
  Bu modül, Simulated Annealing (Tavlama Benzetimi) stokastik metasezgisel algoritmasını
  içerir. Algoritma sonlandığında 'termination_reason' (Sonlandırma Nedeni) bilgisini tam
  ve net olarak raporlar. Canlı akış callback desteği içerir.
================================================================================
"""

import time
import math
import random
import numpy as np
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm, get_best_greedy_initial_solution
from algorithms.penalty_calculator import calculate_full_penalties, check_swap_feasibility, build_worker_request_details, audit_all_hard_constraints
from algorithms.solver_contract import build_standard_solver_result

def run_simulated_annealing(n_workers, n_days, r_day, r_eve, r_night, weights, 
                            t_start=1000.0, t_min=0.01, cooling_rate=0.980, max_iterations=3000, 
                            seed=42, custom_workers=None, callback=None, stream_interval=50):
    """
    Simulated Annealing (Tavlama Benzetimi) Vardiya Optimizasyonu.
    Sonlandırma nedeni (termination_reason) ve canlı izleme callback desteği.
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # =========================================================================
    # 1. BAŞLANGIÇ ÇÖZÜMÜ ÜRETİMİ (INITIAL SOLUTION / SEED GENERATION)
    # =========================================================================
    # Sıfırdan rastgele başlamak yerine, sert kısıtları %100 sağlayan en kaliteli
    # Greedy (Yapıcı Sezgisel) çizelgesini başlangıç noktası (warm-start) olarak alır.
    init_res = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=custom_workers
    )
    init_sched = init_res['schedule']
    workers = init_res['workers']

    # =========================================================================
    # 2. MEVCUT DURUM VE EN İYİ (CHAMPION) ÇÖZÜM HAFIZASI
    # =========================================================================
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

    # =========================================================================
    # 3. TAVLAMA PARAMETRELERİ (ANNEALING SCHEDULE / INITIAL TEMPERATURE)
    # =========================================================================
    T = float(t_start)
    k = 0
    termination_reason = ""

    if callback:
        callback(0, best_score, curr_score, f"| T: {T:.1f}")

    # =========================================================================
    # 4. METROPOLIS TAVLAMA VE KOMŞULUK DÖNGÜSÜ (SIMULATED ANNEALING LOOP)
    # =========================================================================
    while k < max_iterations and T > t_min:
        # ---------------------------------------------------------------------
        # 4.1 KOMŞULUK OPERATÖRÜ (NEIGHBORHOOD MOVE: RANDOM 2-WORKER DAY-SWAP)
        # ---------------------------------------------------------------------
        # Rastgele bir gün (d) ve o gün görev yapan rastgele 2 farklı işçi (w1, w2) seçilir.
        d = random.randint(0, n_days - 1)
        w1_idx = random.randint(0, n_workers - 1)
        w2_idx = random.randint(0, n_workers - 1)

        # Aynı işçi seçildiyse takas anlamsızdır (No-op); adım ilerletilir.
        if w1_idx == w2_idx:
            k += 1
            temp_history.append(T)
            curr_score_history.append(curr_score)
            best_score_history.append(best_score)
            T *= cooling_rate
            if callback and k % stream_interval == 0:
                callback(k, best_score, curr_score, f"| T: {T:.1f}")
            continue

        s1 = curr_schedule[w1_idx, d]
        s2 = curr_schedule[w2_idx, d]

        # ---------------------------------------------------------------------
        # 4.2 HIZLI ÖN-FİLTRELEME (SAME SHIFT FILTERING)
        # ---------------------------------------------------------------------
        # İki işçi zaten aynı vardiyadaysa matris değişmez; soğuma uygulanır.
        if s1 == s2:
            k += 1
            temp_history.append(T)
            curr_score_history.append(curr_score)
            best_score_history.append(best_score)
            T *= cooling_rate
            if callback and k % stream_interval == 0:
                callback(k, best_score, curr_score, f"| T: {T:.1f}")
            continue

        # Hamle geçici olarak uygulanır (Trial Move)
        curr_schedule[w1_idx, d] = s2
        curr_schedule[w2_idx, d] = s1

        # ---------------------------------------------------------------------
        # 4.3 SERT KISIT UYGUNLUK DENETİMİ (HARD FEASIBILITY CHECK)
        # ---------------------------------------------------------------------
        # Takasın sert kısıtları (MYK, 11 saat dinlenme, max 6 gün çalışma) ihlal edip
        # etmediği kontrol edilir. Geçersiz hamleler ceza hesaplanmadan anında reddedilir.
        if check_swap_feasibility(curr_schedule, workers, d, w1_idx, w2_idx, n_workers, n_days, shift_reqs):
            # -----------------------------------------------------------------
            # 4.4 YUMUŞAK CEZA DEĞERLENDİRİCİSİ (OBJECTIVE FUNCTION EVALUATION)
            # -----------------------------------------------------------------
            _, cand_score = calculate_full_penalties(curr_schedule, workers, n_workers, n_days, weights)
            eval_count += 1
            delta_z = cand_score - curr_score

            # -----------------------------------------------------------------
            # 4.5 METROPOLIS KABUL KRİTERİ (METROPOLIS ACCEPTANCE CRITERION)
            # -----------------------------------------------------------------
            # Durum 1: Eğer yeni çözüm daha iyiyse (delta_z < 0), kesinlikle kabul edilir (P = 1.0).
            if delta_z < 0:
                curr_score = cand_score
                accepted_moves += 1
                prob = 1.0
                # Küresel en iyi (best champion) çözüm hafızası güncellenir
                if cand_score < best_score:
                    best_score = cand_score
                    best_schedule = curr_schedule.copy()
                    if callback:
                        callback(k + 1, best_score, curr_score, f"| T: {T:.1f}")
            else:
                # Durum 2: Eğer yeni çözüm daha kötüyse (delta_z > 0), yerel optimumdan (local minima)
                # kaçabilmek için P = exp(-delta_z / T) Boltzmann olasılığıyla yine de kabul edilebilir!
                prob = math.exp(-delta_z / T)
                if random.random() < prob:
                    curr_score = cand_score
                    accepted_moves += 1
                    worse_accepted_moves += 1
                else:
                    # Kabul edilmediyse hamle geri alınır (Rollback)
                    curr_schedule[w1_idx, d] = s1
                    curr_schedule[w2_idx, d] = s2

            acceptance_probs.append(prob)
        else:
            # Sert kısıt ihlali durumunda hamle geri alınır (Rollback)
            curr_schedule[w1_idx, d] = s1
            curr_schedule[w2_idx, d] = s2
            acceptance_probs.append(0.0)

        temp_history.append(T)
        curr_score_history.append(curr_score)
        best_score_history.append(best_score)

        # ---------------------------------------------------------------------
        # 4.6 GEOMETRİK SOĞUMA ADIMI (GEOMETRIC COOLING STEP: T = T * alpha)
        # ---------------------------------------------------------------------
        k += 1
        T *= cooling_rate
        
        if callback and k % stream_interval == 0:
            callback(k, best_score, curr_score, f"| T: {T:.1f}")

    # =========================================================================
    # 5. DURDURMA NEDENİ TESPİTİ (TERMINATION REASON DETERMINATION)
    # =========================================================================
    if T <= t_min:
        termination_reason = f"❄️ HEDEF MİNİMUM SICAKLIĞA ULAŞILDI (T = {T:.4f} <= T_min = {t_min})"
    else:
        termination_reason = f"🏁 İTERASYON LİMİTİNE ULAŞILDI (K = {k} >= K_max = {max_iterations})"

    if callback:
        callback(k, best_score, curr_score, f"| T: {T:.1f}")

    exec_time = round((time.time() - start_time) * 1000, 2)
    final_penalties, _ = calculate_full_penalties(best_schedule, workers, n_workers, n_days, weights)
    improvement_rate = round(((initial_score - best_score) / max(1, initial_score)) * 100, 1)

    request_details = build_worker_request_details(best_schedule, workers, n_days, weights)
    is_feasible, hard_viols_count, hard_violation_logs = audit_all_hard_constraints(
        best_schedule, workers, n_workers, n_days, shift_reqs
    )

    return build_standard_solver_result(
        schedule=best_schedule,
        workers=workers,
        is_feasible=is_feasible,
        hard_violations_count=hard_viols_count,
        hard_violation_logs=hard_violation_logs,
        final_score=best_score,
        initial_score=initial_score,
        improvement_rate=improvement_rate,
        exec_time_ms=exec_time,
        eval_count=eval_count,
        total_iterations=k,
        termination_reason=termination_reason,
        penalties=final_penalties,
        request_details=request_details,
        score_history=best_score_history,
        meta={
            'accepted_moves': accepted_moves,
            'worse_accepted_moves': worse_accepted_moves,
            'final_temperature': round(T, 4),
            'temp_history': temp_history,
            'curr_score_history': curr_score_history,
            'acceptance_probs': acceptance_probs,
            'seed_source': init_res.get('meta', {}).get('seed_source', 'Greedy'),
            'is_csp_fallback': init_res.get('meta', {}).get('is_csp_fallback', False),
            'fallback_reason': init_res.get('meta', {}).get('fallback_reason', '')
        }
    )
