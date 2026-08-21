"""
================================================================================
  ALGORITHMS/HILL_CLIMBING_SOLVER.PY - HILL CLIMBING / LOCAL SEARCH SOLVER
================================================================================
  Bu modül, Tepeden Tırmanma / Yöresel Arama (Hill Climbing Local Search)
  meta-sezgisel algoritmasını içerir. Algoritma sonlandığında 'termination_reason'
  (Sonlandırma Nedeni) bilgisini tam ve net olarak raporlar.
  Canlı akış (real-time stream callback) desteği içerir.
================================================================================
"""

import time
import random
import numpy as np
import pandas as pd
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm, get_best_greedy_initial_solution
from algorithms.penalty_calculator import calculate_full_penalties, check_swap_feasibility, build_worker_request_details, audit_all_hard_constraints
from algorithms.solver_contract import build_standard_solver_result

def run_hill_climbing(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=3000, seed=42, custom_workers=None, callback=None, stream_interval=50, time_limit=None):
    """
    Tepeden Tırmanma (Hill Climbing Local Search) Çözücüsü.
    Sonlandırma nedeni (termination_reason), süre bütçesi (time_limit) ve canlı izleme callback desteği.
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
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
    # 2. MEVCUT DURUM VE BAŞLANGIÇ CEZA SKORU DEĞERLENDİRMESİ
    # =========================================================================
    current_schedule = init_sched.copy()
    current_penalties, current_score = calculate_full_penalties(current_schedule, workers, n_workers, n_days, weights)
    eval_count = 1
    
    initial_score = current_score
    history = [float(current_score)]
    accepted_moves = 0
    no_improve_streak = 0
    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
    termination_reason = ""
    
    if callback:
        callback(0, initial_score, initial_score, "| Başlangıç")
    
    # =========================================================================
    # 3. YÖRESEL ARAMA & KOMŞULUK DÖNGÜSÜ (LOCAL SEARCH / HILL CLIMBING LOOP)
    # =========================================================================
    effective_max_iters = 1000000 if time_limit else max_iterations
    for it in range(effective_max_iters):
        if time_limit and (time.time() - start_time) >= time_limit:
            termination_reason = f"Maksimum Süre Sınırına Ulaşıldı ({time_limit}s)"
            break
        # ---------------------------------------------------------------------
        # 3.1 KOMŞULUK OPERATÖRÜ (NEIGHBORHOOD MOVE: RANDOM 2-WORKER DAY-SWAP)
        # ---------------------------------------------------------------------
        # Rastgele bir gün (d) ve o gün görev yapan rastgele 2 farklı işçi (w1, w2) seçilir.
        d = random.randint(0, n_days - 1)
        w1_idx = random.randint(0, n_workers - 1)
        w2_idx = random.randint(0, n_workers - 1)
        
        # Aynı işçi seçildiyse takas anlamsızdır; pas geçilir (No-op).
        if w1_idx == w2_idx:
            no_improve_streak += 1
            history.append(float(current_score))
            if callback and it % stream_interval == 0:
                callback(it + 1, current_score, current_score, f"| İyileşen: {accepted_moves}")
            continue
            
        s1 = current_schedule[w1_idx, d]
        s2 = current_schedule[w2_idx, d]
        
        # ---------------------------------------------------------------------
        # 3.2 HIZLI ÖN-FİLTRELEME (SAME SHIFT FILTERING)
        # ---------------------------------------------------------------------
        # İki işçi zaten aynı vardiyadaysa (örn: ikisi de Gece) takas matrisi değiştirmez.
        if s1 == s2:
            no_improve_streak += 1
            history.append(float(current_score))
            if callback and it % stream_interval == 0:
                callback(it + 1, current_score, current_score, f"| İyileşen: {accepted_moves}")
            continue
            
        # Hamle geçici olarak uygulanır (Trial Move)
        current_schedule[w1_idx, d] = s2
        current_schedule[w2_idx, d] = s1
        
        # ---------------------------------------------------------------------
        # 3.3 SERT KISIT UYGUNLUK DENETİMİ (HARD FEASIBILITY CHECK)
        # ---------------------------------------------------------------------
        # Yapılan takas; MYK sertifikalarını, 11 saat dinlenmeyi veya 6 gün üst üste
        # çalışma kuralını bozuyor mu denetlenir. Bozuyorsa ceza hesaplanmadan anında reddedilir.
        if check_swap_feasibility(current_schedule, workers, d, w1_idx, w2_idx, n_workers, n_days, shift_reqs):
            # -----------------------------------------------------------------
            # 3.4 YUMUŞAK CEZA DEĞERLENDİRİCİSİ (OBJECTIVE FUNCTION EVALUATION)
            # -----------------------------------------------------------------
            new_penalties, new_score = calculate_full_penalties(current_schedule, workers, n_workers, n_days, weights)
            eval_count += 1
            
            # -----------------------------------------------------------------
            # 3.5 KABUL KRİTERİ (STRICT DESCENT ACCEPTANCE)
            # -----------------------------------------------------------------
            # Hill Climbing yalnızca ve kesinlikle skoru düşüren (iyileştiren) hamleleri kabul eder.
            if new_score < current_score:
                current_score = new_score
                current_penalties = new_penalties
                accepted_moves += 1
                no_improve_streak = 0
                if callback:
                    callback(it + 1, current_score, current_score, f"| İyileşen: {accepted_moves}")
            else:
                # İyileşme yoksa hamle geri alınır (Rollback)
                current_schedule[w1_idx, d] = s1
                current_schedule[w2_idx, d] = s2
                no_improve_streak += 1
        else:
            # Sert kısıt ihlali durumunda hamle geri alınır (Rollback)
            current_schedule[w1_idx, d] = s1
            current_schedule[w2_idx, d] = s2
            no_improve_streak += 1
            
        history.append(float(current_score))
        if callback and it % stream_interval == 0:
            callback(it + 1, current_score, current_score, f"| İyileşen: {accepted_moves}")

        # ---------------------------------------------------------------------
        # 3.6 DURDURMA KRİTERİ: YEREL OPTİMUM TUZAĞI (EARLY STOPPING / STAGNATION)
        # ---------------------------------------------------------------------
        # Son 500 iterasyon boyunca hiçbir iyileştirici komşu bulunamadıysa yerel tepeye ulaşılmıştır.
        if no_improve_streak >= 500:
            termination_reason = "🛑 YEREL OPTİMUMDA DURDU (Son 500 İterasyonda İyileştiren Komşu Kalmadı)"
            break
            
    # -------------------------------------------------------------------------
    # 3.7 DURDURMA KRİTERİ: MAKSİMUM İTERASYON LİMİTİ (MAX ITERATIONS REACHED)
    # -------------------------------------------------------------------------
    if not termination_reason:
        termination_reason = f"🏁 İTERASYON LİMİTİNE ULAŞILDI (Maksimum Hamle Sınırı Doldu: K = {max_iterations})"

    if callback:
        callback(len(history) - 1, current_score, current_score, f"| İyileşen: {accepted_moves}")

    exec_time = round((time.time() - start_time) * 1000, 2)
    improvement_rate = round(((initial_score - current_score) / max(1, initial_score)) * 100, 1)
    
    request_details = build_worker_request_details(current_schedule, workers, n_days, weights)
    is_feasible, hard_viols_count, hard_violation_logs = audit_all_hard_constraints(
        current_schedule, workers, n_workers, n_days, shift_reqs
    )
        
    seed_meta = init_res.get('meta', {})
    meta = {
        'accepted_moves': accepted_moves,
        'seed_source': seed_meta.get('seed_source', 'Greedy'),
        'is_csp_fallback': seed_meta.get('is_csp_fallback', False),
        'fallback_reason': seed_meta.get('fallback_reason', '')
    }
        
    return build_standard_solver_result(
        schedule=current_schedule,
        workers=workers,
        is_feasible=is_feasible,
        hard_violations_count=hard_viols_count,
        hard_violation_logs=hard_violation_logs,
        final_score=current_score,
        initial_score=initial_score,
        improvement_rate=improvement_rate,
        exec_time_ms=exec_time,
        eval_count=eval_count,
        total_iterations=len(history) - 1,
        termination_reason=termination_reason,
        penalties=current_penalties,
        request_details=request_details,
        score_history=history,
        meta=meta
    )
