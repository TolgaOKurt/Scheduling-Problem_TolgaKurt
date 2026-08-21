"""
================================================================================
  ALGORITHMS/ACO_SOLVER.PY - ANT COLONY OPTIMIZATION (ACO) FOR SHIFT SCHEDULING
================================================================================
  Bu modül, Ağır Sanayi ve Hemşire Vardiya Çizelgeleme Problemi (NSP/WSP) için
  Kesikli Karınca Kolonisi Optimizasyonu (Discrete Ant Colony Optimization - ACO)
  metasezgisel motorunu içerir.
  
  Temel Karınca Kolonisi İlkeleri (Dorigo 1992, Stützle & Hoos 2000 - MMAS):
  - Feromon İzi (Pheromone Trails - tau): Başarılı vardiya atamalarının kolektif hafızası
  - Sezgisel Görünürlük (Heuristic Visibility - eta): Yerel kural ve tercih cazibesi
  - Geçiş Olasılığı (Transition Probability - P_ijk): tau^alpha * eta^beta
  - Feromon Buharlaşması (Evaporation - rho): Eski yolların unutulması ve çeşitlilik
  - Feromon Takviyesi (Deposit - Delta tau): En iyi karıncanın çizelgesini pekiştirme
  - Max-Min Sınırları (MMAS): Feromonun taşmasını veya sönmesini önleyen [tau_min, tau_max]
================================================================================
"""

import time
import random
import numpy as np
from typing import List, Dict, Any, Optional

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import get_best_greedy_initial_solution
from algorithms.penalty_calculator import (
    calculate_full_penalties,
    build_fast_evaluator,
    check_swap_feasibility
)
from algorithms.solver_contract import finalize_solver_execution


# =============================================================================
# 1. SEZGİSEL GÖRÜNÜRLÜK MATRİSİ İNŞASI (HEURISTIC VISIBILITY: eta = 1/cost)
# =============================================================================
def _build_heuristic_visibility(workers: List[Dict[str, Any]], n_days: int) -> np.ndarray:
    """
    Her (işçi i, gün t, vardiya k) ataması için sezgisel görünürlük (eta) matrisini oluşturur.
    eta[i, t, k] değeri ne kadar yüksekse, karıncanın o atamayı seçme cazibesi o kadar artar.
    """
    n_workers = len(workers)
    eta = np.ones((n_workers, n_days, 4), dtype=float)

    for i, w in enumerate(workers):
        pref_day_idx = int(w.get('pref_off', 0)) - 1
        is_usta = w.get('is_usta', False)

        for t in range(n_days):
            # 1. Kişisel İzin Tercihi: Talep edilen günde OFF (0) ataması çok caziptir
            if t == pref_day_idx:
                eta[i, t, 0] = 3.5
                eta[i, t, 1:] = 0.5
            else:
                eta[i, t, 0] = 1.0

            # 2. Kıdemli Usta: Ustanın aktif vardiyalarda (1, 2, 3) bulunması tercih edilir
            if is_usta and t != pref_day_idx:
                eta[i, t, 1:] += 0.8

            # 3. Taban görünürlük
            eta[i, t, 1] += 0.2  # Gündüz
            eta[i, t, 2] += 0.1  # Akşam
            eta[i, t, 3] += 0.1  # Gece

    return eta


def run_ant_colony_optimization(
    n_workers: int,
    n_days: int,
    r_day: int,
    r_eve: int,
    r_night: int,
    weights: Dict[str, int],
    n_ants: int = 20,
    max_iterations: int = 60,
    evaporation_rate: float = 0.15,
    alpha: float = 1.0,
    beta: float = 2.0,
    q_deposit: float = 100.0,
    seed: int = 42,
    custom_workers: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Vardiya Çizelgeleme Problemi için Kesikli Karınca Kolonisi Optimizasyonu (Discrete ACO) çalıştırır.
    """
    random.seed(seed)
    np.random.seed(seed)
    start_time = time.time()

    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # =========================================================================
    # 2. BAŞLANGIÇ TOHUMU VE FEROMON MATRİSİ İLKLENDİRMESİ (MMAS INITIALIZATION)
    # =========================================================================
    greedy_seed = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers
    )
    initial_schedule = greedy_seed['schedule'].copy()
    seed_source = greedy_seed.get('meta', {}).get('seed_source', 'Greedy (MRV/LCV)')
    is_csp_fallback = greedy_seed.get('meta', {}).get('is_csp_fallback', False)
    fallback_reason = greedy_seed.get('meta', {}).get('fallback_reason', '')
    csp_failed = greedy_seed.get('meta', {}).get('csp_attempted_and_failed', False)

    # Vektörize Hızlı Değerlendirme Yapısı
    fast_evaluate = build_fast_evaluator(workers, n_days, weights)

    initial_score = fast_evaluate(initial_schedule)
    best_schedule = initial_schedule.copy()
    best_score = initial_score

    # Max-Min Ant System (MMAS) feromon sınırları
    tau_max = 5.0
    tau_min = 0.1
    tau_init = 1.0
    pheromone = np.full((n_workers, n_days, 4), fill_value=tau_init, dtype=float)

    # Başlangıç tohumunun feromon izini belirginleştir
    for i in range(n_workers):
        for t in range(n_days):
            k = initial_schedule[i, t]
            pheromone[i, t, k] += 0.5

    eta = _build_heuristic_visibility(workers, n_days)

    score_history = [float(best_score)]
    iter_best_history = [float(best_score)]
    iter_mean_history = [float(best_score)]
    iter_worst_history = [float(best_score)]
    
    init_active = [pheromone[i, t, initial_schedule[i, t]] for i in range(n_workers) for t in range(n_days)]
    init_inactive = [pheromone[i, t, k] for i in range(n_workers) for t in range(n_days) for k in range(4) if k != initial_schedule[i, t]]
    
    mean_pheromone_history = [round(float(np.mean(pheromone)), 4)]
    active_pheromone_history = [round(float(np.mean(init_active)), 4)]
    inactive_pheromone_history = [round(float(np.mean(init_inactive)), 4)]
    std_pheromone_history = [round(float(np.std(pheromone)), 4)]
    max_pheromone_history = [round(float(np.max(pheromone)), 4)]
    min_pheromone_history = [round(float(np.min(pheromone)), 4)]

    eval_count = 1
    accepted_moves = 0
    total_ants_toured = 0
    swap_prob_rejected = 0
    swap_hard_rejected = 0
    swap_accepted = 0

    # =========================================================================
    # 3. KARINCA KOLONİSİ İTERASYON DÖNGÜSÜ (ACO MAIN LOOP)
    # =========================================================================
    for iteration in range(max_iterations):
        iter_best_schedule = None
        iter_best_score = float('inf')
        ant_scores = []

        # ---------------------------------------------------------------------
        # 3.1 KARINCALARIN ÇİZELGE İNŞASI VE GEZİNTİSİ (ANT TOUR / SEARCH)
        # ---------------------------------------------------------------------
        for ant_id in range(n_ants):
            total_ants_toured += 1
            ant_schedule = best_schedule.copy() if random.random() < 0.60 else initial_schedule.copy()

            # Her karınca için gün bazında feromon güdümlü olasılıksal takaslar
            n_swaps_to_try = random.randint(3, max(4, n_days))

            for _ in range(n_swaps_to_try):
                d = random.randint(0, n_days - 1)
                w1, w2 = random.sample(range(n_workers), 2)
                s1 = ant_schedule[w1, d]
                s2 = ant_schedule[w2, d]

                if s1 == s2:
                    continue

                # -------------------------------------------------------------
                # 3.2 FEROMON VE SEZGİSEL ÇEKİCİLİK HESABI (P = tau^alpha * eta^beta)
                # -------------------------------------------------------------
                desirability_new = (
                    (pheromone[w1, d, s2] ** alpha) * (eta[w1, d, s2] ** beta) *
                    (pheromone[w2, d, s1] ** alpha) * (eta[w2, d, s1] ** beta)
                )
                desirability_curr = (
                    (pheromone[w1, d, s1] ** alpha) * (eta[w1, d, s1] ** beta) *
                    (pheromone[w2, d, s2] ** alpha) * (eta[w2, d, s2] ** beta)
                )

                prob_swap = desirability_new / max(1e-6, (desirability_new + desirability_curr))

                # Olasılıksal Geçiş Kararı
                if random.random() < prob_swap:
                    # ---------------------------------------------------------
                    # 3.3 SERT KISIT UYGUNLUK TESTİ (FEASIBILITY CHECK)
                    # ---------------------------------------------------------
                    ant_schedule[w1, d], ant_schedule[w2, d] = s2, s1
                    if check_swap_feasibility(ant_schedule, workers, d, w1, w2, n_workers, n_days, shift_reqs):
                        accepted_moves += 1
                        swap_accepted += 1
                    else:
                        ant_schedule[w1, d], ant_schedule[w2, d] = s1, s2  # Rollback
                        swap_hard_rejected += 1
                else:
                    swap_prob_rejected += 1

            # Karınca Çözümünün Skorlanması
            ant_score = fast_evaluate(ant_schedule)
            eval_count += 1
            ant_scores.append(ant_score)

            # İterasyonun En İyi Karıncası Tespiti
            if ant_score < iter_best_score:
                iter_best_score = ant_score
                iter_best_schedule = ant_schedule.copy()

        # İterasyon İstatistikleri (Koloni Dağılımı)
        if ant_scores:
            iter_best_history.append(float(min(ant_scores)))
            iter_mean_history.append(round(float(np.mean(ant_scores)), 1))
            iter_worst_history.append(float(max(ant_scores)))

        # ---------------------------------------------------------------------
        # 3.4 DAEMON ACTIONS (LOKAL İYİLEŞTİRME / 12 MİKRO-ONARIM ADIMI)
        # ---------------------------------------------------------------------
        # Koloninin o turdaki en başarılı karıncasına cerrahi mikro takaslar uygulanır
        if iter_best_schedule is not None:
            for _ in range(12):
                d = random.randint(0, n_days - 1)
                w1, w2 = random.sample(range(n_workers), 2)
                if iter_best_schedule[w1, d] != iter_best_schedule[w2, d]:
                    s1 = iter_best_schedule[w1, d]
                    s2 = iter_best_schedule[w2, d]
                    iter_best_schedule[w1, d], iter_best_schedule[w2, d] = s2, s1
                    if check_swap_feasibility(iter_best_schedule, workers, d, w1, w2, n_workers, n_days, shift_reqs):
                        cand_score = fast_evaluate(iter_best_schedule)
                        eval_count += 1
                        if cand_score < iter_best_score:
                            iter_best_score = cand_score
                        else:
                            iter_best_schedule[w1, d], iter_best_schedule[w2, d] = s1, s2
                    else:
                        iter_best_schedule[w1, d], iter_best_schedule[w2, d] = s1, s2

            # Küresel Şampiyonu Güncelle
            if iter_best_score < best_score:
                best_score = iter_best_score
                best_schedule = iter_best_schedule.copy()

        score_history.append(float(best_score))

        # ---------------------------------------------------------------------
        # 3.5 FEROMON BUHARLAŞMASI (PHEROMONE EVAPORATION: tau = (1 - rho) * tau)
        # ---------------------------------------------------------------------
        pheromone = (1.0 - evaporation_rate) * pheromone

        # ---------------------------------------------------------------------
        # 3.6 FEROMON TAKVİYESİ (PHEROMONE DEPOSIT & MMAS CLAMPING)
        # ---------------------------------------------------------------------
        # İterasyonun ve Küresel En İyinin Patikalarına Feromon Takviyesi Yapılır
        if iter_best_schedule is not None:
            delta_tau_iter = q_deposit / max(1.0, float(iter_best_score))
            for i in range(n_workers):
                for t in range(n_days):
                    k_iter = iter_best_schedule[i, t]
                    pheromone[i, t, k_iter] += 0.4 * delta_tau_iter

        delta_tau_global = q_deposit / max(1.0, float(best_score))
        for i in range(n_workers):
            for t in range(n_days):
                k_best = best_schedule[i, t]
                pheromone[i, t, k_best] += 0.6 * delta_tau_global

        # Max-Min Feromon Sınırlandırması (MMAS Clamping: [tau_min, tau_max])
        pheromone = np.clip(pheromone, tau_min, tau_max)

        cur_active = [pheromone[i, t, best_schedule[i, t]] for i in range(n_workers) for t in range(n_days)]
        cur_inactive = [pheromone[i, t, k] for i in range(n_workers) for t in range(n_days) for k in range(4) if k != best_schedule[i, t]]

        mean_pheromone_history.append(round(float(np.mean(pheromone)), 4))
        active_pheromone_history.append(round(float(np.mean(cur_active)), 4))
        inactive_pheromone_history.append(round(float(np.mean(cur_inactive)), 4))
        std_pheromone_history.append(round(float(np.std(pheromone)), 4))
        max_pheromone_history.append(round(float(np.max(pheromone)), 4))
        min_pheromone_history.append(round(float(np.min(pheromone)), 4))

        # 7. CANLI STREAMLIT İLERLEME GERİ BİLDİRİMİ
        if callback:
            mean_phero = float(np.mean(pheromone))
            cur_s = int(iter_best_score) if iter_best_score != float('inf') else int(best_score)
            callback(
                iteration + 1,
                int(best_score),
                cur_s,
                f"| Karınca: {n_ants} | Ort. Feromon: {mean_phero:.2f}"
            )

    # 2D Ortalama Feromon Yoğunluğu Matrisi (Görselleştirme için)
    active_pheromone_matrix = np.zeros((n_workers, n_days), dtype=float)
    for i in range(n_workers):
        for t in range(n_days):
            k = best_schedule[i, t]
            active_pheromone_matrix[i, t] = round(float(pheromone[i, t, k]), 3)

    shift_labels = ["0: İzin (OFF)", "1: Gündüz (08-16)", "2: Akşam (16-24)", "3: Gece (24-08)"]
    shift_pheromone_power = [round(float(np.mean(pheromone[:, :, k] ** alpha)), 3) for k in range(4)]
    shift_heuristic_power = [round(float(np.mean(eta[:, :, k] ** beta)), 3) for k in range(4)]

    meta = {
        'seed_source': seed_source,
        'is_csp_fallback': is_csp_fallback,
        'fallback_reason': fallback_reason,
        'csp_attempted_and_failed': csp_failed,
        'accepted_moves': accepted_moves,
        'n_ants': n_ants,
        'evaporation_rate': evaporation_rate,
        'alpha': alpha,
        'beta': beta,
        'active_pheromone_matrix': active_pheromone_matrix.tolist(),
        'total_ants_toured': total_ants_toured,
        'iter_best_history': iter_best_history,
        'iter_mean_history': iter_mean_history,
        'iter_worst_history': iter_worst_history,
        'mean_pheromone_history': mean_pheromone_history,
        'active_pheromone_history': active_pheromone_history,
        'inactive_pheromone_history': inactive_pheromone_history,
        'std_pheromone_history': std_pheromone_history,
        'max_pheromone_history': max_pheromone_history,
        'min_pheromone_history': min_pheromone_history,
        'shift_labels': shift_labels,
        'shift_pheromone_power': shift_pheromone_power,
        'shift_heuristic_power': shift_heuristic_power,
        'swap_stats': {
            'prob_rejected': swap_prob_rejected,
            'hard_rejected': swap_hard_rejected,
            'accepted': swap_accepted
        }
    }

    return finalize_solver_execution(
        best_schedule=best_schedule,
        workers=workers,
        initial_score=initial_score,
        start_time=start_time,
        eval_count=eval_count,
        total_iterations=max_iterations,
        weights=weights,
        shift_reqs=shift_reqs,
        n_workers=n_workers,
        n_days=n_days,
        termination_reason=f"Maksimum Karınca Turu ({max_iterations} İterasyon, {n_ants} Karınca) Tamamlandı",
        score_history=score_history,
        meta=meta
    )
