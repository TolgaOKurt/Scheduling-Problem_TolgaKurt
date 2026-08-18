"""
================================================================================
  ALGORITHMS/TABU_SEARCH_SOLVER.PY - TABU SEARCH (TABU ARAMASI METASEZGİSELİ)
================================================================================
  Bu modül, Vardiya Çizelgeleme Problemi (NSP) için Tabu Search (Tabu Araması)
  hafıza tabanlı metasezgisel optimizasyon motorunu içerir.
  
  Temel Mekanizmalar:
  - Kısa Vadeli Hafıza: Tabu Listesi & Dinamik Tabu Tenure (L)
  - Aspirasyon Kriteri: Yasaklı hamle global en iyiyi aşarsa tabunun delinmesi
  - Yöresel Komşuluk: Sert kısıtları koruyan 2-işçi vardiya takas operatörü
  - Non-Monotonic Arama: Kötüleşen hamleleri kabul ederek yerel vadilerden deterministik çıkış
  - Uzun Vadeli Hafıza: Frekans matrisi ile çeşitlendirme (Diversification)
  - Yüksek Hızlı Vektörel Değerlendirme (Vectorized High-Speed Penalty Engine)
================================================================================
"""

import time
import math
import random
import numpy as np
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm, get_best_greedy_initial_solution
from algorithms.penalty_calculator import calculate_full_penalties, check_hard_constraints_single_day, build_fast_evaluator
from algorithms.solver_contract import finalize_solver_execution


def run_tabu_search(n_workers, n_days, r_day, r_eve, r_night, weights,
                    max_iterations=1000, tabu_tenure=15, neighborhood_size=20,
                    use_aspiration=True, use_diversification=False,
                    seed=42, custom_workers=None, callback=None, stream_interval=20):
    """
    Yüksek Performanslı Tabu Search (Tabu Araması) Optimizasyon Motoru.
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # 1. Başlangıç Çözümü: En İyi Greedy Çözücüsü (Best-of-Heuristics)
    greedy_seed = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights,
        custom_workers=custom_workers
    )
    greedy_sched = greedy_seed['schedule']
    workers = greedy_seed['workers']

    current_schedule = greedy_sched.copy()
    current_penalties, current_score = calculate_full_penalties(current_schedule, workers, n_workers, n_days, weights)
    eval_count = 1

    best_schedule = current_schedule.copy()
    best_score = current_score
    initial_score = current_score

    # Önceden Hesaplanmış Hızlı Dizi Yapıları (Vektörel Hızlandırma)
    fast_evaluate = build_fast_evaluator(workers, n_days, weights)

    # MYK Zorunlu Sertifika Maskeleri (Hızlı Uygunluk Kontrolü İçin)
    required_skills = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
    skill_masks = []
    for sk in required_skills:
        skill_masks.append(np.array([sk in w['skills'] for w in workers]))

    def fast_feasibility(sched, d, w1_idx, w2_idx):
        # 1. MYK 4 Ehliyet Kontrolü
        col = sched[:, d]
        for k in (1, 2, 3):
            shift_mask = (col == k)
            for s_mask in skill_masks:
                if not (shift_mask & s_mask).any():
                    return False

        # 2. Dinlenme Dinamikleri
        s1 = sched[w1_idx, d]
        s2 = sched[w2_idx, d]
        
        if d > 0 and sched[w1_idx, d-1] == 3 and s1 == 1: return False
        if d < n_days - 1 and s1 == 3 and sched[w1_idx, d+1] == 1: return False
        if d > 0 and sched[w2_idx, d-1] == 3 and s2 == 1: return False
        if d < n_days - 1 and s2 == 3 and sched[w2_idx, d+1] == 1: return False

        if d > 0 and sched[w1_idx, d-1] == 2 and s1 == 1: return False
        if d < n_days - 1 and s1 == 2 and sched[w1_idx, d+1] == 1: return False
        if d > 0 and sched[w2_idx, d-1] == 2 and s2 == 1: return False
        if d < n_days - 1 and s2 == 2 and sched[w2_idx, d+1] == 1: return False

        # 3. 7 günlük kayan pencerede en az 1 gün OFF
        for wid in (w1_idx, w2_idx):
            start_tau = max(0, d - 6)
            end_tau = min(max(0, n_days - 7), d)
            for tau in range(start_tau, end_tau + 1):
                if (sched[wid, tau:tau+7] == 0).sum() == 0:
                    return False

        return True

    # Tabu Hafıza Matrisi
    tabu_matrix = np.zeros((n_workers, n_days), dtype=int)
    frequency_matrix = np.zeros((n_workers, n_days, 4), dtype=int)

    best_score_history = []
    curr_score_history = []
    tabu_size_history = []
    aspiration_events = []
    
    accepted_moves = 0
    aspiration_count = 0
    no_improve_streak = 0
    termination_reason = ""

    # TABU SEARCH ANA İTERASYON DÖNGÜSÜ
    for it in range(max_iterations):
        if use_diversification:
            for w in range(n_workers):
                for d in range(n_days):
                    frequency_matrix[w, d, current_schedule[w, d]] += 1

        candidates = []
        sampled_attempts = 0
        max_attempts = neighborhood_size * 3

        while len(candidates) < neighborhood_size and sampled_attempts < max_attempts:
            sampled_attempts += 1
            d = random.randint(0, n_days - 1)
            w1_idx, w2_idx = random.sample(range(n_workers), 2)
            
            s1 = current_schedule[w1_idx, d]
            s2 = current_schedule[w2_idx, d]
            
            if s1 == s2:
                continue

            cand_sched = current_schedule.copy()
            cand_sched[w1_idx, d] = s2
            cand_sched[w2_idx, d] = s1

            if fast_feasibility(cand_sched, d, w1_idx, w2_idx):
                cand_score = fast_evaluate(cand_sched)
                eval_count += 1
                
                eval_score = cand_score
                if use_diversification and it > 50:
                    freq_penalty = (frequency_matrix[w1_idx, d, s2] + frequency_matrix[w2_idx, d, s1]) * 0.1
                    eval_score += freq_penalty

                is_w1_tabu = (tabu_matrix[w1_idx, d] > it)
                is_w2_tabu = (tabu_matrix[w2_idx, d] > it)
                is_tabu = is_w1_tabu or is_w2_tabu

                is_aspiration = False
                if is_tabu and use_aspiration and cand_score < best_score:
                    is_aspiration = True

                is_admissible = (not is_tabu) or is_aspiration

                candidates.append({
                    'schedule': cand_sched,
                    'raw_score': cand_score,
                    'eval_score': eval_score,
                    'w1': w1_idx,
                    'w2': w2_idx,
                    'day': d,
                    'is_tabu': is_tabu,
                    'is_aspiration': is_aspiration,
                    'is_admissible': is_admissible
                })

        admissible_cands = [c for c in candidates if c['is_admissible']]
        
        if len(admissible_cands) > 0:
            best_cand = min(admissible_cands, key=lambda c: c['eval_score'])
        elif len(candidates) > 0:
            best_cand = min(candidates, key=lambda c: c['eval_score'])
        else:
            no_improve_streak += 1
            best_score_history.append(best_score)
            curr_score_history.append(current_score)
            tabu_size_history.append((tabu_matrix > it).sum())
            continue

        # Hamleyi Gerçekleştir
        current_schedule = best_cand['schedule'].copy()
        current_score = best_cand['raw_score']
        
        # Tabu Listesini Güncelle
        tabu_matrix[best_cand['w1'], best_cand['day']] = it + tabu_tenure
        tabu_matrix[best_cand['w2'], best_cand['day']] = it + tabu_tenure

        if best_cand['is_aspiration']:
            aspiration_count += 1
            aspiration_events.append({'iteration': it + 1, 'score': current_score})

        if current_score < best_score:
            best_score = current_score
            best_schedule = current_schedule.copy()
            accepted_moves += 1
            no_improve_streak = 0
            if callback:
                callback(it + 1, best_score, current_score, f"| Tabu: {(tabu_matrix > it).sum()}")
        else:
            no_improve_streak += 1

        best_score_history.append(best_score)
        curr_score_history.append(current_score)
        tabu_size_history.append((tabu_matrix > it).sum())

        if callback and (it % stream_interval == 0):
            callback(it + 1, best_score, current_score, f"| Tabu: {(tabu_matrix > it).sum()}")

        # Erken Durdurma Kriterleri
        if best_score == 0:
            termination_reason = f"🎯 KUSURSUZ ÇÖZÜME ULAŞILDI (Z = 0, İterasyon {it+1})"
            break

        if no_improve_streak >= 500:
            termination_reason = f"🛑 DURAKLAMA LİMİTİ (Son 500 İterasyonda Gelişme Olmadı)"
            break

    if not termination_reason:
        termination_reason = f"🏁 İTERASYON LİMİTİNE ULAŞILDI (Maksimum Hamle Sınırı: K = {max_iterations})"

    if callback:
        callback(len(best_score_history), best_score, current_score, "| Bitti")

    meta = {
        'accepted_moves': accepted_moves,
        'aspiration_count': aspiration_count,
        'aspiration_events': aspiration_events,
        'tabu_tenure': tabu_tenure,
        'curr_score_history': curr_score_history,
        'tabu_size_history': tabu_size_history,
        'seed_source': greedy_seed.get('meta', {}).get('seed_source', 'Greedy'),
        'is_csp_fallback': greedy_seed.get('meta', {}).get('is_csp_fallback', False),
        'fallback_reason': greedy_seed.get('meta', {}).get('fallback_reason', '')
    }

    return finalize_solver_execution(
        best_schedule=best_schedule,
        workers=workers,
        initial_score=initial_score,
        start_time=start_time,
        eval_count=eval_count,
        total_iterations=len(best_score_history),
        weights=weights,
        shift_reqs=shift_reqs,
        n_workers=n_workers,
        n_days=n_days,
        termination_reason=termination_reason,
        score_history=best_score_history,
        meta=meta
    )
