"""
================================================================================
  ALGORITHMS/VNS_SOLVER.PY - VARIABLE NEIGHBORHOOD SEARCH (DEĞİŞKEN KOMŞULUK ARAMASI)
================================================================================
  Bu modül, Vardiya Çizelgeleme Problemi (NSP) için Değişken Komşuluk Araması
  (Variable Neighborhood Search - VNS) metasezgisel optimizasyon motorunu içerir.
  
  Temel VNS Prensipleri:
  1. Bir komşuluk yapısına göre yerel minimum olan bir çözüm, başka bir komşuluk
     yapısına göre yerel minimum olmak zorunda değildir.
  2. Küresel optimum, tüm olası komşuluk yapılarına göre bir yerel minimumdur.
  3. Yerel minimumlar arama uzayında birbirlerine kümelenmiş olabilir.

  Tanımlanan Hiyerarşik Komşuluk Yapıları (N_1, N_2, N_3):
  - N_1 (Mikro): 2-İşçi Gün İçi Vardiya Takası (Single Day 2-Worker Shift Swap)
  - N_2 (Orta / Mezo): Çok Günlü / Blok İzin-Vardiya Kaydırması (Cycle & Off Swap)
  - N_3 (Makro): 4-Posta Ekip Düzeyinde Vardiya Rotasyonu (Posta-wide Shift Swap)

  Akış:
  - Shaking (Çalkalama) -> Local Search (Lokal İyileştirme) -> Neighborhood Change
================================================================================
"""

import time
import math
import random
import numpy as np
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm, get_best_greedy_initial_solution
from algorithms.penalty_calculator import (
    calculate_full_penalties,
    check_swap_feasibility,
    check_hard_constraints_single_day
)
from algorithms.solver_contract import finalize_solver_execution


def _apply_n1_micro_swap(schedule, workers, n_workers, n_days, shift_reqs):
    """
    N_1 Komşuluğu (Mikro Düzey):
    Rastgele bir günde çalışan iki işçinin vardiyalarını sert kısıtları koruyarak takas eder.
    """
    d = random.randint(0, n_days - 1)
    w1, w2 = random.sample(range(n_workers), 2)
    if schedule[w1, d] == schedule[w2, d]:
        return schedule, False
        
    cand = schedule.copy()
    cand[w1, d], cand[w2, d] = cand[w2, d], cand[w1, d]
    
    if check_swap_feasibility(cand, workers, d, w1, w2, n_workers, n_days, shift_reqs):
        return cand, True
    return schedule, False


def _apply_n2_mezo_cycle_shift(schedule, workers, n_workers, n_days, shift_reqs):
    """
    N_2 Komşuluğu (Orta / Mezo Düzey):
    Aynı postadan veya farklı postadan iki işçinin ardışık 2 günlük vardiya bloğunu takas eder.
    Böylece sirkadiyen ritim ve izin talepleri daha geniş pencerede dengelenir.
    """
    if n_days < 2:
        return _apply_n1_micro_swap(schedule, workers, n_workers, n_days, shift_reqs)
        
    d = random.randint(0, n_days - 2)
    w1, w2 = random.sample(range(n_workers), 2)
    
    cand = schedule.copy()
    # 2 günlük blok takası
    cand[w1, d], cand[w2, d] = cand[w2, d], cand[w1, d]
    cand[w1, d + 1], cand[w2, d + 1] = cand[w2, d + 1], cand[w1, d + 1]
    
    # Her iki gün için de uygunluk kontrolü
    ok_d1 = check_swap_feasibility(cand, workers, d, w1, w2, n_workers, n_days, shift_reqs)
    ok_d2 = check_swap_feasibility(cand, workers, d + 1, w1, w2, n_workers, n_days, shift_reqs)
    
    if ok_d1 and ok_d2:
        return cand, True
        
    # Eğer 2 gün birden olmadıysa tek günü dene
    cand = schedule.copy()
    cand[w1, d], cand[w2, d] = cand[w2, d], cand[w1, d]
    if check_swap_feasibility(cand, workers, d, w1, w2, n_workers, n_days, shift_reqs):
        return cand, True
        
    return schedule, False


def _apply_n3_macro_posta_swap(schedule, workers, n_workers, n_days, shift_reqs):
    """
    N_3 Komşuluğu (Makro Düzey - Posta Düzeyinde Takas):
    Belirli bir günde Posta X ile Posta Y'nin tamamının vardiyalarını takas eder.
    Postalar kendi içinde tam MYK sertifika setine sahip olduğu için bu hamle
    Posta Takım Bütünlüğünü tek hamlede devasa oranda düzeltir.
    """
    d = random.randint(0, n_days - 1)
    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    p1, p2 = random.sample(postas, 2)
    
    p1_wids = [w['id'] for w in workers if w.get('posta') == p1 and w['id'] < n_workers]
    p2_wids = [w['id'] for w in workers if w.get('posta') == p2 and w['id'] < n_workers]
    
    if not p1_wids or not p2_wids:
        return schedule, False
        
    cand = schedule.copy()
    
    # Posta üyelerinin çoğunluk vardiyalarını bul
    shifts_p1 = [cand[wid, d] for wid in p1_wids]
    shifts_p2 = [cand[wid, d] for wid in p2_wids]
    
    # Eşit boyutta takas et
    min_len = min(len(p1_wids), len(p2_wids))
    for idx in range(min_len):
        w1 = p1_wids[idx]
        w2 = p2_wids[idx]
        cand[w1, d], cand[w2, d] = cand[w2, d], cand[w1, d]
        
    # Günlük sert kısıt denetimi
    if check_hard_constraints_single_day(cand, workers, d, n_workers, shift_reqs):
        # İşçilerin dinlenme ihlallerini doğrula
        is_ok = True
        for wid in (p1_wids + p2_wids):
            if d > 0 and cand[wid, d - 1] in (2, 3) and cand[wid, d] == 1:
                is_ok = False; break
            if d < n_days - 1 and cand[wid, d] in (2, 3) and cand[wid, d + 1] == 1:
                is_ok = False; break
        if is_ok:
            return cand, True
            
    return schedule, False


def run_variable_neighborhood_search(
    n_workers, n_days, r_day, r_eve, r_night, weights,
    max_iterations=1000, max_neighborhoods=3, local_search_depth=15,
    seed=42, custom_workers=None, callback=None, stream_interval=20
):
    """
    Değişken Komşuluk Araması (Variable Neighborhood Search - VNS) Optimizasyon Motoru.
    
    Parametreler:
    - max_iterations (int): Toplam VNS dış döngü iterasyon sayısı (K_max)
    - max_neighborhoods (int): Kullanılacak komşuluk yapısı sayısı (K_neigh = 1, 2, 3)
    - local_search_depth (int): Her çalkalama sonrası uygulanacak yerel arama adım sayısı
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
    
    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)
        
    # 1. Başlangıç Çözümü: En İyi Greedy Çözümü (Best-of-Heuristics Seed)
    greedy_seed = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights,
        custom_workers=custom_workers
    )
    current_schedule = greedy_seed['schedule'].copy()
    current_penalties, current_score = calculate_full_penalties(current_schedule, workers, n_workers, n_days, weights)
    eval_count = 1
    
    best_schedule = current_schedule.copy()
    best_score = current_score
    initial_score = current_score
    
    best_score_history = [float(best_score)]
    curr_score_history = [float(current_score)]
    neighborhood_history = [1]
    neighborhood_usage = {1: 0, 2: 0, 3: 0}
    successful_escapes = {1: 0, 2: 0, 3: 0}
    
    k_neigh = 1
    accepted_moves = 0
    
    if callback:
        callback(0, best_score, current_score, f"| Komşuluk: N_{k_neigh}")
        
    # 2. VNS Ana İterasyon Döngüsü
    for it in range(1, max_iterations + 1):
        # -------------------------------------------------------------
        # ADIM 1: SHAKING (ÇALKALAMA - N_k Komşuluğundan Rastgele Örnekleme)
        # -------------------------------------------------------------
        neighborhood_usage[k_neigh] = neighborhood_usage.get(k_neigh, 0) + 1
        neighborhood_history.append(k_neigh)
        
        shaken_sched = current_schedule.copy()
        applied = False
        
        if k_neigh == 1:
            shaken_sched, applied = _apply_n1_micro_swap(shaken_sched, workers, n_workers, n_days, shift_reqs)
        elif k_neigh == 2:
            shaken_sched, applied = _apply_n2_mezo_cycle_shift(shaken_sched, workers, n_workers, n_days, shift_reqs)
        else: # k_neigh >= 3
            shaken_sched, applied = _apply_n3_macro_posta_swap(shaken_sched, workers, n_workers, n_days, shift_reqs)
            
        # -------------------------------------------------------------
        # ADIM 2: LOCAL SEARCH (LOKAL İYİLEŞTİRME - VND / Micro Descent)
        # -------------------------------------------------------------
        refined_sched = shaken_sched.copy()
        _, refined_score = calculate_full_penalties(refined_sched, workers, n_workers, n_days, weights)
        eval_count += 1
        
        for _ in range(local_search_depth):
            cand_sched, ok = _apply_n1_micro_swap(refined_sched, workers, n_workers, n_days, shift_reqs)
            if ok:
                _, cand_score = calculate_full_penalties(cand_sched, workers, n_workers, n_days, weights)
                eval_count += 1
                if cand_score < refined_score:
                    refined_sched = cand_sched
                    refined_score = cand_score
                    
        # -------------------------------------------------------------
        # ADIM 3: NEIGHBORHOOD CHANGE (KOMŞULUK DEĞİŞTİRME MEKANİZMASI)
        # -------------------------------------------------------------
        if refined_score < current_score:
            current_schedule = refined_sched.copy()
            current_score = refined_score
            accepted_moves += 1
            successful_escapes[k_neigh] = successful_escapes.get(k_neigh, 0) + 1
            
            if refined_score < best_score:
                best_score = refined_score
                best_schedule = refined_sched.copy()
                if callback:
                    callback(it, best_score, current_score, f"| N_{k_neigh} İyileştirdi!")
                    
            # Başarılı olunca en küçük komşuluğa (N_1) geri dön!
            k_neigh = 1
        else:
            # İyileşme olmadıysa bir sonraki daha geniş komşuluğa geç
            k_neigh += 1
            if k_neigh > max_neighborhoods:
                k_neigh = 1
                
        best_score_history.append(float(best_score))
        curr_score_history.append(float(current_score))
        
        if callback and (it % max(1, stream_interval) == 0):
            callback(it, best_score, current_score, f"| Komşuluk: N_{k_neigh}")
            
    if callback:
        callback(max_iterations, best_score, current_score, "| Bitti")
        
    term_reason = f"🔄 VNS hiyerarşik 3 komşuluk yapısı (N_1 Mikro, N_2 Mezo, N_3 Makro) ile {max_iterations} iterasyonluk sistematik aramasını tamamladı."

    meta = {
        'max_iterations': max_iterations,
        'max_neighborhoods': max_neighborhoods,
        'local_search_depth': local_search_depth,
        'accepted_moves': accepted_moves,
        'neighborhood_usage': neighborhood_usage,
        'successful_escapes': successful_escapes,
        'neighborhood_history': neighborhood_history,
        'curr_score_history': curr_score_history,
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
        total_iterations=max_iterations,
        weights=weights,
        shift_reqs=shift_reqs,
        n_workers=n_workers,
        n_days=n_days,
        termination_reason=term_reason,
        score_history=best_score_history,
        meta=meta
    )


# Geriye dönük uyumluluk ve alias
run_vns = run_variable_neighborhood_search
