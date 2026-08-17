"""
================================================================================
  ALGORITHMS/GREEDY_SOLVER.PY - GREEDY / YAPICI SEZGİSEL VARDIYA SOLVER
================================================================================
  Bu modül, Sert Kısıtları (Hard Constraints) denetleyen, 4-Posta takım bütünlüğünü
  ve MYK sertifikalarını denetleyip cezalandıran Yapıcı Sezgisel (Constructive / Greedy)
  algoritmasını içerir. 3 farklı literatür modu desteklenir:
  1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)
  2. Kademeli / Desen Tabanlı Yapıcı Sezgisel (Staggered Pattern-Based Greedy)
  3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)
================================================================================
"""

import time
import numpy as np
import pandas as pd
from algorithms.worker_manager import generate_worker_profiles
from algorithms.penalty_calculator import calculate_full_penalties, build_worker_request_details, audit_all_hard_constraints
from algorithms.solver_contract import build_standard_solver_result

def run_greedy_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, solver_mode="1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)", custom_workers=None):
    """
    Greedy / Yapıcı Sezgisel (Constructive Heuristic) Vardiya Çizelgeleme Algoritması.
    """
    start_time = time.time()
    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    actual_n = max((w['id'] for w in workers), default=0) + 1
    actual_n = max(actual_n, n_workers)

    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    req_cert_list = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
    shift_names = {1: "Gündüz (08-16)", 2: "Akşam (16-24)", 3: "Gece (24-08)"}
    
    # Matris: 0: OFF, 1: Gündüz (08-16), 2: Akşam (16-24), 3: Gece (24-08)
    schedule = np.zeros((actual_n, n_days), dtype=int)
    night_counts = np.zeros(actual_n, dtype=int)
    total_worked = np.zeros(actual_n, dtype=int)
    consecutive_work = np.zeros(actual_n, dtype=int)
    
    hard_violation_logs = []
    
    is_mrv_lcv = ("MRV" in solver_mode or "LCV" in solver_mode or "Kısıt Öncelikli" in solver_mode or "Constraint-First" in solver_mode)
    is_staggered = ("Kademeli" in solver_mode or "Akıllı" in solver_mode or "Staggered" in solver_mode or "Pattern" in solver_mode or is_mrv_lcv)
    
    staggered_off_days = {}
    if is_staggered:
        for w in workers:
            wid = w['id']
            if is_mrv_lcv and w.get('pref_off', 0) > 0:
                staggered_off_days[wid] = (int(w['pref_off']) - 1) % 7
            else:
                staggered_off_days[wid] = (wid % 7)
            
    # Gün gün Atama Döngüsü
    for d in range(n_days):
        shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
        
        # 4-Posta Dönüşüm Öncelik Dizilimi (Gündüz, Akşam, Gece)
        target_posta_for_shift = {
            1: postas[d % 4],
            2: postas[(d + 1) % 4],
            3: postas[(d + 2) % 4]
        }
        
        assigned_shifts = {1: [], 2: [], 3: []}
        
        if is_staggered:
            # -------------------------------------------------------------
            # KADEMELİ / MRV-LCV MODU:
            # FAZ 1: Gün içindeki 3 vardiyanın her birine 4 zorunlu MYK ehliyetini
            # önceden paylaştır (Gündüzün tüm ehliyetleri tekeline almasını engeller).
            # -------------------------------------------------------------
            for k in [1, 2, 3]:
                target_p = target_posta_for_shift[k]
                for cert in req_cert_list:
                    if any(cert in w['skills'] for w in assigned_shifts[k]):
                        continue
                    cands = []
                    for w in workers:
                        wid = w['id']
                        if schedule[wid, d] != 0 or any(w in assigned_shifts[s] for s in [1, 2, 3]):
                            continue
                        if d > 0 and schedule[wid, d-1] in (2, 3) and k == 1:
                            continue
                        if (d % 7) == staggered_off_days[wid]:
                            continue
                        if cert in w['skills']:
                            cands.append(w)
                    if cands:
                        if is_mrv_lcv:
                            # LCV: Joker ehliyetlileri sakla, izin gününü koru
                            cands.sort(key=lambda x: (
                                0 if x['posta'] == target_p else 1,
                                1 if (x.get('pref_off', 0) == (d + 1)) else 0,
                                len(x['skills']),
                                total_worked[x['id']]
                            ))
                        else:
                            cands.sort(key=lambda x: (
                                0 if x['posta'] == target_p else 1,
                                len(x['skills']),
                                total_worked[x['id']]
                            ))
                        assigned_shifts[k].append(cands[0])
                        
            # FAZ 2: Kalan boş kadroları doldur
            for k in [1, 2, 3]:
                needed = shift_reqs[k]
                target_p = target_posta_for_shift[k]
                cands = []
                for w in workers:
                    wid = w['id']
                    if schedule[wid, d] != 0 or any(w in assigned_shifts[s] for s in [1, 2, 3]):
                        continue
                    if d > 0 and schedule[wid, d-1] in (2, 3) and k == 1:
                        continue
                    if (d % 7) == staggered_off_days[wid]:
                        continue
                    cands.append(w)
                    
                if is_mrv_lcv:
                    cands.sort(key=lambda x: (
                        0 if x['posta'] == target_p else 1,
                        1 if (x.get('pref_off', 0) == (d + 1)) else 0,
                        night_counts[x['id']] if k == 3 else 0,
                        total_worked[x['id']],
                        -len(x['skills'])
                    ))
                else:
                    cands.sort(key=lambda x: (
                        0 if x['posta'] == target_p else 1,
                        night_counts[x['id']] if k == 3 else 0,
                        total_worked[x['id']]
                    ))
                    
                for c in cands:
                    if len(assigned_shifts[k]) >= needed:
                        break
                    assigned_shifts[k].append(c)
        else:
            # -------------------------------------------------------------
            # 1. SIRALI / MİYOPİK GREEDY MODU (Klasik ardışık doldurma)
            # İleriye bakış yapmaz, Gündüz -> Akşam -> Gece sırasıyla doldurur.
            # -------------------------------------------------------------
            for k in [1, 2, 3]:
                needed = shift_reqs[k]
                target_p = target_posta_for_shift[k]
                cands = []
                for w in workers:
                    wid = w['id']
                    if schedule[wid, d] != 0 or any(w in assigned_shifts[s] for s in [1, 2, 3]):
                        continue
                    if d > 0 and schedule[wid, d-1] in (2, 3) and k == 1:
                        continue
                    # 6 gün kesintisiz çalışan işçiyi kanun gereği 7. gün izne ayır
                    if consecutive_work[wid] >= 6:
                        continue
                    cands.append(w)
                    
                cands.sort(key=lambda x: (
                    0 if x['posta'] == target_p else 1,
                    night_counts[x['id']] if k == 3 else 0,
                    total_worked[x['id']]
                ))
                
                # Sertifika eşleştirme
                for cert in req_cert_list:
                    if len(assigned_shifts[k]) >= needed:
                        break
                    if not any(cert in w['skills'] for w in assigned_shifts[k]):
                        for cand in cands:
                            if cand not in assigned_shifts[k] and cert in cand['skills']:
                                assigned_shifts[k].append(cand)
                                break
                                
                for cand in cands:
                    if len(assigned_shifts[k]) >= needed:
                        break
                    if cand not in assigned_shifts[k]:
                        assigned_shifts[k].append(cand)

        # Atamaları Çizelgeye Kaydet ve Durum Güncelle
        for k in [1, 2, 3]:
            assigned = assigned_shifts[k]
            for w in assigned:
                wid = w['id']
                schedule[wid, d] = k
                total_worked[wid] += 1
                if k == 3:
                    night_counts[wid] += 1

        # Gün sonu kesintisiz çalışma takibi
        for wid in range(actual_n):
            if schedule[wid, d] > 0:
                consecutive_work[wid] += 1
            else:
                consecutive_work[wid] = 0

    # Kişisel İzin Talepleri Detay Takibi (Merkezi Yardımcı Fonksiyon)
    request_details = build_worker_request_details(schedule, workers, n_days, weights)
    
    # YUMUŞAK KISIT CEZALARI VE POSTA TAKIM BÜTÜNLÜĞÜ HESABI (MERKEZİ MOTOR)
    penalties, total_penalty = calculate_full_penalties(schedule, workers, n_workers, n_days, weights)
    
    # Merkezi Sert Kısıt Denetim Uyumluluğu (Tek ve Kesin Kaynak)
    is_feasible, hard_violations_count, hard_violation_logs = audit_all_hard_constraints(
        schedule, workers, n_workers, n_days, {1: r_day, 2: r_eve, 3: r_night}
    )
    exec_time_ms = (time.time() - start_time) * 1000.0

    return build_standard_solver_result(
        schedule=schedule,
        workers=workers,
        is_feasible=is_feasible,
        hard_violations_count=hard_violations_count,
        hard_violation_logs=hard_violation_logs,
        final_score=total_penalty,
        initial_score=total_penalty,
        improvement_rate=0.0,
        exec_time_ms=exec_time_ms,
        eval_count=1,
        total_iterations=1,
        termination_reason=f"Yapıcı sezgisel kural tabanlı atamayı tamamladı ({solver_mode}).",
        penalties=penalties,
        request_details=request_details,
        score_history=[float(total_penalty)],
        meta={'solver_mode': solver_mode}
    )


def get_best_greedy_initial_solution(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=None):
    """
    3 farklı yapıcı sezgiseli (Miyopik, Kademeli, MRV/LCV) saliseler içinde çalıştırıp
    sert kısıtları ihlal etmeyen ve toplam ceza puanı EN DÜŞÜK olan en iyi başlangıç çözümünü döndürür.
    (Best-of-Heuristics Seeding)
    """
    modes = [
        "1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)",
        "2. Kademeli / Desen Tabanlı Yapıcı Sezgisel (Staggered Pattern-Based Greedy)",
        "3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)"
    ]
    
    all_results = []
    for mode in modes:
        res = run_greedy_algorithm(
            n_workers, n_days, r_day, r_eve, r_night, weights,
            solver_mode=mode, custom_workers=custom_workers
        )
        all_results.append(res)
        
    # Sert kısıtı 0 olanları filtrele
    valid_results = [r for r in all_results if r['is_feasible']]
    
    if valid_results:
        # En düşük toplam ceza puanına sahip olanı seç
        return min(valid_results, key=lambda r: r['final_score'])
    else:
        # En az sert kısıt ihlali ve en düşük ceza puanına sahip olanı seç
        return min(all_results, key=lambda r: (r['hard_violations_count'], r['final_score']))
