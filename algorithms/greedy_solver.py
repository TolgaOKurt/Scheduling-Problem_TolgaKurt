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
    
    # =========================================================================
    # 1. PLANLAMA DEĞİŞKENLERİ VE KADEMELİ İZİN TABLOSU
    # =========================================================================
    staggered_off_days = {}
    if is_staggered:
        for w in workers:
            wid = w['id']
            if is_mrv_lcv and w.get('pref_off', 0) > 0:
                staggered_off_days[wid] = (int(w['pref_off']) - 1) % 7
            else:
                staggered_off_days[wid] = (wid % 7)
            
    # =========================================================================
    # 2. GÜNLÜK YAPICI ATAMA DÖNGÜSÜ (DAY-BY-DAY CONSTRUCTIVE LOOP)
    # =========================================================================
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
            # -----------------------------------------------------------------
            # KADEMELİ / MRV-LCV MODU (2-FAZLI İLERİYE BAKIŞLI ATAMA)
            # -----------------------------------------------------------------
            # FAZ 1: Gün içindeki 3 vardiyanın her birine 4 zorunlu MYK ehliyetini
            # önceden paylaştır (Gündüzün tüm ehliyetleri tekeline almasını engeller).
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
                        # Sert Kısıt 1: Biyolojik Dinlenme (Gece/Akşam sonrası Gündüze gelemez)
                        if d > 0 and schedule[wid, d-1] in (2, 3) and k == 1:
                            continue
                        # Kademeli İzin Günü Koruması
                        if (d % 7) == staggered_off_days[wid]:
                            continue
                        if cert in w['skills']:
                            cands.append(w)
                    if cands:
                        if is_mrv_lcv:
                            # LCV Sezgisi: Joker çok ehliyetlileri sakla, izin gününü koru
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
                        
            # -----------------------------------------------------------------
            # FAZ 2: Kalan boş kadroları hedeflenen postaya göre doldur
            # -----------------------------------------------------------------
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
            # -----------------------------------------------------------------
            # 1. SIRALI / MİYOPİK GREEDY MODU (Klasik ardışık doldurma)
            # İleriye bakış yapmaz, Gündüz -> Akşam -> Gece sırasıyla doldurur.
            # -----------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 2.3 ATAMALARIN MATRİSE YAZILMASI VE ÇALIŞMA SAYILARININ GÜNCELLENMESİ
        # ---------------------------------------------------------------------
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

    # =========================================================================
    # 3. YUMUŞAK CEZA DEĞERLENDİRMESİ VE SERT KISIT DENETİMİ
    # =========================================================================
    request_details = build_worker_request_details(schedule, workers, n_days, weights)
    penalties, total_penalty = calculate_full_penalties(schedule, workers, n_workers, n_days, weights)
    
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
    
    Eğer tüm 3 Greedy sezgiseli de miyopik tıkanma nedeniyle ihlalli sonuç üretirse,
    otomatik olarak CSP Backtracking motoru devreye girer ve %100 geçerli bir tohum üretir.
    Bu durum kullanıcıya ve metasezgisellere 'is_csp_fallback' meta bayrağı ile bildirilir.
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
        best_valid = min(valid_results, key=lambda r: r['final_score'])
        if 'meta' not in best_valid or best_valid['meta'] is None:
            best_valid['meta'] = {}
        best_valid['meta']['is_csp_fallback'] = False
        best_valid['meta']['seed_source'] = f"Greedy ({best_valid['meta'].get('solver_mode', 'Yapıcı Sezgisel')})"
        return best_valid
    else:
        # 3 Greedy de ihlalli çıktı! Otomatik CSP Backtracking devreye girsin (Yeterli limit: 15,000 backtrack).
        csp_failed = False
        try:
            from algorithms.csp_backtracking_solver import CSPBacktrackingSolver
            csp = CSPBacktrackingSolver(
                n_workers=n_workers,
                n_days=n_days,
                r_day=r_day,
                r_eve=r_eve,
                r_night=r_night,
                max_backtracks=15000,
                custom_workers=custom_workers,
                weights=weights
            )
            csp_res = csp.solve()
            if csp_res and csp_res.get('is_feasible'):
                if 'meta' not in csp_res or csp_res['meta'] is None:
                    csp_res['meta'] = {}
                csp_res['meta']['is_csp_fallback'] = True
                csp_res['meta']['seed_source'] = "CSP Backtracking (Otomatik Kurtarma / Fallback Tohumu)"
                csp_res['meta']['fallback_reason'] = "Tüm 3 Açgözlü Sezgisel (Miyopik, Kademeli, MRV/LCV) kısıt tıkanması yaşadığı için %100 geçerli CSP tohumu devreye girdi."
                return csp_res
            else:
                csp_failed = True
        except Exception:
            csp_failed = True

        # Eğer CSP de çözemezse (problem matematiksel olarak imkansızsa), en az ihlalliyi döndür
        fallback_res = min(all_results, key=lambda r: (r['hard_violations_count'], r['final_score']))
        if 'meta' not in fallback_res or fallback_res['meta'] is None:
            fallback_res['meta'] = {}
        fallback_res['meta']['is_csp_fallback'] = False
        fallback_res['meta']['csp_attempted_and_failed'] = csp_failed
        fallback_res['meta']['seed_source'] = "Greedy (İhlalli Başlangıç - CSP Sınırında da Çözülemedi)"
        fallback_res['meta']['fallback_reason'] = "3 Greedy de ihlalli sonuç üretti ve CSP Backtracking arama sınırında geçerli bir çizelge bulamadı."
        return fallback_res
