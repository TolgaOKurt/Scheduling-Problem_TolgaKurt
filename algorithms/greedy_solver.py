"""
================================================================================
  ALGORITHMS/GREEDY_SOLVER.PY - GREEDY / HEURISTIC VARDIYA SOLVER
================================================================================
  Bu modül, Sert Kısıtları (Hard Constraints) denetleyen, 4-Posta takım bütünlüğünü
  ve MYK sertifikalarını denetleyip cezalandıran Açgözlü (Greedy) algoritmasını içerir.
================================================================================
"""

import numpy as np
import pandas as pd
from algorithms.worker_manager import generate_worker_profiles
from algorithms.penalty_calculator import calculate_full_penalties, build_worker_request_details

def run_greedy_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, solver_mode="Naif Greedy (Standart Açgözlü Yaklaşım)", custom_workers=None):
    """
    Greedy Vardiya Çizelgeleme Algoritması.
    """
    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # DÜZELTME 1: schedule boyutu gerçek worker sayısına ve en büyük ID'ye göre açılır.
    # custom_workers ID'leri DataFrame index'inden geldiği için n_workers yetmeyebilir.
    actual_n = max((w['id'] for w in workers), default=0) + 1
    actual_n = max(actual_n, n_workers)  # en az n_workers kadar olsun

    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    
    # Matris: 0: OFF, 1: Gündüz (08-16), 2: Akşam (16-24), 3: Gece (24-08)
    schedule = np.zeros((actual_n, n_days), dtype=int)
    night_counts = np.zeros(actual_n, dtype=int)
    total_worked = np.zeros(actual_n, dtype=int)
    
    hard_violation_logs = []
    shift_names = {1: "Gündüz (08-16)", 2: "Akşam (16-24)", 3: "Gece (24-08)"}
    
    staggered_off_days = {}
    if "Akıllı" in solver_mode:
        for i in range(n_workers):
            staggered_off_days[i] = (i % 7)
            
    # Gün gün Atama Döngüsü
    for d in range(n_days):
        shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
        
        # 4-Posta Dönüşüm Öncelik Dizilimi (Gündüz, Akşam, Gece)
        target_posta_for_shift = {
            1: postas[d % 4],
            2: postas[(d + 1) % 4],
            3: postas[(d + 2) % 4]
        }
        
        for k in [1, 2, 3]:
            needed = shift_reqs[k]
            target_p = target_posta_for_shift[k]
            candidates = []
            
            for w in workers:
                wid = w['id']
                
                # Sert Kısıt 1: Aynı gün 2 vardiyaya yazılmaz
                if schedule[wid, d] != 0:
                    continue
                
                # Sert Kısıt 3 & 4: Gece (3) çalıştıysa ertesi gün Gündüz (1) yazılamaz (11-16 saat dinlenme)
                if d > 0 and schedule[wid, d-1] == 3 and k == 1:
                    continue
                
                # Sert Kısıt: Akşam (2) çalıştıysa ertesi gün Gündüz (1) yazılamaz (yetersiz dinlenme)
                if d > 0 and schedule[wid, d-1] == 2 and k == 1:
                    continue
                
                # Mod kontrolü: Akıllı Greedy ise kademeli izin gününde çalıştırılmaz
                if "Akıllı" in solver_mode and (d % 7) == staggered_off_days[wid]:
                    continue
                
                # Sert Kısıt 5: Naif Greedy modunda 7 gün üst üste çalışanlar zorunlu izin alır.
                # DÜZELTME 2: Gerçekten 7 ardışık günü kontrol etmek için d>=7 ve range(d-7, d) kullanılmalı.
                if "Naif" in solver_mode and d >= 7:
                    past_7_offs = sum(1 for tau in range(d-7, d) if schedule[wid, tau] == 0)
                    if past_7_offs == 0:
                        continue
                        
                candidates.append(w)
            
            # Posta Öncelikli Sıralama: O vardiyanın hedef Posta grubuna ait olan işçilere birincil öncelik ver
            candidates.sort(key=lambda x: (
                0 if x['posta'] == target_p else 1,
                night_counts[x['id']] if k == 3 else 0,
                total_worked[x['id']]
            ))
            
            # SERT KISIT 2 (MYK KRİTİK SERTİFİKA EŞLEŞTİRME)
            assigned = []
            req_cert_list = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
            
            for cert in req_cert_list:
                if len(assigned) >= needed:
                    break
                if not any(cert in w['skills'] for w in assigned):
                    for cand in candidates:
                        if cand not in assigned and cert in cand['skills']:
                            assigned.append(cand)
                            break
                            
            for cand in candidates:
                if len(assigned) >= needed:
                    break
                if cand not in assigned:
                    assigned.append(cand)
            
            if len(assigned) < needed:
                shortage = needed - len(assigned)
                # DÜZELTME 3: Bu log kadro yetersizliğini raporlar; "Sert Kısıt 1" (aynı gün 2 vardiya) ile karışmaması için etiket düzeltildi.
                log_msg = f"❌ Kadro Yetersizliği [Gün {d+1} - {shift_names[k]}]: İstenen min {needed} kişi, atanan {len(assigned)} kişi! (Eksik: {shortage} kişi)"
                hard_violation_logs.append(log_msg)
            
            for w in assigned:
                wid = w['id']
                schedule[wid, d] = k
                total_worked[wid] += 1
                if k == 3:
                    night_counts[wid] += 1

            if len(assigned) > 0:
                assigned_skills = set()
                for w in assigned:
                    assigned_skills.update(w['skills'])
                
                for req_skill in req_cert_list:
                    if req_skill not in assigned_skills:
                        log_msg = f"⚠️ Sert Kısıt 2 İhlali [Gün {d+1} - {shift_names[k]}]: Eksik Kritik Sertifika ➔ Vardiyada '{req_skill}' ehliyetli eleman yok!"
                        hard_violation_logs.append(log_msg)

    # Kişisel İzin Talepleri Detay Takibi (Merkezi Yardımcı Fonksiyon)
    request_details = build_worker_request_details(schedule, workers, n_days, weights)
    
    # YUMUŞAK KISIT CEZALARI VE POSTA TAKIM BÜTÜNLÜĞÜ HESABI (MERKEZİ MOTOR)
    penalties, total_penalty = calculate_full_penalties(schedule, workers, n_workers, n_days, weights)
    hard_violations_count = len(hard_violation_logs)
    
    return schedule, workers, penalties, total_penalty, hard_violations_count, hard_violation_logs, request_details


def get_best_greedy_initial_solution(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=None):
    """
    Hem Naif Greedy hem de Akıllı Kademeli Greedy çözücülerini saliseler içinde çalıştırıp
    sert kısıtları ihlal etmeyen ve toplam ceza puanı EN DÜŞÜK olan en iyi başlangıç çözümünü döndürür.
    (Best-of-Heuristics Seeding)
    """
    # 1. Naif Greedy
    res_naif = run_greedy_algorithm(
        n_workers, n_days, r_day, r_eve, r_night, weights,
        solver_mode="Naif Greedy (Standart Açgözlü Yaklaşım)", custom_workers=custom_workers
    )
    
    # 2. Akıllı Kademeli Greedy
    res_smart = run_greedy_algorithm(
        n_workers, n_days, r_day, r_eve, r_night, weights,
        solver_mode="Akıllı Kademeli Greedy (İzinleri Günlere Yayan)", custom_workers=custom_workers
    )
    
    s_naif, w_naif, p_naif, score_naif, hard_naif, logs_naif, req_naif = res_naif
    s_smart, w_smart, p_smart, score_smart, hard_smart, logs_smart, req_smart = res_smart
    
    # Eğer her ikisi de sert kısıtları sağlıyorsa, ceza puanı daha düşük olanı seç
    if hard_naif == 0 and hard_smart == 0:
        if score_naif <= score_smart:
            return res_naif
        else:
            return res_smart
    elif hard_naif == 0:
        return res_naif
    else:
        return res_smart
