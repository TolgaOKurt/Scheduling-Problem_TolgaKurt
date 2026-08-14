"""
================================================================================
  ALGORITHMS/HILL_CLIMBING_SOLVER.PY - HILL CLIMBING / LOCAL SEARCH SOLVER
================================================================================
  Bu modül, Tepeden Tırmanma / Yöresel Arama (Hill Climbing Local Search)
  meta-sezgisel algoritmasını içerir. Algoritma sonlandığında 'termination_reason'
  (Sonlandırma Nedeni) bilgisini tam ve net olarak raporlar.
================================================================================
"""

import time
import random
import numpy as np
import pandas as pd
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm

def check_hard_constraints_single_day(schedule, workers, day, n_workers, shift_reqs):
    """Belirli bir günde Sert Kısıtların (Min İhtiyaç ve 4 MYK Ehliyeti) sağlanıp sağlanmadığını denetler."""
    shift_counts = {1: 0, 2: 0, 3: 0}
    shift_skills = {1: set(), 2: set(), 3: set()}
    
    for wid in range(n_workers):
        k = schedule[wid, day]
        if k > 0:
            shift_counts[k] += 1
            shift_skills[k].update(workers[wid]['skills'])
            
    # Min kadro kontrolü
    for k in [1, 2, 3]:
        if shift_counts[k] < shift_reqs[k]:
            return False
            
    # MYK ehliyet kontrolü
    req_certs = {'Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu'}
    for k in [1, 2, 3]:
        if not req_certs.issubset(shift_skills[k]):
            return False
            
    return True

def calculate_full_penalties(schedule, workers, n_workers, n_days, weights):
    """Çizelgenin tüm yumuşak kısıt cezalarını ve sert ihlallerini hesaplar."""
    penalties = {
        'Posta Takım Bütünlüğü İhlali': 0,
        'Sirkadiyen Ritim İhlali (Akşam->Gündüz)': 0,
        'Gece Nöbeti Dengesizliği': 0,
        'Kıdem & MYK Sertifika Eksikliği': 0,
        'Kişisel İzin İhlali': 0
    }
    
    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    w_circ = weights.get('circadian', 50)
    w_pref = weights.get('pref_off', 40)
    w_night_imb = weights.get('night_imb', 25)
    w_posta = weights.get('posta', 35)
    w_exp = weights.get('exp_mix', 30)
    
    # 1. Posta Takım Bütünlüğü
    for d in range(n_days):
        for p in postas:
            p_wids = [w['id'] for w in workers if w['posta'] == p]
            active_shifts = [schedule[wid, d] for wid in p_wids if schedule[wid, d] != 0]
            if len(active_shifts) > 1:
                counts = [active_shifts.count(s) for s in set(active_shifts)]
                majority = max(counts)
                deviated = len(active_shifts) - majority
                penalties['Posta Takım Bütünlüğü İhlali'] += deviated * w_posta
                
    # 2. Sirkadiyen Ritim & Gece Dengesi & Kişisel İzin
    night_counts = np.array([np.sum(schedule[i, :] == 3) for i in range(n_workers)])
    avg_night = np.mean(night_counts) if n_workers > 0 else 0
    
    for i in range(n_workers):
        for d in range(n_days - 1):
            if schedule[i, d] == 2 and schedule[i, d+1] == 1:
                penalties['Sirkadiyen Ritim İhlali (Akşam->Gündüz)'] += w_circ
                
        diff = abs(night_counts[i] - avg_night)
        penalties['Gece Nöbeti Dengesizliği'] += int(diff * w_night_imb)
        
        p_day = workers[i]['pref_off']
        if p_day < n_days and schedule[i, p_day] != 0:
            penalties['Kişisel İzin İhlali'] += w_pref
            
    # 3. Kıdem Eksikliği
    worker_map = {w['id']: w for w in workers}
    for d in range(n_days):
        for k in [1, 2, 3]:
            shift_wids = [w['id'] for w in workers if schedule[w['id'], d] == k]
            ustas = sum(1 for wid in shift_wids if worker_map[wid]['is_usta'])
            if len(shift_wids) > 0 and ustas == 0:
                penalties['Kıdem & MYK Sertifika Eksikliği'] += w_exp
                
    return penalties, sum(penalties.values())

def run_hill_climbing(n_workers, n_days, r_day, r_eve, r_night, weights, max_iterations=3000, seed=42, custom_workers=None):
    """
    Tepeden Tırmanma (Hill Climbing Local Search) Çözücüsü.
    Sonlandırma nedeni (termination_reason) eklendi.
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    init_sched, workers, init_penalties, init_total_score, hard_viols, hard_logs, req_details = run_greedy_algorithm(
        n_workers, n_days, r_day, r_eve, r_night, weights, solver_mode="Akıllı Kademeli Greedy (İzinleri Günlere Yayan)", custom_workers=custom_workers
    )
    
    current_schedule = init_sched.copy()
    current_penalties, current_score = calculate_full_penalties(current_schedule, workers, n_workers, n_days, weights)
    
    initial_score = current_score
    history = [current_score]
    accepted_moves = 0
    no_improve_streak = 0
    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
    termination_reason = ""
    
    # LOCAL SEARCH İTERASYON DÖNGÜSÜ
    for it in range(max_iterations):
        d = random.randint(0, n_days - 1)
        w1_idx = random.randint(0, n_workers - 1)
        w2_idx = random.randint(0, n_workers - 1)
        
        if w1_idx == w2_idx:
            no_improve_streak += 1
            continue
            
        s1 = current_schedule[w1_idx, d]
        s2 = current_schedule[w2_idx, d]
        
        if s1 == s2:
            no_improve_streak += 1
            continue
            
        current_schedule[w1_idx, d] = s2
        current_schedule[w2_idx, d] = s1
        
        valid_w1 = True
        valid_w2 = True
        
        if d > 0 and current_schedule[w1_idx, d-1] == 3 and s2 == 1: valid_w1 = False
        if d < n_days - 1 and s2 == 3 and current_schedule[w1_idx, d+1] == 1: valid_w1 = False
        if d > 0 and current_schedule[w2_idx, d-1] == 3 and s1 == 1: valid_w2 = False
        if d < n_days - 1 and s1 == 3 and current_schedule[w2_idx, d+1] == 1: valid_w2 = False
        # Akşam→Gündüz dinlenme kontrolü
        if d > 0 and current_schedule[w1_idx, d-1] == 2 and s2 == 1: valid_w1 = False
        if d < n_days - 1 and s2 == 2 and current_schedule[w1_idx, d+1] == 1: valid_w1 = False
        if d > 0 and current_schedule[w2_idx, d-1] == 2 and s1 == 1: valid_w2 = False
        if d < n_days - 1 and s1 == 2 and current_schedule[w2_idx, d+1] == 1: valid_w2 = False
        
        if valid_w1 and valid_w2:
            valid_w1 = check_hard_constraints_single_day(current_schedule, workers, d, n_workers, shift_reqs)
            if valid_w1:
                for wid in [w1_idx, w2_idx]:
                    start_tau = max(0, d - 6)
                    end_tau = min(max(0, n_days - 7), d)
                    for tau in range(start_tau, end_tau + 1):
                        if np.sum(current_schedule[wid, tau:tau+7] == 0) == 0:
                            valid_w1 = False
                            break
                    if not valid_w1:
                        break
            
        if valid_w1:
            new_penalties, new_score = calculate_full_penalties(current_schedule, workers, n_workers, n_days, weights)
            
            if new_score < current_score:
                current_score = new_score
                current_penalties = new_penalties
                accepted_moves += 1
                no_improve_streak = 0
            else:
                current_schedule[w1_idx, d] = s1
                current_schedule[w2_idx, d] = s2
                no_improve_streak += 1
        else:
            current_schedule[w1_idx, d] = s1
            current_schedule[w2_idx, d] = s2
            no_improve_streak += 1
            
        history.append(current_score)

        # Erken Durdurma Kontrolü (Yerel Optimum Tuzak)
        if no_improve_streak >= 500:
            termination_reason = f"🛑 YEREL OPTİMUMDA DURDU (Son 500 İterasyonda İyileştiren Komşu Kalmadı)"
            break
            
    if not termination_reason:
        termination_reason = f"🏁 İTERASYON LİMİTİNE ULAŞILDI (Maksimum Hamle Sınırı Doldu: K = {max_iterations})"

    exec_time = round((time.time() - start_time) * 1000, 2)
    improvement_rate = round(((initial_score - current_score) / max(1, initial_score)) * 100, 1)
    
    request_details = []
    for w in workers:
        wid = w['id']
        p_day = w['pref_off']
        assigned_shift = current_schedule[wid, p_day] if p_day < n_days else 0
        is_fulfilled = (assigned_shift == 0)
        request_details.append({
            'id': wid,
            'name': w['name'],
            'posta': w['posta'],
            'unvan': "Kıdemli Usta" if w['is_usta'] else "Operatör/İşçi",
            'talep_gun': f"Gün {p_day + 1}",
            'atandi_vardiya': "OFF (İzin)" if is_fulfilled else ("Gündüz" if assigned_shift==1 else ("Akşam" if assigned_shift==2 else "Gece")),
            'durum': "✅ Karşılandı (OFF verildi)" if is_fulfilled else "❌ İhlal Edildi (Vardiyaya Yazıldı)",
            'ceza_puani': 0 if is_fulfilled else weights.get('pref_off', 40)
        })
        
    return {
        'initial_score': initial_score,
        'final_score': current_score,
        'improvement_rate': improvement_rate,
        'accepted_moves': accepted_moves,
        'total_iterations': len(history) - 1,
        'termination_reason': termination_reason,
        'exec_time_ms': exec_time,
        'history': history,
        'schedule': current_schedule,
        'workers': workers,
        'penalties': current_penalties,
        'request_details': request_details
    }
