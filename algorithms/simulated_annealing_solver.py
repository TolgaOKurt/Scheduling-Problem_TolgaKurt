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
    """Vardiya matrisi için 5 yumuşak ceza puanını tam olarak hesaplar."""
    w_posta = weights.get('posta', 35)
    w_circ = weights.get('circadian', 50)
    w_pref = weights.get('pref_off', 40)
    w_night_imb = weights.get('night_imb', 25)
    w_exp = weights.get('exp_mix', 30)

    night_counts = np.array([np.sum(schedule[i, :] == 3) for i in range(n_workers)])
    avg_night = np.mean(night_counts) if n_workers > 0 else 0

    penalties = {
        'Posta Takım Bütünlüğü İhlali': 0,
        'Sirkadiyen Ritim İhlali (Akşam->Gündüz)': 0,
        'Gece Nöbeti Dengesizliği': 0,
        'Kıdem & MYK Sertifika Eksikliği': 0,
        'Kişisel İzin İhlali': 0
    }

    # 1. Sirkadiyen & Kişisel İzin
    for i in range(n_workers):
        for t in range(n_days - 1):
            if schedule[i, t] == 2 and schedule[i, t+1] == 1:
                penalties['Sirkadiyen Ritim İhlali (Akşam->Gündüz)'] += w_circ
        
        diff = abs(night_counts[i] - avg_night)
        penalties['Gece Nöbeti Dengesizliği'] += int(diff * w_night_imb)
        
        p_day = workers[i]['pref_off']
        if p_day < n_days and schedule[i, p_day] != 0:
            penalties['Kişisel İzin İhlali'] += w_pref

    # 2. Kıdemli Usta Varlığı
    worker_map = {w['id']: w for w in workers}
    for t in range(n_days):
        for k in [1, 2, 3]:
            shift_wids = [w['id'] for w in workers if schedule[w['id'], t] == k]
            ustas = sum(1 for wid in shift_wids if worker_map[wid]['is_usta'])
            if len(shift_wids) > 0 and ustas == 0:
                penalties['Kıdem & MYK Sertifika Eksikliği'] += w_exp

    # 3. Posta Takım Bütünlüğü
    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    for t in range(n_days):
        for p in postas:
            p_wids = [w['id'] for w in workers if w['posta'] == p]
            active_shifts = [schedule[wid, t] for wid in p_wids if schedule[wid, t] != 0]
            if len(active_shifts) > 1:
                counts = [active_shifts.count(s) for s in set(active_shifts)]
                majority = max(counts)
                deviated = len(active_shifts) - majority
                penalties['Posta Takım Bütünlüğü İhlali'] += deviated * w_posta

    return penalties, sum(penalties.values())

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

    init_sched, workers, _, _, _, _, _ = run_greedy_algorithm(
        n_workers, n_days, r_day, r_eve, r_night, weights, solver_mode="Akıllı Kademeli Greedy (İzinleri Günlere Yayan)", custom_workers=custom_workers
    )

    curr_schedule = init_sched.copy()
    _, curr_score = calculate_full_penalties(curr_schedule, workers, n_workers, n_days, weights)
    
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

        valid_w1 = True
        valid_w2 = True

        if d > 0 and curr_schedule[w1_idx, d-1] == 3 and s2 == 1: valid_w1 = False
        if d < n_days - 1 and s2 == 3 and curr_schedule[w1_idx, d+1] == 1: valid_w1 = False
        if d > 0 and curr_schedule[w2_idx, d-1] == 3 and s1 == 1: valid_w2 = False
        if d < n_days - 1 and s1 == 3 and curr_schedule[w2_idx, d+1] == 1: valid_w2 = False
        # Akşam→Gündüz dinlenme kontrolü
        if d > 0 and curr_schedule[w1_idx, d-1] == 2 and s2 == 1: valid_w1 = False
        if d < n_days - 1 and s2 == 2 and curr_schedule[w1_idx, d+1] == 1: valid_w1 = False
        if d > 0 and curr_schedule[w2_idx, d-1] == 2 and s1 == 1: valid_w2 = False
        if d < n_days - 1 and s1 == 2 and curr_schedule[w2_idx, d+1] == 1: valid_w2 = False

        if valid_w1 and valid_w2:
            valid_w1 = check_hard_constraints_single_day(curr_schedule, workers, d, n_workers, shift_reqs)
            if valid_w1:
                for wid in [w1_idx, w2_idx]:
                    start_tau = max(0, d - 6)
                    end_tau = min(max(0, n_days - 7), d)
                    for tau in range(start_tau, end_tau + 1):
                        if np.sum(curr_schedule[wid, tau:tau+7] == 0) == 0:
                            valid_w1 = False
                            break
                    if not valid_w1:
                        break

        if valid_w1:
            _, cand_score = calculate_full_penalties(curr_schedule, workers, n_workers, n_days, weights)
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
    improvement_rate = round(((initial_score - best_score) / max(1, initial_score)) * 100, 1)

    return {
        'initial_score': initial_score,
        'final_score': best_score,
        'improvement_rate': improvement_rate,
        'accepted_moves': accepted_moves,
        'worse_accepted_moves': worse_accepted_moves,
        'total_iterations': k,
        'final_temp': round(T, 4),
        'termination_reason': termination_reason,
        'schedule': best_schedule,
        'workers': workers,
        'exec_time_ms': exec_time,
        'penalties': final_penalties,
        'curr_score_history': curr_score_history,
        'best_score_history': best_score_history,
        'temp_history': temp_history,
        'acceptance_probs': acceptance_probs
    }
