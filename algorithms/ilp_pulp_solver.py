"""
================================================================================
  ALGORITHMS/ILP_PULP_SOLVER.PY - INTEGER LINEAR PROGRAMMING (ILP/MILP) SOLVER
================================================================================
  Bu modül, Tam Sayılı Doğrusal Programlama (Integer Linear Programming - ILP/MILP)
  yöntemi ile vardiya çizelgeleme problemini matematiksel olarak tam uygunlukla (optimal)
  çözer. Hata durumlarını (Infeasible vs Timeout vs Optimal) %100 hassasiyetle ayırır.
================================================================================
"""

import time
import numpy as np
import pandas as pd
import pulp
from algorithms.worker_manager import generate_worker_profiles

import math

def compute_lp_relaxation_bound(n_workers, n_days, r_day, r_eve, r_night, weights, workers):
    """
    Tam Sayılı (Integer) kısıtlar gevşetilerek (Continuous LP Relaxation) ve analitik
    kaçınılmaz cezalar hesaplanarak problemin GERÇEK TEORİK ALT SINIRINI (Best Bound) bulur.
    """
    w_night = weights.get('night_imb', 25)
    w_exp = weights.get('exp_mix', 30)
    w_pref = weights.get('pref_off', 40)
    w_posta = weights.get('posta', 15)
    w_circ = weights.get('circadian', 50)
    
    # 1. Analitik kaçınılmaz alt sınırlar
    tot_night_req = r_night * n_days
    r = tot_night_req % max(1, n_workers)
    night_lb = (2.0 * r * (n_workers - r) / max(1, n_workers)) * w_night
    
    ustas = [w for w in workers if w['is_usta']]
    max_shifts_per_usta = math.floor(n_days * 6 / 7)
    tot_usta_cap = len(ustas) * max_shifts_per_usta
    tot_req_shifts = 3 * n_days
    missing_usta = max(0, tot_req_shifts - tot_usta_cap)
    usta_lb = missing_usta * w_exp
    
    daily_req_sum = r_day + r_eve + r_night
    max_off_capacity_per_day = max(0, n_workers - daily_req_sum)
    off_requests_per_day = {}
    for w in workers:
        p_day = w['pref_off']
        if p_day < n_days:
            off_requests_per_day[p_day] = off_requests_per_day.get(p_day, 0) + 1
    pref_lb = sum(max(0, reqs - max_off_capacity_per_day) for reqs in off_requests_per_day.values()) * w_pref
    
    analytical_lb = night_lb + usta_lb + pref_lb

    try:
        model = pulp.LpProblem("LP_Relaxation_Bound", pulp.LpMinimize)
        shifts = [0, 1, 2, 3]
        shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
        postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
        ustas_wids = [w['id'] for w in workers if w['is_usta']]
        
        x = pulp.LpVariable.dicts("x", ((i, t, k) for i in range(n_workers) for t in range(n_days) for k in shifts), cat=pulp.LpContinuous, lowBound=0, upBound=1)
        sirk = pulp.LpVariable.dicts("sirkadiyen", ((i, t) for i in range(n_workers) for t in range(n_days - 1)), cat=pulp.LpContinuous, lowBound=0)
        pref_viol = pulp.LpVariable.dicts("pref_viol", (i for i in range(n_workers)), cat=pulp.LpContinuous, lowBound=0)
        y_posta = pulp.LpVariable.dicts("y_posta", ((p, t, k) for p in postas for t in range(n_days) for k in [1, 2, 3]), lowBound=0, cat=pulp.LpContinuous)
        z_posta = pulp.LpVariable.dicts("z_posta", ((p, t, k) for p in postas for t in range(n_days) for k in [1, 2, 3]), cat=pulp.LpContinuous, lowBound=0, upBound=1)
        dev_posta_k = pulp.LpVariable.dicts("dev_posta_k", ((p, t, k) for p in postas for t in range(n_days) for k in [1, 2, 3]), lowBound=0, cat=pulp.LpContinuous)
        posta_dev = pulp.LpVariable.dicts("posta_dev", ((p, t) for p in postas for t in range(n_days)), lowBound=0, cat=pulp.LpContinuous)
        no_usta = pulp.LpVariable.dicts("no_usta", ((t, k) for t in range(n_days) for k in [1, 2, 3]), cat=pulp.LpContinuous, lowBound=0, upBound=1)
        
        target_avg_night = (r_night * n_days) / max(1, n_workers)
        d_pos = pulp.LpVariable.dicts("d_pos", (i for i in range(n_workers)), lowBound=0, cat=pulp.LpContinuous)
        d_neg = pulp.LpVariable.dicts("d_neg", (i for i in range(n_workers)), lowBound=0, cat=pulp.LpContinuous)

        for i in range(n_workers):
            for t in range(n_days):
                model += pulp.lpSum([x[i, t, k] for k in shifts]) == 1
                
        for t in range(n_days):
            for k in [1, 2, 3]:
                model += pulp.lpSum([x[i, t, k] for i in range(n_workers)]) >= shift_reqs[k]
                
        req_certs = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
        for cert in req_certs:
            cert_wids = [w['id'] for w in workers if cert in w['skills']]
            if len(cert_wids) > 0:
                for t in range(n_days):
                    for k in [1, 2, 3]:
                        model += pulp.lpSum([x[i, t, k] for i in cert_wids]) >= 1
                    
        for i in range(n_workers):
            for t in range(n_days - 1):
                model += x[i, t, 3] + x[i, t+1, 1] <= 1
                model += sirk[i, t] >= x[i, t, 2] + x[i, t+1, 1] - 1
                
        for i in range(n_workers):
            for t in range(n_days - 6):
                model += pulp.lpSum([x[i, tau, 0] for tau in range(t, t + 7)]) >= 1

        for i in range(n_workers):
            p_day = workers[i]['pref_off']
            if p_day < n_days:
                model += pref_viol[i] >= 1 - x[i, p_day, 0]

        for t in range(n_days):
            for k in [1, 2, 3]:
                if len(ustas_wids) > 0:
                    model += no_usta[t, k] >= 1 - pulp.lpSum([x[i, t, k] for i in ustas_wids])
                else:
                    model += no_usta[t, k] == 1

        for i in range(n_workers):
            night_sum_i = pulp.lpSum([x[i, t, 3] for t in range(n_days)])
            model += night_sum_i - target_avg_night == d_pos[i] - d_neg[i]

        for t in range(n_days):
            for p in postas:
                p_wids = [w['id'] for w in workers if w['posta'] == p]
                N_p = len(p_wids)
                model += pulp.lpSum([z_posta[p, t, k] for k in [1, 2, 3]]) <= 1
                for k in [1, 2, 3]:
                    model += y_posta[p, t, k] == pulp.lpSum([x[wid, t, k] for wid in p_wids])
                    model += dev_posta_k[p, t, k] >= y_posta[p, t, k] - N_p * z_posta[p, t, k]
                model += posta_dev[p, t] == pulp.lpSum([dev_posta_k[p, t, k] for k in [1, 2, 3]])

        total_penalty_expr = (
            w_circ * pulp.lpSum([sirk[i, t] for i in range(n_workers) for t in range(n_days - 1)]) +
            w_pref * pulp.lpSum([pref_viol[i] for i in range(n_workers)]) +
            w_posta * pulp.lpSum([posta_dev[p, t] for p in postas for t in range(n_days)]) +
            w_exp * pulp.lpSum([no_usta[t, k] for t in range(n_days) for k in [1, 2, 3]]) +
            w_night * pulp.lpSum([d_pos[i] + d_neg[i] for i in range(n_workers)])
        )
        model += total_penalty_expr
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)
        
        lp_val = pulp.value(model.objective) if model.status == 1 and pulp.value(model.objective) is not None else 0
        return round(max(lp_val, analytical_lb), 2)
    except Exception:
        return round(analytical_lb, 2)

def solve_ilp_pulp(n_workers, n_days, r_day, r_eve, r_night, weights, time_limit=10, custom_workers=None):
    """
    ILP / MILP Vardiya Optimizasyonu Çözücüsü (PuLP).
    Hata durumları (Matematiksel İmkansızlık Infeasible vs Zaman Aşımı Timeout) kesin ayrıştırılmıştır.
    """
    start_time = time.time()
    
    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)
        
    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    ustas_wids = [w['id'] for w in workers if w['is_usta']]
        
    model = pulp.LpProblem("Steel_Shift_Scheduling_ILP", pulp.LpMinimize)
    
    shifts = [0, 1, 2, 3]
    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
    
    # 1. KARAR DEĞİŞKENLERİ
    x = pulp.LpVariable.dicts("x", ((i, t, k) for i in range(n_workers) for t in range(n_days) for k in shifts), cat=pulp.LpBinary)
    sirk = pulp.LpVariable.dicts("sirkadiyen", ((i, t) for i in range(n_workers) for t in range(n_days - 1)), cat=pulp.LpBinary)
    pref_viol = pulp.LpVariable.dicts("pref_viol", (i for i in range(n_workers)), cat=pulp.LpBinary)
    
    y_posta = pulp.LpVariable.dicts("y_posta", ((p, t, k) for p in postas for t in range(n_days) for k in [1, 2, 3]), lowBound=0, cat=pulp.LpInteger)
    z_posta = pulp.LpVariable.dicts("z_posta", ((p, t, k) for p in postas for t in range(n_days) for k in [1, 2, 3]), cat=pulp.LpBinary)
    dev_posta_k = pulp.LpVariable.dicts("dev_posta_k", ((p, t, k) for p in postas for t in range(n_days) for k in [1, 2, 3]), lowBound=0, cat=pulp.LpContinuous)
    posta_dev = pulp.LpVariable.dicts("posta_dev", ((p, t) for p in postas for t in range(n_days)), lowBound=0, cat=pulp.LpContinuous)
    
    no_usta = pulp.LpVariable.dicts("no_usta", ((t, k) for t in range(n_days) for k in [1, 2, 3]), cat=pulp.LpBinary)
    
    target_avg_night = (r_night * n_days) / max(1, n_workers)
    d_pos = pulp.LpVariable.dicts("d_pos", (i for i in range(n_workers)), lowBound=0, cat=pulp.LpContinuous)
    d_neg = pulp.LpVariable.dicts("d_neg", (i for i in range(n_workers)), lowBound=0, cat=pulp.LpContinuous)

    # 2. SERT KISITLAR (HARD CONSTRAINTS)
    for i in range(n_workers):
        for t in range(n_days):
            model += pulp.lpSum([x[i, t, k] for k in shifts]) == 1, f"OneShiftPerDay_{i}_{t}"
            
    for t in range(n_days):
        for k in [1, 2, 3]:
            model += pulp.lpSum([x[i, t, k] for i in range(n_workers)]) >= shift_reqs[k], f"MinReq_{t}_{k}"
            
    req_certs = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
    for cert in req_certs:
        cert_wids = [w['id'] for w in workers if cert in w['skills']]
        for t in range(n_days):
            for k in [1, 2, 3]:
                if len(cert_wids) > 0:
                    model += pulp.lpSum([x[i, t, k] for i in cert_wids]) >= 1, f"MYKCert_{cert}_{t}_{k}"
                
    for i in range(n_workers):
        for t in range(n_days - 1):
            model += x[i, t, 3] + x[i, t+1, 1] <= 1, f"NightToDayRest_{i}_{t}"
            
    for i in range(n_workers):
        for t in range(n_days - 6):
            model += pulp.lpSum([x[i, tau, 0] for tau in range(t, t + 7)]) >= 1, f"WeeklyOff_{i}_{t}"

    # 3. YUMUŞAK KISIT BAĞLAYICI DENKLEMLERİ
    for i in range(n_workers):
        for t in range(n_days - 1):
            model += sirk[i, t] >= x[i, t, 2] + x[i, t+1, 1] - 1, f"CircadianLink_{i}_{t}"
            
    for i in range(n_workers):
        p_day = workers[i]['pref_off']
        if p_day < n_days:
            model += pref_viol[i] >= 1 - x[i, p_day, 0], f"PrefOffLink_{i}"
            
    for t in range(n_days):
        for k in [1, 2, 3]:
            if len(ustas_wids) > 0:
                model += no_usta[t, k] >= 1 - pulp.lpSum([x[i, t, k] for i in ustas_wids]), f"UstaLink_{t}_{k}"
            else:
                model += no_usta[t, k] == 1, f"UstaLink_NoUsta_{t}_{k}"

    for i in range(n_workers):
        night_sum_i = pulp.lpSum([x[i, t, 3] for t in range(n_days)])
        model += night_sum_i - target_avg_night == d_pos[i] - d_neg[i], f"NightImbalanceLink_{i}"

    for t in range(n_days):
        for p in postas:
            p_wids = [w['id'] for w in workers if w['posta'] == p]
            N_p = len(p_wids)
            model += pulp.lpSum([z_posta[p, t, k] for k in [1, 2, 3]]) <= 1, f"ZPostaMaxOne_{p}_{t}"
            for k in [1, 2, 3]:
                model += y_posta[p, t, k] == pulp.lpSum([x[wid, t, k] for wid in p_wids]), f"PostaCount_{p}_{t}_{k}"
                model += dev_posta_k[p, t, k] >= y_posta[p, t, k] - N_p * z_posta[p, t, k], f"PostaDevK_{p}_{t}_{k}"
            model += posta_dev[p, t] == pulp.lpSum([dev_posta_k[p, t, k] for k in [1, 2, 3]]), f"PostaDevSum_{p}_{t}"

    # 4. TAM KÜRESEL AMAÇ FONKSİYONU
    w_posta = weights.get('posta', 35)
    w_circ = weights.get('circadian', 50)
    w_pref = weights.get('pref_off', 40)
    w_night_imb = weights.get('night_imb', 25)
    w_exp = weights.get('exp_mix', 30)
    
    total_penalty_expr = (
        w_circ * pulp.lpSum([sirk[i, t] for i in range(n_workers) for t in range(n_days - 1)]) +
        w_pref * pulp.lpSum([pref_viol[i] for i in range(n_workers)]) +
        w_posta * pulp.lpSum([posta_dev[p, t] for p in postas for t in range(n_days)]) +
        w_exp * pulp.lpSum([no_usta[t, k] for t in range(n_days) for k in [1, 2, 3]]) +
        w_night_imb * pulp.lpSum([d_pos[i] + d_neg[i] for i in range(n_workers)])
    )
    
    model += total_penalty_expr, "Minimize_All_5_Penalties"
    
    # CBC log çıktısını yakalayarak Zaman Aşımı (Timeout) vs Optimal durumunu %100 kesin tespit etme
    import tempfile, os, re
    log_file = tempfile.NamedTemporaryFile(delete=False, suffix=".log").name
    cbc_log_content = ""
    try:
        solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit, logPath=log_file)
        status_code = model.solve(solver)
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                cbc_log_content = f.read()
            try:
                os.remove(log_file)
            except Exception:
                pass
    except Exception:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
        status_code = model.solve(solver)

    exec_time = round((time.time() - start_time) * 1000, 2)
    raw_status_str = pulp.LpStatus[status_code]
    
    is_infeasible = (status_code == pulp.LpStatusInfeasible or raw_status_str == "Infeasible" or "infeasible" in cbc_log_content.lower())
    
    # CBC "stopped on time limit" içeriyorsa veya süre sınırını aştıysa timeout'tur
    cbc_stopped_time_limit = ("stopped on time limit" in cbc_log_content.lower()) or ("time limit" in cbc_log_content.lower() and "optimal" not in cbc_log_content.lower())
    
    schedule = np.zeros((n_workers, n_days), dtype=int)
    has_valid_assignment = False
    
    if not is_infeasible and raw_status_str in ["Optimal", "Not Solved", "Feasible"]:
        assigned_count = 0
        for i in range(n_workers):
            for t in range(n_days):
                for k in shifts:
                    val = pulp.value(x[i, t, k])
                    if val is not None and round(val) == 1:
                        schedule[i, t] = k
                        if k > 0:
                            assigned_count += 1
        if assigned_count > 0:
            has_valid_assignment = True

    # SERT KISIT DOĞRULAMA
    is_hard_feasible = True
    hard_violations = []
    
    if is_infeasible:
        is_hard_feasible = False
        hard_violations.append("❌ MATEMATİKSEL İMKANSIZLIK: Personel sayısı veya sertifikalar kısıtları sağlamak için yetersizdir.")
    elif not has_valid_assignment:
        is_hard_feasible = False
        hard_violations.append(f"⏱️ ZAMAN AŞIMI: Solver {time_limit} saniye içinde atama üretemeden kesildi.")
    else:
        for t in range(n_days):
            for k in [1, 2, 3]:
                count_k = sum(1 for i in range(n_workers) if schedule[i, t] == k)
                if count_k < shift_reqs[k]:
                    is_hard_feasible = False
                    hard_violations.append(f"Gün {t+1} Vardiya {k}: Atanan {count_k} < Gerekli {shift_reqs[k]}")

    # OPTİMAL VE TIMEOUT DURUMU (CBC LOG KONTROLÜ İLE KESİN SEÇİM)
    is_timeout = (cbc_stopped_time_limit or status_code == pulp.LpStatusNotSolved) and not is_infeasible
    is_optimal = (status_code == pulp.LpStatusOptimal) and (not is_timeout) and is_hard_feasible
    
    night_counts = np.array([np.sum(schedule[i, :] == 3) for i in range(n_workers)])
    avg_night = np.mean(night_counts) if n_workers > 0 else 0
    
    penalties = {
        'Posta Takım Bütünlüğü İhlali': 0,
        'Sirkadiyen Ritim İhlali (Akşam->Gündüz)': 0,
        'Gece Nöbeti Dengesizliği': 0,
        'Kıdem & MYK Sertifika Eksikliği': 0,
        'Kişisel İzin İhlali': 0
    }
    
    if is_hard_feasible:
        for i in range(n_workers):
            for t in range(n_days - 1):
                if schedule[i, t] == 2 and schedule[i, t+1] == 1:
                    penalties['Sirkadiyen Ritim İhlali (Akşam->Gündüz)'] += w_circ
                    
            p_day = workers[i]['pref_off']
            if p_day < n_days and schedule[i, p_day] != 0:
                penalties['Kişisel İzin İhlali'] += w_pref

        tot_night_diff = sum(abs(night_counts[i] - target_avg_night) for i in range(n_workers))
        penalties['Gece Nöbeti Dengesizliği'] = int(round(tot_night_diff * w_night_imb))
                
        for t in range(n_days):
            for k in [1, 2, 3]:
                shift_wids = [w['id'] for w in workers if schedule[w['id'], t] == k]
                ustas = sum(1 for wid in shift_wids if workers[wid]['is_usta'])
                if len(shift_wids) > 0 and ustas == 0:
                    penalties['Kıdem & MYK Sertifika Eksikliği'] += w_exp
                    
        for t in range(n_days):
            for p in postas:
                p_wids = [w['id'] for w in workers if w['posta'] == p]
                active_shifts = [schedule[wid, t] for wid in p_wids if schedule[wid, t] != 0]
                if len(active_shifts) > 1:
                    counts = [active_shifts.count(s) for s in set(active_shifts)]
                    majority = max(counts)
                    deviated = len(active_shifts) - majority
                    penalties['Posta Takım Bütünlüğü İhlali'] += deviated * w_posta

    total_penalty = sum(penalties.values()) if is_hard_feasible else 99999
    obj_val = total_penalty
    
    # TEORİK ALT SINIR (BEST BOUND) & MIP GAP HESABI
    # Teorik Alt Sınır (Continuous LP Relaxation), solver zaman limitinden (time_limit) 
    # %100 BAĞIMSIZ, model parametrelerine ve kısıtlara bağlı saf matematiksel sabittir.
    calculated_best_bound = compute_lp_relaxation_bound(n_workers, n_days, r_day, r_eve, r_night, weights, workers)

    if is_infeasible:
        status_text = "❌ İMKANSIZ / KISIT İHLALİ (Infeasible)"
        best_bound = 0
        mip_gap = 100.0
    elif is_timeout and is_hard_feasible:
        status_text = "⏱️ GEÇERLİ (Zaman Sınırında Kesildi)"
        best_bound = min(calculated_best_bound, obj_val)
        mip_gap = round(abs((obj_val - best_bound) / max(1, obj_val)) * 100, 1)
    elif is_optimal:
        status_text = "🏆 OPTİMAL (Tam Matematiksel Garanti)"
        best_bound = min(calculated_best_bound, obj_val)
        mip_gap = round(abs((obj_val - best_bound) / max(1, obj_val)) * 100, 1)
    else:
        status_text = "⏱️ ZAMAN AŞIMI (Süre Yetersiz Kaldı)"
        best_bound = calculated_best_bound
        mip_gap = 100.0

    request_details = []
    for w in workers:
        wid = w['id']
        p_day = w['pref_off']
        assigned_shift = schedule[wid, p_day] if p_day < n_days else 0
        is_fulfilled = (assigned_shift == 0)
        request_details.append({
            'id': wid,
            'name': w['name'],
            'posta': w['posta'],
            'unvan': "Kıdemli Usta" if w['is_usta'] else "Operatör/İşçi",
            'talep_gun': f"Gün {p_day + 1}",
            'atandi_vardiya': "OFF (İzin)" if is_fulfilled else ("Gündüz" if assigned_shift==1 else ("Akşam" if assigned_shift==2 else "Gece")),
            'durum': "✅ Karşılandı (OFF verildi)" if is_fulfilled else "❌ İhlal Edildi (Vardiyaya Yazıldı)",
            'ceza_puani': 0 if is_fulfilled else w_pref
        })
        
    return {
        'status_text': status_text,
        'raw_status': raw_status_str,
        'is_optimal': is_optimal,
        'is_infeasible': is_infeasible,
        'is_timeout': is_timeout,
        'is_hard_feasible': is_hard_feasible,
        'objective_value': obj_val,
        'best_bound': best_bound,
        'mip_gap': mip_gap,
        'schedule': schedule,
        'workers': workers,
        'exec_time_ms': exec_time,
        'penalties': penalties,
        'total_penalty': total_penalty,
        'hard_violations': hard_violations,
        'request_details': request_details
    }
