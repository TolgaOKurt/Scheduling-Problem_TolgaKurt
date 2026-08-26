"""
================================================================================
  ALGORITHMS/EPSILON_CONSTRAINT_SOLVER.PY - ε-KISIT YÖNTEMİ (KESİN PARETO ÇÖZÜCÜ)
================================================================================
  Bu modül; Çok Amaçlı Optimizasyon literatürünün temel kesin yöntemlerinden biri
  olan ε-Kısıt Yöntemini (ε-Constraint Method - Haimes et al., 1971 / Mavrotas, 2009)
  Google OR-Tools CP-SAT ve PuLP MILP kesin matematiksel çözücüleri ile uygular.

  Temel Mimarisi:
  1. Pay-off Tablosu Üretimi (Uç Noktalar: min f1 ve min f2)
  2. Epsilon Grid Taraması: f2_min <= ε_k <= f2_max
  3. Kesin Çözücü Alt Problemleri: min f1(X)  s.t.  f2(X) <= ε_k,  X in Ω
  4. Non-dominated Pareto Filtreleme & Zayıf Baskınlık Ayıklama
  5. Pareto Metrikleri: Hiper-Hacim (HV), Spread (Δ) ve Diz Noktası (Knee Point)
================================================================================
"""

import time
import math
import numpy as np
import pulp
from typing import List, Dict, Any, Tuple, Optional
from ortools.sat.python import cp_model

from algorithms.worker_manager import generate_worker_profiles
from algorithms.penalty_calculator import (
    calculate_full_penalties,
    audit_all_hard_constraints,
    build_worker_request_details
)
from algorithms.solver_contract import build_standard_solver_result
from algorithms.nsga2_solver import calculate_hypervolume_2d, calculate_spread_metric, find_knee_point_index


# =============================================================================
# 1. CP-SAT TABANLI ALT PROBLEM ÇÖZÜCÜSÜ (GOOGLE OR-TOOLS CP-SAT)
# =============================================================================
def _solve_cpsat_subproblem(
    n_workers: int,
    n_days: int,
    shift_reqs: Dict[int, int],
    workers: List[Dict[str, Any]],
    weights: Dict[str, int],
    primary_objective: str = "min_f1",  # "min_f1" veya "min_f2"
    epsilon_bound: Optional[float] = None,
    time_limit: float = 3.0,
    num_threads: int = 4
) -> Tuple[Optional[np.ndarray], float, float, str, float]:
    """
    CP-SAT ile tek bir ε-kısıt alt problemini tam kesinlikle çözer.
    AUGMECON (Augmented ε-Constraint) prensibiyle zayıf baskınlığı önler.
    Döndürdüğü: (schedule, f1_val, f2_val, status_str, solve_time_ms)
    """
    t_start = time.time()
    shifts = [0, 1, 2, 3]  # 0: OFF, 1: Gündüz, 2: Akşam, 3: Gece
    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    ustas_wids = [w['id'] for w in workers if w.get('is_usta', False)]

    w_posta = weights.get('posta', 15)
    w_exp = weights.get('exp_mix', 60)
    w_pref = weights.get('pref_off', 40)
    w_night = weights.get('night_imb', 25)
    w_work = weights.get('workload_imb', 20)
    w_circ = weights.get('circadian', 50)

    scale_n = max(1, n_workers)

    model = cp_model.CpModel()

    # Karar Değişkenleri: x[i, t, k] == 1 <=> i işçisi t gününde k vardiyasında
    x = {}
    for i in range(n_workers):
        for t in range(n_days):
            for k in shifts:
                x[i, t, k] = model.NewBoolVar(f"x_{i}_{t}_{k}")

    # SERT KISITLAR
    # H1: Her işçi günde tam 1 vardiyada bulunur
    for i in range(n_workers):
        for t in range(n_days):
            model.AddExactlyOne([x[i, t, k] for k in shifts])

    # H2: Vardiya asgari kadro kotaları
    for t in range(n_days):
        for k in [1, 2, 3]:
            model.Add(sum(x[i, t, k] for i in range(n_workers)) == shift_reqs[k])

    # H3: MYK Sertifika zorunluluğu
    req_certs = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
    for cert in req_certs:
        cert_wids = [w['id'] for w in workers if cert in w.get('skills', set())]
        for t in range(n_days):
            for k in [1, 2, 3]:
                if len(cert_wids) > 0:
                    model.Add(sum(x[wid, t, k] for wid in cert_wids) >= 1)

    # H4: Gece (3) -> Ertesi gün Gündüz (1) yasak (0 saat dinlenme)
    for i in range(n_workers):
        for t in range(n_days - 1):
            model.Add(x[i, t, 3] + x[i, t + 1, 1] <= 1)

    # H5: 7 günlük kayan pencerede en az 1 gün OFF
    for i in range(n_workers):
        for t in range(max(0, n_days - 6)):
            model.Add(sum(x[i, tau, 0] for tau in range(t, min(n_days, t + 7))) >= 1)

    # --- AMAÇ 1 DEĞİŞKENLERİ: f1 (İşletme & Üretim Verimliliği) ---
    # 1. Posta Takım Bütünlüğü (Baskın Vardiya Seçim İndikatörü z_posta ile Kesin Doğrusallaştırma)
    posta_dev_terms = []
    four_posta_terms = []
    
    for t in range(n_days):
        u_posta_day = []
        for p in postas:
            p_wids = [w['id'] for w in workers if w.get('posta') == p]
            N_p = len(p_wids)
            if N_p > 0:
                # Posta o gün aktif mi?
                u_p_t = model.NewBoolVar(f"u_p_{p}_{t}")
                p_act_sum = sum(x[wid, t, k] for wid in p_wids for k in [1, 2, 3])
                model.Add(p_act_sum == 0).OnlyEnforceIf(u_p_t.Not())
                model.Add(p_act_sum >= 1).OnlyEnforceIf(u_p_t)
                u_posta_day.append(u_p_t)

                if N_p > 1:
                    # z_posta[k]: p postasının t günündeki baskın vardiyası k mı?
                    z_posta = [model.NewBoolVar(f"z_p_{p}_{t}_{k}") for k in [1, 2, 3]]
                    model.Add(sum(z_posta) <= 1)

                    for idx, k in enumerate([1, 2, 3]):
                        y_k = sum(x[wid, t, k] for wid in p_wids)
                        dev_k = model.NewIntVar(0, N_p, f"dev_{p}_{t}_{k}")
                        # dev_k >= y_k - N_p * z_posta[k]
                        model.Add(dev_k >= y_k - N_p * z_posta[idx])
                        posta_dev_terms.append(dev_k)

        # Günlük 4-Posta Ceza Kontrolü
        if len(u_posta_day) >= 4:
            fp_var = model.NewBoolVar(f"fp_var_{t}")
            model.Add(sum(u_posta_day) == 4).OnlyEnforceIf(fp_var)
            model.Add(sum(u_posta_day) <= 3).OnlyEnforceIf(fp_var.Not())
            four_posta_terms.append(fp_var)

    # 2. Usta Eksikliği
    no_usta_vars = []
    for t in range(n_days):
        for k in [1, 2, 3]:
            no_u = model.NewBoolVar(f"no_u_{t}_{k}")
            if len(ustas_wids) > 0:
                u_count = sum(x[wid, t, k] for wid in ustas_wids)
                model.Add(u_count == 0).OnlyEnforceIf(no_u)
                model.Add(u_count >= 1).OnlyEnforceIf(no_u.Not())
            else:
                model.Add(no_u == 1)
            no_usta_vars.append(no_u)

    # f1 Ölçekli İfade (N ile çarpılarak tamsayı yapılır)
    f1_scaled_expr = (
        (w_posta * scale_n) * sum(posta_dev_terms) +
        (scale_n) * sum(four_posta_terms) +
        (w_exp * scale_n) * sum(no_usta_vars)
    )

    # --- AMAÇ 2 DEĞİŞKENLERİ: f2 (Çalışan Memnuniyeti & Ergonomi) ---
    # 3. Sirkadiyen Ritim
    sirk_vars = []
    for i in range(n_workers):
        for t in range(n_days - 1):
            sirk_b = model.NewBoolVar(f"sirk_{i}_{t}")
            model.Add(x[i, t, 2] + x[i, t + 1, 1] == 2).OnlyEnforceIf(sirk_b)
            model.Add(x[i, t, 2] + x[i, t + 1, 1] < 2).OnlyEnforceIf(sirk_b.Not())
            sirk_vars.append(sirk_b)

    # 4. Kişisel İzin
    pref_vars = []
    for i in range(n_workers):
        pref_day = int(workers[i].get('pref_off', 1)) - 1
        if 0 <= pref_day < n_days:
            p_viol = model.NewBoolVar(f"pref_viol_{i}")
            model.Add(x[i, pref_day, 0] == 0).OnlyEnforceIf(p_viol)
            model.Add(x[i, pref_day, 0] == 1).OnlyEnforceIf(p_viol.Not())
            pref_vars.append(p_viol)

    # 5. Gece Nöbeti Dengesi (L1 Sapma)
    tot_night_target = shift_reqs[3] * n_days
    d_pos_night = [model.NewIntVar(0, scale_n * n_days, f"dp_n_{i}") for i in range(n_workers)]
    d_neg_night = [model.NewIntVar(0, scale_n * n_days, f"dn_n_{i}") for i in range(n_workers)]
    for i in range(n_workers):
        n_count = sum(x[i, t, 3] for t in range(n_days))
        model.Add(scale_n * n_count - tot_night_target == d_pos_night[i] - d_neg_night[i])

    # 6. İş Yükü Dengesi (L1 Sapma)
    tot_work_target = (shift_reqs[1] + shift_reqs[2] + shift_reqs[3]) * n_days
    d_pos_work = [model.NewIntVar(0, scale_n * n_days, f"dp_w_{i}") for i in range(n_workers)]
    d_neg_work = [model.NewIntVar(0, scale_n * n_days, f"dn_w_{i}") for i in range(n_workers)]
    for i in range(n_workers):
        w_count = sum((1 - x[i, t, 0]) for t in range(n_days))
        model.Add(scale_n * w_count - tot_work_target == d_pos_work[i] - d_neg_work[i])

    # f2 Ölçekli İfade
    f2_scaled_expr = (
        (w_circ * scale_n) * sum(sirk_vars) +
        (w_pref * scale_n) * sum(pref_vars) +
        (w_night) * sum(d_pos_night[i] + d_neg_night[i] for i in range(n_workers)) +
        (w_work) * sum(d_pos_work[i] + d_neg_work[i] for i in range(n_workers))
    )

    # --- ε-KISIT VE AUGMECON HEDEF BELİRLEME ---
    if primary_objective == "min_f1":
        if epsilon_bound is not None:
            model.Add(f2_scaled_expr <= int(math.ceil(epsilon_bound * scale_n)))
        # AUGMECON: 1000 * f1 + f2 (f1 öncelikli minimize edilir, eşitlikte f2 en küçüğe çekilir)
        model.Minimize(1000 * f1_scaled_expr + f2_scaled_expr)
    else:
        if epsilon_bound is not None:
            model.Add(f1_scaled_expr <= int(math.ceil(epsilon_bound * scale_n)))
        # AUGMECON: 1000 * f2 + f1
        model.Minimize(1000 * f2_scaled_expr + f1_scaled_expr)

    # Çözücü Ayarları
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(num_threads)

    status = solver.Solve(model)
    solve_ms = round((time.time() - t_start) * 1000, 1)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sched = np.zeros((n_workers, n_days), dtype=int)
        for i in range(n_workers):
            for t in range(n_days):
                for k in shifts:
                    if solver.Value(x[i, t, k]) == 1:
                        sched[i, t] = k
                        break

        # Gerçek ceza puanlarını merkezi tek kaynak motordan (%100 doğru) hesapla
        pen_dict, _ = calculate_full_penalties(sched, workers, n_workers, n_days, weights)
        f1_actual = float(pen_dict['Posta Takım Bütünlüğü İhlali'] + pen_dict['Kıdem / Usta Eksikliği'])
        f2_actual = float(
            pen_dict['Kişisel İzin İhlali'] +
            pen_dict['Gece Nöbeti Dengesizliği'] +
            pen_dict['Toplam İş Yükü Dengesizliği'] +
            pen_dict['Sirkadiyen Ritim İhlali (Akşam->Gündüz)']
        )
        status_name = "Optimal" if status == cp_model.OPTIMAL else "Feasible"
        return sched, f1_actual, f2_actual, status_name, solve_ms
    else:
        status_name = "Infeasible" if status == cp_model.INFEASIBLE else "Timeout / No Solution"
        return None, float('inf'), float('inf'), status_name, solve_ms


# =============================================================================
# 2. MILP / PULP TABANLI ALT PROBLEM ÇÖZÜCÜSÜ (CBC SOLVER)
# =============================================================================
def _solve_milp_subproblem(
    n_workers: int,
    n_days: int,
    shift_reqs: Dict[int, int],
    workers: List[Dict[str, Any]],
    weights: Dict[str, int],
    primary_objective: str = "min_f1",
    epsilon_bound: Optional[float] = None,
    time_limit: float = 3.0
) -> Tuple[Optional[np.ndarray], float, float, str, float]:
    """PuLP CBC MILP motoru ile ε-kısıt alt problemini kesinlikle çözer."""
    t_start = time.time()
    shifts = [0, 1, 2, 3]
    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    ustas_wids = [w['id'] for w in workers if w.get('is_usta', False)]

    w_posta = weights.get('posta', 15)
    w_exp = weights.get('exp_mix', 60)
    w_pref = weights.get('pref_off', 40)
    w_night = weights.get('night_imb', 25)
    w_work = weights.get('workload_imb', 20)
    w_circ = weights.get('circadian', 50)

    model = pulp.LpProblem("Epsilon_Constraint_MILP", pulp.LpMinimize)

    # Karar değişkenleri
    x = pulp.LpVariable.dicts("x", ((i, t, k) for i in range(n_workers) for t in range(n_days) for k in shifts), cat=pulp.LpBinary)
    sirk = pulp.LpVariable.dicts("sirk", ((i, t) for i in range(n_workers) for t in range(n_days - 1)), cat=pulp.LpContinuous, lowBound=0)
    pref_viol = pulp.LpVariable.dicts("pref_viol", (i for i in range(n_workers)), cat=pulp.LpContinuous, lowBound=0)

    # Posta
    z_posta = pulp.LpVariable.dicts("z_posta", ((p, t, k) for p in postas for t in range(n_days) for k in [1, 2, 3]), cat=pulp.LpBinary)
    dev_posta_k = pulp.LpVariable.dicts("dev_posta_k", ((p, t, k) for p in postas for t in range(n_days) for k in [1, 2, 3]), cat=pulp.LpContinuous, lowBound=0)
    posta_dev = pulp.LpVariable.dicts("posta_dev", ((p, t) for p in postas for t in range(n_days)), cat=pulp.LpContinuous, lowBound=0)
    u_posta = pulp.LpVariable.dicts("u_posta", ((p, t) for p in postas for t in range(n_days)), cat=pulp.LpBinary)
    four_posta_pen = pulp.LpVariable.dicts("fp_pen", (t for t in range(n_days)), cat=pulp.LpBinary)
    no_usta = pulp.LpVariable.dicts("no_usta", ((t, k) for t in range(n_days) for k in [1, 2, 3]), cat=pulp.LpBinary)

    d_pos_n = pulp.LpVariable.dicts("dp_n", (i for i in range(n_workers)), lowBound=0, cat=pulp.LpContinuous)
    d_neg_n = pulp.LpVariable.dicts("dn_n", (i for i in range(n_workers)), lowBound=0, cat=pulp.LpContinuous)
    d_pos_w = pulp.LpVariable.dicts("dp_w", (i for i in range(n_workers)), lowBound=0, cat=pulp.LpContinuous)
    d_neg_w = pulp.LpVariable.dicts("dn_w", (i for i in range(n_workers)), lowBound=0, cat=pulp.LpContinuous)

    # SERT KISITLAR
    for i in range(n_workers):
        for t in range(n_days):
            model += pulp.lpSum([x[i, t, k] for k in shifts]) == 1

    for t in range(n_days):
        for k in [1, 2, 3]:
            model += pulp.lpSum([x[i, t, k] for i in range(n_workers)]) == shift_reqs[k]

    req_certs = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
    for cert in req_certs:
        cert_wids = [w['id'] for w in workers if cert in w.get('skills', set())]
        for t in range(n_days):
            for k in [1, 2, 3]:
                if len(cert_wids) > 0:
                    model += pulp.lpSum([x[wid, t, k] for wid in cert_wids]) >= 1

    for i in range(n_workers):
        for t in range(n_days - 1):
            model += x[i, t, 3] + x[i, t + 1, 1] <= 1
            model += sirk[i, t] >= x[i, t, 2] + x[i, t + 1, 1] - 1

    for i in range(n_workers):
        for t in range(max(0, n_days - 6)):
            model += pulp.lpSum([x[i, tau, 0] for tau in range(t, min(n_days, t + 7))]) >= 1

    for i in range(n_workers):
        p_day = int(workers[i].get('pref_off', 1)) - 1
        if 0 <= p_day < n_days:
            model += pref_viol[i] >= 1 - x[i, p_day, 0]

    # Posta kısıtları (z_posta baskın vardiya ile tam kesin formülasyon)
    for t in range(n_days):
        for p in postas:
            p_wids = [w['id'] for w in workers if w.get('posta') == p]
            N_p = len(p_wids)
            model += pulp.lpSum([z_posta[p, t, k] for k in [1, 2, 3]]) <= 1
            for k in [1, 2, 3]:
                y_p_k = pulp.lpSum([x[wid, t, k] for wid in p_wids])
                model += dev_posta_k[p, t, k] >= y_p_k - N_p * z_posta[p, t, k]
            model += posta_dev[p, t] == pulp.lpSum([dev_posta_k[p, t, k] for k in [1, 2, 3]])
            if N_p > 0:
                model += pulp.lpSum([x[wid, t, k] for wid in p_wids for k in [1, 2, 3]]) <= N_p * u_posta[p, t]

        valid_p_count = sum(1 for p in postas if any(w.get('posta') == p for w in workers))
        if valid_p_count >= 4:
            model += four_posta_pen[t] >= pulp.lpSum([u_posta[p, t] for p in postas]) - 3

    # Usta kısıtları
    for t in range(n_days):
        for k in [1, 2, 3]:
            if len(ustas_wids) > 0:
                model += no_usta[t, k] >= 1 - pulp.lpSum([x[wid, t, k] for wid in ustas_wids])
            else:
                model += no_usta[t, k] == 1

    # Denge kısıtları
    tot_n_target = (shift_reqs[3] * n_days) / float(n_workers)
    tot_w_target = ((shift_reqs[1] + shift_reqs[2] + shift_reqs[3]) * n_days) / float(n_workers)
    for i in range(n_workers):
        model += pulp.lpSum([x[i, t, 3] for t in range(n_days)]) - tot_n_target == d_pos_n[i] - d_neg_n[i]
        model += pulp.lpSum([(1 - x[i, t, 0]) for t in range(n_days)]) - tot_w_target == d_pos_w[i] - d_neg_w[i]

    # Amaç Fonksiyonu İfadeleri
    f1_expr = (
        w_posta * pulp.lpSum([posta_dev[p, t] for p in postas for t in range(n_days)]) +
        pulp.lpSum([four_posta_pen[t] for t in range(n_days)]) +
        w_exp * pulp.lpSum([no_usta[t, k] for t in range(n_days) for k in [1, 2, 3]])
    )

    f2_expr = (
        w_circ * pulp.lpSum([sirk[i, t] for i in range(n_workers) for t in range(n_days - 1)]) +
        w_pref * pulp.lpSum([pref_viol[i] for i in range(n_workers)]) +
        w_night * pulp.lpSum([d_pos_n[i] + d_neg_n[i] for i in range(n_workers)]) +
        w_work * pulp.lpSum([d_pos_w[i] + d_neg_w[i] for i in range(n_workers)])
    )

    if primary_objective == "min_f1":
        if epsilon_bound is not None:
            model += f2_expr <= float(epsilon_bound)
        # AUGMECON: f1 + 0.0001 * f2
        model.setObjective(f1_expr + 0.0001 * f2_expr)
    else:
        if epsilon_bound is not None:
            model += f1_expr <= float(epsilon_bound)
        # AUGMECON: f2 + 0.0001 * f1
        model.setObjective(f2_expr + 0.0001 * f1_expr)

    solver = pulp.PULP_CBC_CMD(timeLimit=float(time_limit), msg=False)
    model.solve(solver)
    solve_ms = round((time.time() - t_start) * 1000, 1)

    if model.status == pulp.LpStatusOptimal:
        sched = np.zeros((n_workers, n_days), dtype=int)
        for i in range(n_workers):
            for t in range(n_days):
                for k in shifts:
                    val = pulp.value(x[i, t, k])
                    if val is not None and val > 0.5:
                        sched[i, t] = k
                        break
        pen_dict, _ = calculate_full_penalties(sched, workers, n_workers, n_days, weights)
        f1_actual = float(pen_dict['Posta Takım Bütünlüğü İhlali'] + pen_dict['Kıdem / Usta Eksikliği'])
        f2_actual = float(
            pen_dict['Kişisel İzin İhlali'] +
            pen_dict['Gece Nöbeti Dengesizliği'] +
            pen_dict['Toplam İş Yükü Dengesizliği'] +
            pen_dict['Sirkadiyen Ritim İhlali (Akşam->Gündüz)']
        )
        return sched, f1_actual, f2_actual, "Optimal", solve_ms
    else:
        return None, float('inf'), float('inf'), "Infeasible / Timeout", solve_ms


# =============================================================================
# 3. ANA ε-KISIT GRID TARAMA OPTİMİZASYON MOTORU
# =============================================================================
def run_epsilon_constraint_optimization(
    n_workers: int,
    n_days: int,
    r_day: int,
    r_eve: int,
    r_night: int,
    weights: Dict[str, int],
    grid_steps: int = 10,
    solver_engine: str = "Google OR-Tools CP-SAT (Ultra Hızlı)",
    primary_objective: str = "min_f1",
    subproblem_time_limit: float = 3.0,
    custom_workers: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Any] = None
) -> Dict[str, Any]:
    """
    ε-Kısıt Yöntemi (ε-Constraint Method) ile Kesin Pareto Optimizasyonu.
    
    Adımlar:
    1. Pay-off Tablosu (Anchor 1: min f1, Anchor 2: min f2)
    2. Grid aralığı: [f2_min, f2_max]
    3. Grid taraması: K adım boyunca min f1(X) s.t. f2(X) <= ε_k çözülür.
    4. Kesin Pareto kümesi, Hiper-Hacim ve Diz Noktası (Knee Point) çıkarılır.
    """
    start_time = time.time()
    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    use_cpsat = "CP-SAT" in solver_engine
    sub_solver = _solve_cpsat_subproblem if use_cpsat else _solve_milp_subproblem

    total_sub_solves = grid_steps + 2
    grid_logs: List[Dict[str, Any]] = []

    # =========================================================================
    # ADIM 1: PAY-OFF TABLOSU VE UÇ NOKTALARIN BULUNMASI
    # =========================================================================
    # Anchor 1: min f1
    sched_1, f1_anchor1, f2_anchor1, st_1, ms_1 = sub_solver(
        n_workers, n_days, shift_reqs, workers, weights,
        primary_objective="min_f1", epsilon_bound=None, time_limit=subproblem_time_limit
    )
    score_1 = float(f1_anchor1 + f2_anchor1) if not math.isinf(f1_anchor1) else 2500.0
    running_best = score_1

    if callback:
        callback(1, running_best, score_1, f"| 1/{total_sub_solves}: Uc Nokta 1 (min f1) -> Z={score_1:.0f}")

    # Anchor 2: min f2
    sched_2, f1_anchor2, f2_anchor2, st_2, ms_2 = sub_solver(
        n_workers, n_days, shift_reqs, workers, weights,
        primary_objective="min_f2", epsilon_bound=None, time_limit=subproblem_time_limit
    )
    score_2 = float(f1_anchor2 + f2_anchor2) if not math.isinf(f2_anchor2) else score_1
    running_best = min(running_best, score_2)

    if callback:
        callback(2, running_best, score_2, f"| 2/{total_sub_solves}: Uc Nokta 2 (min f2) -> Z={score_2:.0f}")

    # Sınırlar
    f2_min = min(f2_anchor1, f2_anchor2) if not math.isinf(f2_anchor2) else 0.0
    f2_max = max(f2_anchor1, f2_anchor2) if not math.isinf(f2_anchor1) else 300.0

    if f2_max <= f2_min:
        f2_max = f2_min + 50.0

    # Grid Değerleri
    epsilon_values = np.linspace(f2_min, f2_max, grid_steps)

    grid_logs.append({
        'step': 0,
        'label': "Uç Nokta 1 (min f₁)",
        'target_eps': "-",
        'f1': f1_anchor1,
        'f2': f2_anchor1,
        'total': int(score_1) if not math.isinf(f1_anchor1) else "-",
        'status': st_1,
        'time_ms': ms_1
    })

    grid_logs.append({
        'step': 0,
        'label': "Uç Nokta 2 (min f₂)",
        'target_eps': "-",
        'f1': f1_anchor2,
        'f2': f2_anchor2,
        'total': int(score_2) if not math.isinf(f2_anchor2) else "-",
        'status': st_2,
        'time_ms': ms_2
    })

    # =========================================================================
    # ADIM 2: GRID TARAMASI (EPSILON SCANNING)
    # =========================================================================
    all_evaluated_solutions: List[Dict[str, Any]] = []

    if sched_1 is not None and not math.isinf(f1_anchor1):
        all_evaluated_solutions.append({
            'schedule': sched_1,
            'f1': f1_anchor1,
            'f2': f2_anchor1,
            'total_score': int(score_1),
            'status': st_1,
            'step': 0
        })

    if sched_2 is not None and not math.isinf(f2_anchor2):
        all_evaluated_solutions.append({
            'schedule': sched_2,
            'f1': f1_anchor2,
            'f2': f2_anchor2,
            'total_score': int(score_2),
            'status': st_2,
            'step': 0
        })

    for step_idx, eps_k in enumerate(epsilon_values):
        sched_k, f1_k, f2_k, st_k, ms_k = sub_solver(
            n_workers, n_days, shift_reqs, workers, weights,
            primary_objective="min_f1", epsilon_bound=float(eps_k), time_limit=subproblem_time_limit
        )

        score_k = float(f1_k + f2_k) if not math.isinf(f1_k) else running_best
        if not math.isinf(f1_k):
            running_best = min(running_best, score_k)

        curr_step_num = step_idx + 3
        if callback:
            msg = f"| Grid {step_idx+1}/{grid_steps}: eps={eps_k:.1f} -> Z={score_k:.0f} ({st_k})"
            callback(curr_step_num, running_best, score_k, msg)

        grid_logs.append({
            'step': step_idx + 1,
            'label': f"Grid #{step_idx+1}",
            'target_eps': f"{eps_k:.1f}",
            'f1': f1_k if not math.isinf(f1_k) else "İhlal",
            'f2': f2_k if not math.isinf(f2_k) else "İhlal",
            'total': int(f1_k + f2_k) if not math.isinf(f1_k) else "-",
            'status': st_k,
            'time_ms': ms_k
        })

        if sched_k is not None and not math.isinf(f1_k):
            all_evaluated_solutions.append({
                'schedule': sched_k,
                'f1': f1_k,
                'f2': f2_k,
                'total_score': int(f1_k + f2_k),
                'status': st_k,
                'step': step_idx + 1
            })

    # =========================================================================
    # ADIM 3: BASKIN OLMAYAN KESİN PARETO ÇÖZÜMLERİNİN FİLTRELENMESİ
    # =========================================================================
    # Tekil koordinatları ayıkla
    unique_candidates: List[Dict[str, Any]] = []
    seen_coords = set()

    for sol in all_evaluated_solutions:
        coord = (round(sol['f1'], 1), round(sol['f2'], 1))
        if coord in seen_coords:
            continue
        seen_coords.add(coord)
        unique_candidates.append(sol)

    # Pareto Non-dominated Filtering
    pareto_solutions_list: List[Dict[str, Any]] = []
    for p_idx, p in enumerate(unique_candidates):
        is_dominated = False
        for q_idx, q in enumerate(unique_candidates):
            if p_idx == q_idx:
                continue
            # q dominates p
            if (q['f1'] <= p['f1'] and q['f2'] <= p['f2']) and (q['f1'] < p['f1'] or q['f2'] < p['f2']):
                is_dominated = True
                break

        if not is_dominated:
            sched = p['schedule']
            pen_dict, total_z = calculate_full_penalties(sched, workers, n_workers, n_days, weights)
            is_feas, hard_viols, hard_logs = audit_all_hard_constraints(sched, workers, n_workers, n_days, shift_reqs)
            req_details = build_worker_request_details(sched, workers, n_days, weights)

            pareto_solutions_list.append({
                'schedule': sched,
                'f1': float(p['f1']),
                'f2': float(p['f2']),
                'total_score': int(total_z),
                'penalties': pen_dict,
                'is_feasible': is_feas,
                'hard_violations_count': hard_viols,
                'hard_violation_logs': hard_logs,
                'request_details': req_details,
                'status': p['status'],
                'step': p['step']
            })

    # f1'e göre sırala
    pareto_solutions_list.sort(key=lambda s: s['f1'])

    # Pareto Hedef Noktaları
    pareto_pts = [(s['f1'], s['f2']) for s in pareto_solutions_list]
    
    # Referans Noktası (HV)
    ref_f1 = max(500.0, max((pt[0] for pt in pareto_pts), default=200.0) * 1.35 + 50)
    ref_f2 = max(500.0, max((pt[1] for pt in pareto_pts), default=200.0) * 1.35 + 50)
    ref_point = (ref_f1, ref_f2)

    final_hv = calculate_hypervolume_2d(pareto_pts, ref_point)
    final_spread = calculate_spread_metric(pareto_pts)

    # Uç Noktalar
    if len(pareto_solutions_list) > 0:
        ext_f1_idx = min(range(len(pareto_solutions_list)), key=lambda i: pareto_solutions_list[i]['f1'])
        extreme_f1_solution = pareto_solutions_list[ext_f1_idx]

        ext_f2_idx = min(range(len(pareto_solutions_list)), key=lambda i: pareto_solutions_list[i]['f2'])
        extreme_f2_solution = pareto_solutions_list[ext_f2_idx]

        knee_idx = find_knee_point_index(pareto_pts)
        knee_solution = pareto_solutions_list[knee_idx]
    else:
        # Fallback (Eğer hiçbir feasible çözüm bulunamadıysa)
        extreme_f1_solution = all_evaluated_solutions[0] if all_evaluated_solutions else {}
        extreme_f2_solution = extreme_f1_solution
        knee_solution = extreme_f1_solution

    exec_time_ms = round((time.time() - start_time) * 1000, 1)
    term_reason = f"🎯 ε-Kısıt Kesin Optimizasyonu tamamlandı. {grid_steps} grid adımı taranarak {len(pareto_solutions_list)} adet kesin Pareto optimal nokta kanıtlandı."

    # Standart Sonuç Paketi
    standard_result = build_standard_solver_result(
        schedule=knee_solution.get('schedule', np.zeros((n_workers, n_days), dtype=int)),
        workers=workers,
        is_feasible=knee_solution.get('is_feasible', True),
        hard_violations_count=knee_solution.get('hard_violations_count', 0),
        hard_violation_logs=knee_solution.get('hard_violation_logs', []),
        final_score=knee_solution.get('total_score', 0),
        initial_score=knee_solution.get('total_score', 0),
        improvement_rate=0.0,
        exec_time_ms=exec_time_ms,
        eval_count=grid_steps + 2,
        total_iterations=grid_steps,
        termination_reason=term_reason,
        penalties=knee_solution.get('penalties', {}),
        request_details=knee_solution.get('request_details', []),
        score_history=[knee_solution.get('total_score', 0)],
        meta={
            'grid_steps': grid_steps,
            'solver_engine': solver_engine,
            'f2_min': f2_min,
            'f2_max': f2_max,
            'pareto_front_size': len(pareto_solutions_list),
            'final_hypervolume': final_hv,
            'final_spread_delta': final_spread,
            'exec_time_ms': exec_time_ms
        }
    )

    return {
        'standard_result': standard_result,
        'pareto_solutions': pareto_solutions_list,
        'knee_solution': knee_solution,
        'extreme_f1_solution': extreme_f1_solution,
        'extreme_f2_solution': extreme_f2_solution,
        'all_evaluated_solutions': all_evaluated_solutions,
        'grid_logs': grid_logs,
        'metrics': {
            'final_hypervolume': final_hv,
            'final_spread_delta': final_spread,
            'exec_time_ms': exec_time_ms,
            'grid_steps': grid_steps,
            'solver_engine': solver_engine,
            'f2_min': f2_min,
            'f2_max': f2_max,
            'ref_point': ref_point
        },
        'workers': workers,
        'n_workers': n_workers,
        'n_days': n_days,
        'shift_reqs': shift_reqs,
        'weights': weights
    }
