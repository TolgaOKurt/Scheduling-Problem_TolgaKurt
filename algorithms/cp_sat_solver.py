"""
================================================================================
  ALGORITHMS/CP_SAT_SOLVER.PY - CONSTRAINT PROGRAMMING / CP-SAT (GOOGLE OR-TOOLS)
================================================================================
  Bu modül, Vardiya Çizelgeleme Problemi (NSP) için Google OR-Tools CP-SAT
  (Constraint Programming - Boolean Satisfiability) motorunu içerir.
  Lazy Clause Generation (LCG), Conflict-Driven Clause Learning (CDCL) ve
  çok çekirdekli paralel portföy aramasını (LNS) tam destekler.
================================================================================
"""

import time
import math
from typing import Dict, Any, List, Optional
import numpy as np
from ortools.sat.python import cp_model

from algorithms.worker_manager import generate_worker_profiles
from algorithms.penalty_calculator import (
    calculate_full_penalties,
    audit_all_hard_constraints,
    build_worker_request_details
)
from algorithms.greedy_solver import get_best_greedy_initial_solution
from algorithms.solver_contract import build_standard_solver_result


class CpSatSolutionTracker(cp_model.CpSolverSolutionCallback):
    """
    CP-SAT arama sürecinde bulunan ara çözümleri ve skor gelişimini
    canlı olarak yakalayan callback sınıfı.
    """
    def __init__(self, callback=None, stream_interval=1, scale_factor=1):
        super().__init__()
        self.callback = callback
        self.stream_interval = stream_interval
        self.scale_factor = max(1, scale_factor)
        self.solution_count = 0
        self.start_time = time.time()
        self.history = []

    def on_solution_callback(self):
        self.solution_count += 1
        elapsed = round(time.time() - self.start_time, 3)
        obj_val = round(self.ObjectiveValue() / self.scale_factor, 1)
        bound_val = round(self.BestObjectiveBound() / self.scale_factor, 1) if self.BestObjectiveBound() is not None else 0
        
        self.history.append({
            "time_sec": elapsed,
            "solution_idx": self.solution_count,
            "incumbent": obj_val,
            "best_bound": bound_val
        })
        
        if self.callback and (self.solution_count % max(1, self.stream_interval) == 0):
            msg = f"| Çözüm #{self.solution_count} | Kanıtlanan Alt Sınır: {bound_val}"
            try:
                self.callback(self.solution_count, obj_val, obj_val, msg)
                time.sleep(0.03)  # Streamlit WebSocket kuyruğunun DOM'a çizim yapabilmesi için mikro-bekleme
            except Exception:
                pass


def solve_cp_sat(
    n_workers: int,
    n_days: int,
    r_day: int,
    r_eve: int,
    r_night: int,
    weights: Dict[str, int],
    time_limit: float = 10.0,
    num_threads: int = 8,
    custom_workers: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Any] = None,
    stream_interval: int = 1
) -> Dict[str, Any]:
    """
    Google OR-Tools CP-SAT (Constraint Programming) Vardiya Optimizasyonu Çözücüsü.
    
    Parametreler:
    - time_limit (float): Maksimum süre sınırı (saniye)
    - num_threads (int): Paralel çalıştırılacak CPU arama iş parçacığı sayısı
    - callback: Canlı izleme fonksiyonu
    """
    start_time = time.time()
    
    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
    shifts = [0, 1, 2, 3] # 0: OFF, 1: Gündüz, 2: Akşam, 3: Gece
    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    ustas_wids = [w['id'] for w in workers if w.get('is_usta', False)]

    # Başlangıç Sezgisel Kök Maliyeti (Benchmark & Hinting)
    try:
        greedy_seed = get_best_greedy_initial_solution(n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers)
        initial_seed_cost = greedy_seed['final_score'] if (greedy_seed and greedy_seed['is_feasible']) else None
        seed_schedule = greedy_seed['schedule'] if (greedy_seed and greedy_seed['is_feasible']) else None
    except Exception:
        initial_seed_cost = None
        seed_schedule = None

    # 1. CP-SAT Model Nesnesi
    model = cp_model.CpModel()

    # =========================================================================
    # 1. KARAR DEĞİŞKENLERİ TANIMI (BOOLEAN DECISION VARIABLES)
    # =========================================================================
    # x[i, t, k] == 1 <=> i işçisi t gününde k vardiyasına atanırsa
    x = {}
    for i in range(n_workers):
        for t in range(n_days):
            for k in shifts:
                x[i, t, k] = model.NewBoolVar(f"x_{i}_{t}_{k}")

    # =========================================================================
    # 2. SERT KISITLAR (HARD CONSTRAINTS - %100 SAĞLANMASI ZORUNLU KANUNİ KISITLAR)
    # =========================================================================
    
    # H1: Günde Tam 1 Vardiya (Gündüz, Akşam, Gece veya OFF - ExactlyOne)
    for i in range(n_workers):
        for t in range(n_days):
            model.AddExactlyOne([x[i, t, k] for k in shifts])

    # H2: Vardiya Başı Kadro Kotaları (Tam Eşitlik)
    for t in range(n_days):
        for k in [1, 2, 3]:
            model.Add(sum(x[i, t, k] for i in range(n_workers)) == shift_reqs[k])

    # H3: Kritik 4 MYK Sertifikası Zorunluluğu (Vinç, Potacı, Döküm, Gaz İzleme)
    req_certs = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
    for cert in req_certs:
        cert_wids = [w['id'] for w in workers if cert in w.get('skills', set())]
        for t in range(n_days):
            for k in [1, 2, 3]:
                if len(cert_wids) > 0:
                    model.Add(sum(x[wid, t, k] for wid in cert_wids) >= 1)

    # H4: Vardiyalar Arası Yasal Dinlenme (Gece 08:00 Çıkış -> Sabah 08:00 Giriş YASAK - 0 Saat Dinlenme)
    for i in range(n_workers):
        for t in range(n_days - 1):
            model.Add(x[i, t, 3] + x[i, t + 1, 1] <= 1)

    # H5: Zorunlu Hafta Tatili (Kayan 7 Günlük Pencerede En Az 1 Gün OFF - Max 6 Gün Üst Üste Çalışma)
    for i in range(n_workers):
        for t in range(n_days - 6):
            model.Add(sum(x[i, tau, 0] for tau in range(t, t + 7)) >= 1)

    # =========================================================================
    # 3. YUMUŞAK KISITLAR & CEZA DOĞRUSALLAŞTIRMALARI (SOFT PENALTIES)
    # =========================================================================
    # CP-SAT tamsayı (integer) motoru olduğundan, kesirli ortalama sapmasını (35/28 = 1.25)
    # ILP / LP Relaxation ile birebir (%100) eşitlemek için tüm amaç fonksiyonu N ile ölçeklenir.
    N = max(1, n_workers)
    w_posta = weights.get('posta', weights.get('posta_unity', 15))
    w_circ = weights.get('circadian', 50)
    w_night = weights.get('night_imb', 25)
    w_workload = weights.get('workload_imb', weights.get('workload', 20))
    w_exp = weights.get('exp_mix', 60)
    w_pref = weights.get('pref_off', 40)

    penalty_terms = []

    # S1: Posta Takım Bütünlüğü (A, B, C, D takımlarının bölünmeme kuralı)
    u_posta_vars = {}
    for t in range(n_days):
        for p in postas:
            p_wids = [w['id'] for w in workers if w.get('posta') == p]
            N_p = len(p_wids)
            if N_p > 0:
                u_p_t = model.NewBoolVar(f"u_posta_{p}_{t}")
                u_posta_vars[p, t] = u_p_t
                p_active_sum = sum(x[wid, t, k] for wid in p_wids for k in [1, 2, 3])
                model.Add(p_active_sum == 0).OnlyEnforceIf(u_p_t.Not())
                model.Add(p_active_sum >= 1).OnlyEnforceIf(u_p_t)

            if N_p > 1:
                # z_posta[p, t, k]: p postasının t günündeki baskın vardiyası k mı?
                z_posta = {}
                for k in [1, 2, 3]:
                    z_posta[k] = model.NewBoolVar(f"z_posta_{p}_{t}_{k}")
                model.Add(sum(z_posta[k] for k in [1, 2, 3]) <= 1)
                
                for k in [1, 2, 3]:
                    y_count = sum(x[wid, t, k] for wid in p_wids)
                    dev_k = model.NewIntVar(0, N_p, f"dev_{p}_{t}_{k}")
                    # dev_k >= y_count - N_p * z_posta[k]
                    model.Add(dev_k >= y_count - N_p * z_posta[k])
                    penalty_terms.append(N * w_posta * dev_k)

        # Günlük 4-Posta Varlığı Kontrolü: 4 posta da aktifse +1 puan ceza (N * 1)
        valid_p_t = [u_posta_vars[p, t] for p in postas if (p, t) in u_posta_vars]
        if len(valid_p_t) >= 4:
            four_posta_var = model.NewBoolVar(f"four_posta_{t}")
            model.Add(sum(valid_p_t) == 4).OnlyEnforceIf(four_posta_var)
            model.Add(sum(valid_p_t) <= 3).OnlyEnforceIf(four_posta_var.Not())
            penalty_terms.append(N * 1 * four_posta_var)

    # S2: Sirkadiyen Ritim (Akşam 2 -> Ertesi Gün Gündüz 1 Ters Dönüş Cezası)
    for i in range(n_workers):
        for t in range(n_days - 1):
            sirk_var = model.NewBoolVar(f"sirk_{i}_{t}")
            # sirk_var == 1 <=> x[i, t, 2] == 1 AND x[i, t+1, 1] == 1
            model.AddBoolAnd([x[i, t, 2], x[i, t + 1, 1]]).OnlyEnforceIf(sirk_var)
            model.AddBoolOr([x[i, t, 2].Not(), x[i, t + 1, 1].Not()]).OnlyEnforceIf(sirk_var.Not())
            penalty_terms.append(N * w_circ * sirk_var)

    # S3: Gece Nöbeti Dengesizliği (Bireysel gece nöbeti ile kesirli ortalama sapması)
    # N * night_sum - tot_night_req == d_pos - d_neg
    tot_night_req = r_night * n_days
    for i in range(n_workers):
        night_sum = sum(x[i, t, 3] for t in range(n_days))
        d_pos = model.NewIntVar(0, N * n_days, f"dpos_{i}")
        d_neg = model.NewIntVar(0, N * n_days, f"dneg_{i}")
        model.Add(N * night_sum - tot_night_req == d_pos - d_neg)
        penalty_terms.append(w_night * (d_pos + d_neg))

    # S4: Toplam Çalışma / İş Yükü Dengesizliği (Bireysel toplam vardiya sayısı ile kesirli ortalama sapması)
    tot_work_req = (r_day + r_eve + r_night) * n_days
    for i in range(n_workers):
        work_sum = sum(x[i, t, k] for t in range(n_days) for k in [1, 2, 3])
        d_pos_work = model.NewIntVar(0, N * n_days, f"dpos_work_{i}")
        d_neg_work = model.NewIntVar(0, N * n_days, f"dneg_work_{i}")
        model.Add(N * work_sum - tot_work_req == d_pos_work - d_neg_work)
        penalty_terms.append(w_workload * (d_pos_work + d_neg_work))

    # S5: Kıdemli Usta Varlığı (Her vardiyada en az 1 usta)
    for t in range(n_days):
        for k in [1, 2, 3]:
            no_usta_var = model.NewBoolVar(f"no_usta_{t}_{k}")
            if len(ustas_wids) > 0:
                usta_sum = sum(x[wid, t, k] for wid in ustas_wids)
                model.Add(usta_sum == 0).OnlyEnforceIf(no_usta_var)
                model.Add(usta_sum >= 1).OnlyEnforceIf(no_usta_var.Not())
            else:
                model.Add(no_usta_var == 1)
            penalty_terms.append(N * w_exp * no_usta_var)

    # S6: Kişisel İzin Talepleri (pref_off gününde izinli olma)
    for i in range(n_workers):
        p_day_idx = int(workers[i].get('pref_off', 1)) - 1
        if 0 <= p_day_idx < n_days:
            pref_viol_var = model.NewBoolVar(f"pref_viol_{i}")
            # pref_viol_var == 1 <=> x[i, p_day_idx, 0] == 0 (çalışıyor)
            model.Add(x[i, p_day_idx, 0] == 0).OnlyEnforceIf(pref_viol_var)
            model.Add(x[i, p_day_idx, 0] == 1).OnlyEnforceIf(pref_viol_var.Not())
            penalty_terms.append(N * w_pref * pref_viol_var)

    # =========================================================================
    # 4. KÜRESEL AMAÇ FONKSİYONU VE SICAK BAŞLANGIÇ (WARM-START / HINTING)
    # =========================================================================
    model.Minimize(sum(penalty_terms))

    # Başlangıç Sezgisel Tohumunu Çözücüye İpucu Olarak Ver (Warm-Starting / Hinting)
    if seed_schedule is not None:
        for i in range(n_workers):
            for t in range(n_days):
                for k in shifts:
                    model.AddHint(x[i, t, k], 1 if seed_schedule[i, t] == k else 0)

    # =========================================================================
    # 5. ÇÖZÜCÜ YAPILANDIRMASI & PARALEL ÇALIŞTIRMA (PARALLEL LNS SEARCH)
    # =========================================================================
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max(0.5, time_limit))
    solver.parameters.num_search_workers = max(1, int(num_threads))
    solver.parameters.log_search_progress = False

    tracker = CpSatSolutionTracker(callback=callback, stream_interval=stream_interval, scale_factor=N)
    status = solver.Solve(model, tracker)
    exec_time_ms = round((time.time() - start_time) * 1000, 2)

    # =========================================================================
    # 7. ÇÖZÜM MATRİSİNİ ÇIKARMA VE DOĞRULAMA
    # =========================================================================
    schedule = np.zeros((n_workers, n_days), dtype=int)
    is_optimal = (status == cp_model.OPTIMAL)
    is_feasible = (status in (cp_model.OPTIMAL, cp_model.FEASIBLE))
    is_infeasible = (status == cp_model.INFEASIBLE)

    if is_feasible:
        for i in range(n_workers):
            for t in range(n_days):
                for k in shifts:
                    if solver.Value(x[i, t, k]) == 1:
                        schedule[i, t] = k
                        break

    # Sert Kısıt Bağımsız Denetimi
    is_hard_feas, hard_viols_count, hard_violation_logs = audit_all_hard_constraints(
        schedule, workers, n_workers, n_days, shift_reqs
    )

    if is_infeasible:
        is_hard_feas = False
        hard_viols_count = max(1, hard_viols_count)
        hard_violation_logs.append("❌ MATEMATİKSEL İMKANSIZLIK: Personel sayısı veya sertifikalar kısıtları sağlamak için yetersizdir (CP-SAT Infeasible).")

    if is_hard_feas:
        penalties_dict, final_score = calculate_full_penalties(schedule, workers, n_workers, n_days, weights)
    else:
        penalties_dict = {
            'Posta Takım Bütünlüğü İhlali': 0,
            'Sirkadiyen Ritim İhlali (Akşam->Gündüz)': 0,
            'Gece Nöbeti Dengesizliği': 0,
            'Toplam İş Yükü Dengesizliği': 0,
            'Kıdem / Usta Eksikliği': 0,
            'Kişisel İzin İhlali': 0
        }
        final_score = 99999

    if is_optimal:
        best_bound = float(final_score)
        mip_gap = 0.0
    elif is_feasible:
        raw_bound = round(solver.BestObjectiveBound() / N, 1)
        best_bound = min(float(final_score), raw_bound)
        mip_gap = round(abs((final_score - best_bound) / max(1, final_score)) * 100, 1)
    else:
        best_bound = 0.0
        mip_gap = 0.0

    if is_optimal:
        term_reason = f"🏆 OPTİMAL ÇÖZÜM BULUNDU (Google CP-SAT {solver.WallTime():.2f} sn'de matematiksel kanıtla tamamlandı, Gap: %0.0)."
    elif is_feasible:
        term_reason = f"⏱️ SÜRE SINIRINDA TAMAMLANDI (Google CP-SAT {time_limit} sn süre sınırında %100 geçerli çözüm üretti, Gap: %{mip_gap})."
    elif is_infeasible:
        term_reason = "❌ MATEMATİKSEL İMKANSIZLIK (CP-SAT arama uzayında hiçbir geçerli atama bulunamayacağını kanıtladı)."
    else:
        term_reason = f"⏱️ ZAMAN AŞIMI (Solver {time_limit} sn içinde geçerli bir çözüm üretemeden kesildi)."

    score_history = [pt["incumbent"] for pt in tracker.history] if tracker.history else [float(final_score)]
    request_details = build_worker_request_details(schedule, workers, n_days, weights)

    initial_score_val = score_history[0] if score_history else (initial_seed_cost if initial_seed_cost is not None else final_score)
    improvement_rate = 0.0
    if initial_score_val and initial_score_val > final_score and initial_score_val > 0:
        improvement_rate = round(((initial_score_val - final_score) / initial_score_val) * 100, 2)

    return build_standard_solver_result(
        schedule=schedule,
        workers=workers,
        is_feasible=bool(is_hard_feas and is_feasible),
        hard_violations_count=hard_viols_count,
        hard_violation_logs=hard_violation_logs,
        final_score=final_score,
        initial_score=initial_score_val,
        improvement_rate=improvement_rate,
        exec_time_ms=exec_time_ms,
        eval_count=max(1, tracker.solution_count),
        total_iterations=max(1, tracker.solution_count),
        termination_reason=term_reason,
        penalties=penalties_dict,
        request_details=request_details,
        score_history=score_history,
        meta={
            'solver_status': solver.StatusName(status),
            'wall_time': round(solver.WallTime(), 3),
            'best_bound': best_bound,
            'mip_gap': mip_gap,
            'is_optimal': is_optimal,
            'is_infeasible': is_infeasible,
            'greedy_seed_score': initial_seed_cost,
            'branches': solver.NumBranches(),
            'conflicts': solver.NumConflicts(),
            'num_threads': num_threads,
            'solutions_found': tracker.solution_count
        }
    )
