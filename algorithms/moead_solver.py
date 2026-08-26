"""
================================================================================
  ALGORITHMS/MOEAD_SOLVER.PY - MOEA/D (AYRIŞTIRMA TABANLI ÇOK AMAÇLI OPTİMİZASYON)
================================================================================
  Bu modül; Qingfu Zhang & Hui Li (IEEE TEVC 2007) tarafından geliştirilen
  "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"
  algoritmasını Vardiya Çizelgeleme Problemi (NSP/NRP) için uygular.

  Temel Mimarisi:
  1. Ağırlık Vektörleri ile Ayrıştırma (Decomposition into N Subproblems)
  2. Öklid Komşuluk Matrisi ve Bilgi Paylaşımı (Neighborhood B(i))
  3. Skalerleştirme Fonksiyonları: Tchebycheff, Weighted Sum, PBI
  4. İdeal Referans Noktası (z*) Anlık Güncelleme
  5. Komşuluk İçi Çaprazlama, Mutasyon ve Yerel Güncelleme
  6. Dış Pareto Arşivi (External Population - EP)
  7. Pareto Metrikleri: Hiper-Hacim (HV), Spread (Δ) ve Diz Noktası (Knee Point)
================================================================================
"""

import time
import math
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import get_best_greedy_initial_solution
from algorithms.penalty_calculator import (
    calculate_full_penalties,
    audit_all_hard_constraints,
    build_worker_request_details,
    check_swap_feasibility
)
from algorithms.evolutionary_engine import initialize_population, crossover_daywise, mutate_swap
from algorithms.nsga2_solver import (
    build_fast_multi_objective_evaluator,
    calculate_hypervolume_2d,
    calculate_spread_metric,
    find_knee_point_index
)
from algorithms.solver_contract import build_standard_solver_result


# =============================================================================
# 1. AĞIRLIK VEKTÖRLERİ VE KOMŞULUK YAPISI
# =============================================================================
def generate_weight_vectors_2d(n_subproblems: int) -> np.ndarray:
    """
    2 boyutlu amaç uzayını homojen bölen N adet ağırlık vektörü üretir.
    lambda^i = (i / (N - 1), 1 - i / (N - 1))
    """
    weights = np.zeros((n_subproblems, 2), dtype=float)
    for i in range(n_subproblems):
        w1 = i / max(1, n_subproblems - 1)
        w2 = 1.0 - w1
        weights[i, 0] = w1
        weights[i, 1] = w2
    return weights


def compute_neighborhoods(weight_vectors: np.ndarray, neighborhood_size: int) -> np.ndarray:
    """
    Her ağırlık vektörü için Öklid mesafesine göre en yakın T adet komşunun indislerini hesaplar.
    Döndürdüğü: (N, T) boyutunda komşuluk matrisi B
    """
    n = len(weight_vectors)
    t_size = min(n, max(2, neighborhood_size))
    neighborhoods = np.zeros((n, t_size), dtype=int)

    for i in range(n):
        dists = np.linalg.norm(weight_vectors - weight_vectors[i], axis=1)
        # En yakın T komşunun indislerini sırala
        neighborhoods[i] = np.argsort(dists)[:t_size]

    return neighborhoods


# =============================================================================
# 2. SKALERLEŞTİRME / AYRIŞTIRMA FONKSİYONLARI (SCALARIZING FUNCTIONS)
# =============================================================================
def evaluate_tchebycheff(f_vals: Tuple[float, float], weight: np.ndarray, ideal_point: np.ndarray) -> float:
    """
    Tchebycheff Yaklaşımı:
    g^te(x | lambda, z*) = max { lambda_1 * |f1(x) - z1*|, lambda_2 * |f2(x) - z2*| }
    """
    diff = np.abs(np.array(f_vals, dtype=float) - ideal_point)
    # 0 ağırlık durumunda bölme/çarpma kararlılığı için küçük epsilon
    w_eff = np.maximum(weight, 1e-4)
    return float(np.max(w_eff * diff))


def evaluate_weighted_sum(f_vals: Tuple[float, float], weight: np.ndarray) -> float:
    """
    Ağırlıklı Toplam Yaklaşımı:
    g^ws(x | lambda) = lambda_1 * f1(x) + lambda_2 * f2(x)
    """
    return float(weight[0] * f_vals[0] + weight[1] * f_vals[1])


def evaluate_pbi(f_vals: Tuple[float, float], weight: np.ndarray, ideal_point: np.ndarray, theta: float = 5.0) -> float:
    """
    Penalty-based Boundary Intersection (PBI):
    g^pbi(x | lambda, z*) = d1 + theta * d2
    """
    f_arr = np.array(f_vals, dtype=float)
    w_norm = np.linalg.norm(weight)
    if w_norm == 0:
        w_unit = np.array([0.5, 0.5])
    else:
        w_unit = weight / w_norm

    diff = f_arr - ideal_point
    d1 = np.dot(diff, w_unit)
    d2 = np.linalg.norm(diff - d1 * w_unit)
    return float(d1 + theta * d2)


# =============================================================================
# 3. DIŞ PARETO ARŞİVİ (EXTERNAL POPULATION - EP) YÖNETİMİ
# =============================================================================
class ExternalParetoArchive:
    """
    MOEA/D evrimi sırasında keşfedilen tüm baskın olmayan (non-dominated)
    geçerli çizelge çözümlerini saklayan dinamik dış arşiv.
    """
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.archive: List[Dict[str, Any]] = []

    def update(self, individual: np.ndarray, f_vals: Tuple[float, float]):
        f1_new, f2_new = f_vals

        # 1. Kontrol: Arşivdeki herhangi bir birey yeni çözümü domine ediyor mu?
        for item in self.archive:
            f1_arch, f2_arch = item['f_vals']
            # Dominance kontrolü: item dominates new
            if (f1_arch <= f1_new and f2_arch <= f2_new) and (f1_arch < f1_new or f2_arch < f2_new):
                return  # Yeni çözüm domine edildi, eklenmez

        # 2. Yeni çözüm tarafından domine edilen arşiv üyelerini sil
        survived = []
        for item in self.archive:
            f1_arch, f2_arch = item['f_vals']
            # New dominates item
            if (f1_new <= f1_arch and f2_new <= f2_arch) and (f1_new < f1_arch or f2_new < f2_arch):
                continue  # Silinir
            survived.append(item)

        # 3. Tekil koordinat kontrolü (Aynı f1, f2 değerine sahip çiftleri engelle)
        for item in survived:
            if abs(item['f_vals'][0] - f1_new) < 1e-3 and abs(item['f_vals'][1] - f2_new) < 1e-3:
                self.archive = survived
                return

        # 4. Yeni çözümü ekle
        survived.append({
            'schedule': individual.copy(),
            'f_vals': (float(f1_new), float(f2_new))
        })

        # Arşiv boyut sınırlaması
        if len(survived) > self.max_size:
            # f1'e göre sırala ve kalabalıklaşmaya göre hafifçe filtrele
            survived.sort(key=lambda it: it['f_vals'][0])
            step = len(survived) / float(self.max_size)
            filtered = [survived[int(i * step)] for i in range(self.max_size)]
            self.archive = filtered
        else:
            self.archive = survived

    def get_pareto_points(self) -> List[Tuple[float, float]]:
        return [it['f_vals'] for it in self.archive]


# =============================================================================
# 4. ANA MOEA/D EVRİMSEL ÇÖZÜCÜ FONKSİYONU
# =============================================================================
def solve_moead(
    n_workers: int,
    n_days: int,
    r_day: int,
    r_eve: int,
    r_night: int,
    weights: Dict[str, int],
    pop_size: int = 30,             # N: Alt problem / popülasyon sayısı
    max_generations: int = 50,       # G: Maksimum jenerasyon sayısı
    neighborhood_size: int = 5,      # T: Komşuluk boyutu
    decomposition_method: str = "Tchebycheff (Önerilen)",
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.08,
    seed: Optional[int] = None,
    custom_workers: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Any] = None,
    stream_interval: int = 1
) -> Dict[str, Any]:
    """
    MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) Çözücüsü.
    
    Adımlar:
    1. N adet ağırlık vektörü lambda^1 ... lambda^N üretilir.
    2. Öklid komşuluk matrisi B(i) hesaplanır.
    3. Başlangıç popülasyonu üretilir ve z* (ideal nokta) ilklendirilir.
    4. G jenerasyon boyunca her alt problem komşuluk içi çaprazlama ve mutasyonla ürer.
    5. Alt problemler ve Dış Pareto Arşivi (EP) güncellenir.
    6. Pareto analitiği, Hiper-Hacim (HV), Spread (Δ) ve Diz Noktası (Knee Point) çıkarılır.
    """
    start_time = time.time()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # 1. Hızlı İki-Amaçlı Değerlendirici Closure
    fast_eval = build_fast_multi_objective_evaluator(workers, n_days, weights)

    # 2. Ağırlık Vektörleri ve Komşuluk Yapısı
    weight_vectors = generate_weight_vectors_2d(pop_size)
    neighborhoods = compute_neighborhoods(weight_vectors, neighborhood_size)

    # Skalerleştirme fonksiyonu seçimi
    is_tchebycheff = "Tchebycheff" in decomposition_method
    is_pbi = "PBI" in decomposition_method

    def scalar_fitness(f_vals: Tuple[float, float], weight_idx: int, z_star: np.ndarray) -> float:
        w = weight_vectors[weight_idx]
        if is_tchebycheff:
            return evaluate_tchebycheff(f_vals, w, z_star)
        elif is_pbi:
            return evaluate_pbi(f_vals, w, z_star)
        else:
            return evaluate_weighted_sum(f_vals, w)

    # 3. Başlangıç Popülasyonunun Üretilmesi
    greedy_seed = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers
    )
    base_sched = greedy_seed['schedule'] if (greedy_seed and greedy_seed['is_feasible']) else np.zeros((n_workers, n_days), dtype=int)

    population = initialize_population(base_sched, pop_size, n_workers, n_days, workers, shift_reqs)
    obj_values = [fast_eval(ind) for ind in population]

    # İdeal Referans Noktası (z* = min f1, min f2)
    z_star = np.array([
        min(ov[0] for ov in obj_values),
        min(ov[1] for ov in obj_values)
    ], dtype=float)

    # Dış Pareto Arşivi (EP)
    ep_archive = ExternalParetoArchive(max_size=100)
    for idx, ind in enumerate(population):
        ep_archive.update(ind, obj_values[idx])

    # Referans Noktası (HV Hesabı İçin)
    ref_f1 = max(500.0, max(ov[0] for ov in obj_values) * 1.35 + 50)
    ref_f2 = max(500.0, max(ov[1] for ov in obj_values) * 1.35 + 50)
    ref_point = (ref_f1, ref_f2)

    # Metrik Takip Geçmişi
    hv_history: List[float] = []
    ep_size_history: List[int] = []
    min_f1_history: List[float] = []
    min_f2_history: List[float] = []
    avg_f1_history: List[float] = []
    avg_f2_history: List[float] = []
    best_total_history: List[int] = []

    # =========================================================================
    # 5. ANA EVRİMSEL DÖNGÜ (MAIN MOEA/D GENERATION LOOP)
    # =========================================================================
    for gen in range(1, max_generations + 1):
        for i in range(pop_size):
            # 1. Ebeveyn Seçimi: Komşuluk B(i) içinden rastgele 2 farklı ebeveyn seç
            neighbor_indices = neighborhoods[i]
            p1_idx, p2_idx = random.sample(list(neighbor_indices), 2)
            parent1 = population[p1_idx]
            parent2 = population[p2_idx]

            # 2. Çaprazlama (Crossover)
            child = crossover_daywise(parent1, parent2, n_days, workers, n_workers, shift_reqs, crossover_rate)

            # 3. Sert Kısıt Korumalı Takas Mutasyonu
            child = mutate_swap(child, n_workers, n_days, workers, shift_reqs, mutation_rate)

            # 4. Değerlendirme
            f_child = fast_eval(child)

            # 5. İdeal Referans Noktası z* Güncelleme
            z_star[0] = min(z_star[0], f_child[0])
            z_star[1] = min(z_star[1], f_child[1])

            # 6. Komşuluk İçi Güncelleme (Neighborhood Update)
            # Çocuk, komşu alt problemlerin mevcut çözümünden daha iyi skaler değere sahipse yerini alır
            for j in neighbor_indices:
                curr_scalar = scalar_fitness(obj_values[j], j, z_star)
                child_scalar = scalar_fitness(f_child, j, z_star)

                if child_scalar < curr_scalar:
                    population[j] = child.copy()
                    obj_values[j] = f_child

            # 7. Dış Pareto Arşivini Güncelle
            ep_archive.update(child, f_child)

        # İstatistikler
        cur_f1s = [ov[0] for ov in obj_values]
        cur_f2s = [ov[1] for ov in obj_values]
        cur_totals = [int(ov[0] + ov[1]) for ov in obj_values]

        cur_ep_points = ep_archive.get_pareto_points()
        cur_hv = calculate_hypervolume_2d(cur_ep_points, ref_point)
        cur_best_total = min(cur_totals)

        hv_history.append(cur_hv)
        ep_size_history.append(len(cur_ep_points))
        min_f1_history.append(float(min(cur_f1s)))
        min_f2_history.append(float(min(cur_f2s)))
        avg_f1_history.append(float(np.mean(cur_f1s)))
        avg_f2_history.append(float(np.mean(cur_f2s)))
        best_total_history.append(cur_best_total)

        if callback and (gen % max(1, stream_interval) == 0 or gen == max_generations):
            msg = f"| Nesil {gen}/{max_generations} | Arşiv: {len(cur_ep_points)} | HV: {cur_hv:,.0f} | Min Total Z: {cur_best_total}"
            callback(gen, cur_best_total, cur_best_total, msg)

    # =========================================================================
    # 6. NİHAİ PARETO KÜMESİNİN DERLENMESİ VE KARAR DESTEK ANALİZİ
    # =========================================================================
    final_archive = ep_archive.archive

    # Eğer arşiv boşsa popülasyondan türet
    if not final_archive:
        for idx, ind in enumerate(population):
            final_archive.append({'schedule': ind, 'f_vals': obj_values[idx]})

    # Tüm arşiv çözümlerini detaylı denetimden geçir
    pareto_solutions_list: List[Dict[str, Any]] = []
    for item in final_archive:
        sched = item['schedule']
        f1_val, f2_val = item['f_vals']

        pen_dict, total_z = calculate_full_penalties(sched, workers, n_workers, n_days, weights)
        is_feas, hard_viols, hard_logs = audit_all_hard_constraints(sched, workers, n_workers, n_days, shift_reqs)
        req_details = build_worker_request_details(sched, workers, n_days, weights)

        pareto_solutions_list.append({
            'schedule': sched,
            'f1': float(f1_val),
            'f2': float(f2_val),
            'total_score': int(total_z),
            'penalties': pen_dict,
            'is_feasible': is_feas,
            'hard_violations_count': hard_viols,
            'hard_violation_logs': hard_logs,
            'request_details': req_details
        })

    # f1'e göre sırala
    pareto_solutions_list.sort(key=lambda s: s['f1'])

    final_pareto_pts = [(s['f1'], s['f2']) for s in pareto_solutions_list]
    final_hv = calculate_hypervolume_2d(final_pareto_pts, ref_point)
    final_spread = calculate_spread_metric(final_pareto_pts)

    # Uç Noktalar ve Diz Noktası (Knee Point / Altın Denge)
    ext_f1_idx = min(range(len(pareto_solutions_list)), key=lambda i: pareto_solutions_list[i]['f1'])
    extreme_f1_solution = pareto_solutions_list[ext_f1_idx]

    ext_f2_idx = min(range(len(pareto_solutions_list)), key=lambda i: pareto_solutions_list[i]['f2'])
    extreme_f2_solution = pareto_solutions_list[ext_f2_idx]

    knee_idx = find_knee_point_index(final_pareto_pts)
    knee_solution = pareto_solutions_list[knee_idx]

    exec_time_ms = round((time.time() - start_time) * 1000, 1)
    eval_count = pop_size + (pop_size * max_generations)
    term_reason = f"🌐 MOEA/D Optimizasyonu başarıyla tamamlandı. {max_generations} jenerasyon sonunda {len(pareto_solutions_list)} adet baskın olmayan Pareto optimal çözüm arşive alındı."

    # Standart Sonuç Paketi
    standard_result = build_standard_solver_result(
        schedule=knee_solution['schedule'],
        workers=workers,
        is_feasible=knee_solution['is_feasible'],
        hard_violations_count=knee_solution['hard_violations_count'],
        hard_violation_logs=knee_solution['hard_violation_logs'],
        final_score=knee_solution['total_score'],
        initial_score=best_total_history[0] if best_total_history else knee_solution['total_score'],
        improvement_rate=round(((best_total_history[0] - knee_solution['total_score']) / max(1, best_total_history[0])) * 100, 2) if best_total_history else 0.0,
        exec_time_ms=exec_time_ms,
        eval_count=eval_count,
        total_iterations=max_generations,
        termination_reason=term_reason,
        penalties=knee_solution['penalties'],
        request_details=knee_solution['request_details'],
        score_history=best_total_history,
        meta={
            'pop_size': pop_size,
            'max_generations': max_generations,
            'neighborhood_size': neighborhood_size,
            'decomposition_method': decomposition_method,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
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
        'population_solutions': population,
        'population_objectives': obj_values,
        'weight_vectors': weight_vectors,
        'metrics': {
            'hv_history': hv_history,
            'ep_size_history': ep_size_history,
            'min_f1_history': min_f1_history,
            'min_f2_history': min_f2_history,
            'avg_f1_history': avg_f1_history,
            'avg_f2_history': avg_f2_history,
            'best_total_history': best_total_history,
            'final_hypervolume': final_hv,
            'final_spread_delta': final_spread,
            'exec_time_ms': exec_time_ms,
            'eval_count': eval_count,
            'ref_point': ref_point
        },
        'workers': workers,
        'n_workers': n_workers,
        'n_days': n_days,
        'shift_reqs': shift_reqs,
        'weights': weights
    }
