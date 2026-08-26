"""
================================================================================
  ALGORITHMS/NSGA2_SOLVER.PY - NSGA-II ÇOK AMAÇLI EVRİMSEL OPTİMİZASYON ÇÖZÜCÜSÜ
================================================================================
  Bu modül; Kalyanmoy Deb et al. (2002) tarafından geliştirilen ve literatürün
  altın standardı olan "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
  algoritmasını Vardiya Çizelgeleme Problemi (NSP/NRP) için eksiksiz uygular.

  Temel Mimarisi:
  1. Hızlı Baskın Olmayan Sıralama (Fast Non-dominated Sorting - O(M*N^2))
  2. Kalabalıklaşma Mesafesi Ataması (Crowding Distance Assignment)
  3. Kalabalık Karşılaştırma Turnuva Seçimi (Crowded Comparison Operator ≺n)
  4. Sert Kısıt Korumalı Gün Bazlı Çaprazlama & Takas Mutasyonu
  5. (N + N) -> N Elitist Popülasyon Seçimi
  6. Pareto Analitiği: Hypervolume (HV), Deb's Spread (Δ) ve Diz Noktası (Knee Point)
================================================================================
"""

import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import get_best_greedy_initial_solution
from algorithms.penalty_calculator import (
    calculate_full_penalties,
    check_swap_feasibility,
    build_worker_request_details,
    audit_all_hard_constraints
)
from algorithms.evolutionary_engine import initialize_population, crossover_daywise, mutate_swap
from algorithms.solver_contract import build_standard_solver_result


# =============================================================================
# 1. HIZLI ÇOK AMAÇLI CEZA HESAPLAYICI (FAST MULTI-OBJECTIVE EVALUATOR)
# =============================================================================
def build_fast_multi_objective_evaluator(workers, n_days, weights):
    """
    NSGA-II iç döngülerinde binlerce kez çağrılan yüksek hızlı, vektörize
    2-amaçlı [f1(X), f2(X)] değerlendirici closure fonksiyonunu üretir.

    f1(X) = İşletme & Üretim Verimliliği Cezası (Posta Bütünlüğü + Usta Varlığı + 4-Posta)
    f2(X) = Çalışan Memnuniyeti Cezası (Kişisel İzin + Gece Nöbeti + İş Yükü + Sirkadiyen)
    """
    if weights is None:
        weights = {}

    w_posta = weights.get('posta', weights.get('posta_unity', 15))
    w_circ = weights.get('circadian', 50)
    w_night_imb = weights.get('night_imb', 25)
    w_workload_imb = weights.get('workload_imb', weights.get('workload', 20))
    w_exp = weights.get('exp_mix', 60)
    w_pref = weights.get('pref_off', 40)

    is_usta_arr = np.array([w.get('is_usta', False) for w in workers])
    pref_days = np.array([int(w.get('pref_off', 1)) - 1 for w in workers])
    valid_pref = (pref_days >= 0) & (pref_days < n_days)
    pref_rows = np.arange(len(pref_days))[valid_pref]
    pref_cols = pref_days[valid_pref]

    postas = np.array([w.get('posta', '') for w in workers])
    posta_groups = [np.where(postas == p)[0] for p in ['Posta A', 'Posta B', 'Posta C', 'Posta D']]
    valid_groups = [g for g in posta_groups if len(g) > 0]

    def fast_evaluate_objectives(sched: np.ndarray) -> Tuple[float, float]:
        # --- f1: İşletme & Üretim Verimliliği ---
        # 1. Posta Bütünlüğü
        p_posta = 0
        for g in posta_groups:
            if len(g) > 1:
                for t in range(n_days):
                    col_g = sched[g, t]
                    act = col_g[col_g > 0]
                    if len(act) > 1:
                        bc = np.bincount(act)
                        p_posta += (len(act) - bc.max()) * w_posta

        # Günlük 4-Posta Varlığı Kontrolü
        if len(valid_groups) >= 4:
            active_post_per_day = sum((sched[g, :] > 0).any(axis=0) for g in valid_groups)
            p_posta += int((active_post_per_day >= 4).sum())

        # 2. Usta Varlığı
        p_exp = 0
        for t in range(n_days):
            col = sched[:, t]
            for k in (1, 2, 3):
                mask = (col == k)
                if mask.any() and not is_usta_arr[mask].any():
                    p_exp += w_exp

        f1 = float(p_posta + p_exp)

        # --- f2: Çalışan Memnuniyeti, Adalet & Ergonomi ---
        # 3. Sirkadiyen Ritim
        p_circ = int(((sched[:, :-1] == 2) & (sched[:, 1:] == 1)).sum() * w_circ) if n_days > 1 else 0

        # 4. Gece Nöbeti Dengesizliği
        night_counts = (sched == 3).sum(axis=1)
        avg_night = night_counts.mean()
        p_night = int(np.abs(night_counts - avg_night).sum() * w_night_imb)

        # 5. Toplam İş Yükü Dengesizliği
        work_counts = (sched > 0).sum(axis=1)
        avg_work = work_counts.mean()
        p_workload = int(np.abs(work_counts - avg_work).sum() * w_workload_imb)

        # 6. Kişisel İzin
        p_pref = int((sched[pref_rows, pref_cols] != 0).sum() * w_pref) if len(pref_rows) > 0 else 0

        f2 = float(p_circ + p_night + p_workload + p_pref)

        return f1, f2

    return fast_evaluate_objectives


# =============================================================================
# 2. HIZLI BASKIN OLMAYAN SIRALAMA (FAST NON-DOMINATED SORTING - Deb et al., 2002)
# =============================================================================
def fast_non_dominated_sort(objectives: List[Tuple[float, float]]) -> Tuple[List[List[int]], np.ndarray]:
    """
    Deb et al. (2002) O(M * N^2) Hızlı Baskınlık Sıralaması algoritması.
    Popülasyonu hiyerarşik Pareto katmanlarına (Front 1, Front 2, ...) ayırır.

    Girdi:
      - objectives: [(f1_0, f2_0), (f1_1, f2_1), ...]

    Çıktı:
      - fronts: [[front1_indices], [front2_indices], ...]
      - ranks: Her bireyin Pareto rütbesi dizisi (1, 2, ...)
    """
    n = len(objectives)
    domination_counts = np.zeros(n, dtype=int)  # n_p: p'ye baskın gelen birey sayısı
    dominated_sets = [[] for _ in range(n)]      # S_p: p'nin baskın geldiği bireyler
    ranks = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        p_f1, p_f2 = objectives[p]
        for q in range(p + 1, n):
            q_f1, q_f2 = objectives[q]

            # p dominates q
            if (p_f1 <= q_f1 and p_f2 <= q_f2) and (p_f1 < q_f1 or p_f2 < q_f2):
                dominated_sets[p].append(q)
                domination_counts[q] += 1
            # q dominates p
            elif (q_f1 <= p_f1 and q_f2 <= p_f2) and (q_f1 < p_f1 or q_f2 < p_f2):
                dominated_sets[q].append(p)
                domination_counts[p] += 1

        if domination_counts[p] == 0:
            ranks[p] = 1
            fronts[0].append(p)

    curr_front_idx = 0
    while len(fronts[curr_front_idx]) > 0:
        next_front: List[int] = []
        for p in fronts[curr_front_idx]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    ranks[q] = curr_front_idx + 2
                    next_front.append(q)

        curr_front_idx += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts, ranks


# =============================================================================
# 3. KALABALIKLAŞMA MESAFESİ (CROWDING DISTANCE ASSIGNMENT - Deb et al., 2002)
# =============================================================================
def calculate_crowding_distance(front_indices: List[int], objectives: List[Tuple[float, float]]) -> Dict[int, float]:
    """
    Belirli bir Pareto katmanındaki (Front) bireylerin amaç uzayındaki
    yoğunluk mesafelerini hesaplar. Uç noktalara sonsuz (1e9) mesafe verilir.
    """
    l = len(front_indices)
    distances = {idx: 0.0 for idx in front_indices}
    if l == 0:
        return distances
    if l <= 2:
        for idx in front_indices:
            distances[idx] = 1e9
        return distances

    num_objectives = 2  # f1 ve f2
    for m in range(num_objectives):
        # Front'u m. amaca göre artan sırada diz
        sorted_front = sorted(front_indices, key=lambda idx: objectives[idx][m])
        
        # Uç noktalara sonsuz mesafe
        distances[sorted_front[0]] = 1e9
        distances[sorted_front[-1]] = 1e9

        f_min = objectives[sorted_front[0]][m]
        f_max = objectives[sorted_front[-1]][m]
        norm = f_max - f_min

        if norm > 1e-9:
            for k in range(1, l - 1):
                prev_val = objectives[sorted_front[k - 1]][m]
                next_val = objectives[sorted_front[k + 1]][m]
                distances[sorted_front[k]] += (next_val - prev_val) / norm

    return distances


# =============================================================================
# 4. KALABALIK KARŞILAŞTIRMA OPERATÖRÜ VE SEÇİM (CROWDED COMPARISON & TOURNAMENT)
# =============================================================================
def crowded_comparison_operator(idx_a: int, idx_b: int, ranks: np.ndarray, distances: Dict[int, float]) -> int:
    """
    Deb et al. (2002) Crowded-Comparison Operator (≺n):
    1. Daha düşük (iyi) Pareto rütbesine sahip birey kazanır: rank_a < rank_b
    2. Aynı rütbede iseler, daha az kalabalık bölgedeki (yüksek mesafe) birey kazanır: dist_a > dist_b
    """
    rank_a = ranks[idx_a]
    rank_b = ranks[idx_b]

    if rank_a < rank_b:
        return idx_a
    elif rank_b < rank_a:
        return idx_b
    else:
        dist_a = distances.get(idx_a, 0.0)
        dist_b = distances.get(idx_b, 0.0)
        if dist_a >= dist_b:
            return idx_a
        else:
            return idx_b


def binary_tournament_selection(population: List[np.ndarray], ranks: np.ndarray, distances: Dict[int, float]) -> np.ndarray:
    """Popülasyondan rastgele 2 aday seçip kalabalık karşılaştırma ile galibi döndürür."""
    n = len(population)
    i1, i2 = random.sample(range(n), 2)
    winner_idx = crowded_comparison_operator(i1, i2, ranks, distances)
    return population[winner_idx].copy()


# =============================================================================
# 5. PARETO ANALİTİK METRİKLERİ (HYPERVOLUME, SPREAD & KNEE POINT)
# =============================================================================
def calculate_hypervolume_2d(pareto_objectives: List[Tuple[float, float]], ref_point: Tuple[float, float]) -> float:
    """
    2-Boyutlu amaç uzayında Pareto Ön Cephesinin kapsadığı kesin Hiper-Hacmi (Hypervolume)
    merdiven integrasyonu yöntemiyle analitik olarak hesaplar.
    """
    if len(pareto_objectives) == 0:
        return 0.0

    # f1'e göre artan sırada diz (f1 artarken f2 monoton azalmalıdır)
    sorted_pts = sorted(pareto_objectives, key=lambda pt: (pt[0], pt[1]))
    
    # Filtrele: Referans noktasından daha kötü olan noktaları dışarıda tut
    valid_pts = [pt for pt in sorted_pts if pt[0] <= ref_point[0] and pt[1] <= ref_point[1]]
    if not valid_pts:
        return 0.0

    # 2D Staircase Entegrasyonu
    hv = 0.0
    curr_f2_bound = ref_point[1]
    for i, (f1, f2) in enumerate(valid_pts):
        if f2 < curr_f2_bound:
            width = ref_point[0] - f1
            height = curr_f2_bound - f2
            hv += width * height
            curr_f2_bound = f2

    return float(max(0.0, hv))


def calculate_spread_metric(pareto_objectives: List[Tuple[float, float]]) -> float:
    """
    Deb et al. (2002) Yayılım / Dağılım Metriği (Spread Metric Δ):
    Çözümlerin Pareto cephesine ne kadar homojen yayıldığını ölçer (0 = Kusursuz Eşit Dağılım).
    """
    if len(pareto_objectives) < 2:
        return 0.0

    sorted_pts = sorted(pareto_objectives, key=lambda pt: pt[0])
    consecutive_dists = []
    for i in range(len(sorted_pts) - 1):
        p1 = sorted_pts[i]
        p2 = sorted_pts[i + 1]
        dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        consecutive_dists.append(dist)

    d_mean = float(np.mean(consecutive_dists))
    if d_mean < 1e-9:
        return 0.0

    d_f = consecutive_dists[0]
    d_l = consecutive_dists[-1]
    sum_diffs = sum(abs(d - d_mean) for d in consecutive_dists)

    delta = (d_f + d_l + sum_diffs) / (d_f + d_l + (len(consecutive_dists)) * d_mean)
    return float(round(delta, 3))


def find_knee_point_index(pareto_objectives: List[Tuple[float, float]]) -> int:
    """
    Pareto Ön Cephesi üzerindeki Diz Noktasını (Knee Point / Altın Denge Noktası) tespit eder.
    Normalleştirilmiş amaç uzayında ideal [0, 0] noktasına en yakın olan çözümü seçer.
    """
    n = len(pareto_objectives)
    if n == 0:
        return 0
    if n <= 2:
        return 0

    f1_vals = [pt[0] for pt in pareto_objectives]
    f2_vals = [pt[1] for pt in pareto_objectives]

    min_f1, max_f1 = min(f1_vals), max(f1_vals)
    min_f2, max_f2 = min(f2_vals), max(f2_vals)

    range_f1 = max_f1 - min_f1 if (max_f1 - min_f1) > 1e-9 else 1.0
    range_f2 = max_f2 - min_f2 if (max_f2 - min_f2) > 1e-9 else 1.0

    # Normalleştirilmiş ideal nokta mesafesi
    best_dist = float('inf')
    knee_idx = 0
    for idx, (f1, f2) in enumerate(pareto_objectives):
        norm_f1 = (f1 - min_f1) / range_f1
        norm_f2 = (f2 - min_f2) / range_f2
        dist = np.sqrt(norm_f1**2 + norm_f2**2)
        if dist < best_dist:
            best_dist = dist
            knee_idx = idx

    return knee_idx


# =============================================================================
# 6. ANA NSGA-II ÇOK AMAÇLI OPTİMİZASYON FONKSİYONU
# =============================================================================
def run_nsga2_optimization(
    n_workers: int,
    n_days: int,
    r_day: int,
    r_eve: int,
    r_night: int,
    weights: Dict[str, int],
    pop_size: int = 60,
    generations: int = 100,
    crossover_rate: float = 0.85,
    mutation_rate: float = 0.08,
    seed: int = 42,
    custom_workers: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Any] = None,
    stream_interval: int = 2,
    time_limit: Optional[float] = None
) -> Dict[str, Any]:
    """
    NSGA-II Çok Amaçlı Evrimsel Vardiya Optimizasyonu Motoru.

    Döndürdüğü:
      - pareto_solutions (list[dict]): Rank 1 Pareto cephesindeki tüm çözümler
      - knee_solution (dict): Altın denge noktası çözümü
      - extreme_f1_solution (dict): İşletme odaklı uç çözüm
      - extreme_f2_solution (dict): Çalışan odaklı uç çözüm
      - pareto_metrics (dict): HV, Spread, Nesil geçmişleri vb.
      - solver_result (dict): Standard solver contract ile %100 uyumlu ana çözüm
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # 1. Hızlı Çok Amaçlı Değerlendiriciyi Hazırla
    fast_eval = build_fast_multi_objective_evaluator(workers, n_days, weights)

    # 2. Başlangıç Popülasyonu (Warm-Start Greedy Seed + Feasible Swaps)
    greedy_seed = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=custom_workers
    )
    greedy_sched = greedy_seed['schedule']
    workers = greedy_seed['workers']

    population: List[np.ndarray] = initialize_population(
        greedy_sched, pop_size, n_workers, n_days, workers, shift_reqs
    )

    # 3. Başlangıç Popülasyonunun Amaç Değerlerini Hesapla
    eval_count = 0
    pop_objectives: List[Tuple[float, float]] = []
    for chrom in population:
        f1, f2 = fast_eval(chrom)
        eval_count += 1
        pop_objectives.append((f1, f2))

    # Referans Noktası Belirleme (HV için geniş güvenli sınır)
    max_init_f1 = max(pt[0] for pt in pop_objectives)
    max_init_f2 = max(pt[1] for pt in pop_objectives)
    ref_point = (max(500.0, max_init_f1 * 1.35 + 50), max(500.0, max_init_f2 * 1.35 + 50))

    # Takip Geçmişleri
    hv_history: List[float] = []
    front_size_history: List[int] = []
    min_f1_history: List[float] = []
    min_f2_history: List[float] = []
    avg_f1_history: List[float] = []
    avg_f2_history: List[float] = []
    best_total_score_history: List[float] = []

    # =========================================================================
    # EVRİMSEL JENERASYON DÖNGÜSÜ (NSGA-II MAIN GENERATIONS LOOP)
    # =========================================================================
    effective_generations = 100000 if time_limit else generations
    for gen in range(effective_generations):
        if time_limit and (time.time() - start_time) >= time_limit:
            break

        # ---------------------------------------------------------------------
        # ADIM 1: HIZLI BASKIN OLMAYAN SIRALAMA (FAST NON-DOMINATED SORT)
        # ---------------------------------------------------------------------
        fronts, ranks = fast_non_dominated_sort(pop_objectives)

        # ---------------------------------------------------------------------
        # ADIM 2: HER FRONT İÇİN KALABALIKLAŞMA MESAFESİ ATAMASI
        # ---------------------------------------------------------------------
        distances: Dict[int, float] = {}
        for front in fronts:
            f_dists = calculate_crowding_distance(front, pop_objectives)
            distances.update(f_dists)

        # Metrikleri kaydet
        rank1_indices = fronts[0] if len(fronts) > 0 else []
        rank1_objectives = [pop_objectives[idx] for idx in rank1_indices]
        
        # Tekil Pareto noktalarını ayıkla (HV ve Spread için)
        unique_pareto_pts = list(set(rank1_objectives))
        curr_hv = calculate_hypervolume_2d(unique_pareto_pts, ref_point)
        
        min_f1 = min(pt[0] for pt in pop_objectives)
        min_f2 = min(pt[1] for pt in pop_objectives)
        avg_f1 = float(np.mean([pt[0] for pt in pop_objectives]))
        avg_f2 = float(np.mean([pt[1] for pt in pop_objectives]))
        best_tot = min(pt[0] + pt[1] for pt in pop_objectives)

        hv_history.append(curr_hv)
        front_size_history.append(len(unique_pareto_pts))
        min_f1_history.append(min_f1)
        min_f2_history.append(min_f2)
        avg_f1_history.append(avg_f1)
        avg_f2_history.append(avg_f2)
        best_total_score_history.append(best_tot)

        if callback and (gen % max(1, stream_interval) == 0):
            status_msg = f"| Pareto: {len(unique_pareto_pts)} Çözüm | Min f1: {min_f1:.0f} | Min f2: {min_f2:.0f}"
            callback(gen + 1, best_tot, float(np.mean([pt[0] + pt[1] for pt in pop_objectives])), status_msg)

        # ---------------------------------------------------------------------
        # ADIM 3: YAVRU POPÜLASYON ÜRETİMİ (OFFSPRING REPRODUCTION Q_t)
        # ---------------------------------------------------------------------
        offspring_pop: List[np.ndarray] = []
        offspring_objectives: List[Tuple[float, float]] = []

        while len(offspring_pop) < pop_size:
            # Turnuva Seçimi (Crowded Comparison Operator)
            p1 = binary_tournament_selection(population, ranks, distances)
            p2 = binary_tournament_selection(population, ranks, distances)

            # Çaprazlama
            child = crossover_daywise(
                p1, p2, n_days,
                workers=workers, n_workers=n_workers, shift_reqs=shift_reqs,
                crossover_rate=crossover_rate
            )

            # Mutasyon
            child = mutate_swap(
                child, n_workers, n_days, workers, shift_reqs,
                mutation_rate=mutation_rate
            )

            c_f1, c_f2 = fast_eval(child)
            eval_count += 1
            offspring_pop.append(child)
            offspring_objectives.append((c_f1, c_f2))

        # ---------------------------------------------------------------------
        # ADIM 4: (N + N) -> N ELİTİST POPÜLASYON SEÇİMİ (R_t = P_t ∪ Q_t)
        # ---------------------------------------------------------------------
        combined_pop = population + offspring_pop
        combined_objectives = pop_objectives + offspring_objectives

        combined_fronts, combined_ranks = fast_non_dominated_sort(combined_objectives)

        next_population: List[np.ndarray] = []
        next_objectives: List[Tuple[float, float]] = []

        for front in combined_fronts:
            if len(next_population) + len(front) <= pop_size:
                # Front tamamen sığıyorsa doğrudan ekle
                for idx in front:
                    next_population.append(combined_pop[idx])
                    next_objectives.append(combined_objectives[idx])
            else:
                # Front tamamen sığmıyorsa Kalabalıklaşma Mesafesine göre sıralayıp kalan kontenjanı doldur
                needed = pop_size - len(next_population)
                front_dists = calculate_crowding_distance(front, combined_objectives)
                # Mesafesi en büyük olandan en küçük olana doğru diz
                sorted_front_by_dist = sorted(front, key=lambda idx: front_dists.get(idx, 0.0), reverse=True)
                for idx in sorted_front_by_dist[:needed]:
                    next_population.append(combined_pop[idx])
                    next_objectives.append(combined_objectives[idx])
                break

        population = next_population
        pop_objectives = next_objectives

    # =========================================================================
    # NİHAİ PARETO ÖN CEPHESİ VE ÇÖZÜMLERİN PAKETLENMESİ
    # =========================================================================
    final_fronts, final_ranks = fast_non_dominated_sort(pop_objectives)
    final_distances: Dict[int, float] = {}
    for front in final_fronts:
        final_distances.update(calculate_crowding_distance(front, pop_objectives))

    rank1_indices = final_fronts[0] if len(final_fronts) > 0 else list(range(len(population)))

    # Tekil Pareto çözümlerini çıkar (Aynı f1, f2 skoruna sahip kopyaları birleştir)
    seen_coords = set()
    pareto_solutions_list: List[Dict[str, Any]] = []

    for idx in rank1_indices:
        sched = population[idx]
        f1_val, f2_val = pop_objectives[idx]
        coord_key = (round(f1_val, 1), round(f2_val, 1))
        if coord_key in seen_coords:
            continue
        seen_coords.add(coord_key)

        penalties_dict, total_score = calculate_full_penalties(sched, workers, n_workers, n_days, weights)
        is_feas, hard_viols, hard_logs = audit_all_hard_constraints(sched, workers, n_workers, n_days, shift_reqs)
        req_details = build_worker_request_details(sched, workers, n_days, weights)

        pareto_solutions_list.append({
            'schedule': sched,
            'f1': float(f1_val),
            'f2': float(f2_val),
            'total_score': int(total_score),
            'penalties': penalties_dict,
            'is_feasible': is_feas,
            'hard_violations_count': hard_viols,
            'hard_violation_logs': hard_logs,
            'request_details': req_details,
            'crowding_distance': float(final_distances.get(idx, 0.0)),
            'rank': 1
        })

    # Pareto listesini f1'e göre sırala
    pareto_solutions_list.sort(key=lambda s: s['f1'])

    # Pareto Hedef Noktaları
    pareto_pts = [(s['f1'], s['f2']) for s in pareto_solutions_list]
    final_hv = calculate_hypervolume_2d(pareto_pts, ref_point)
    final_spread = calculate_spread_metric(pareto_pts)

    # 1. İşletme Odaklı Uç Çözüm (Min f1)
    extreme_f1_idx = min(range(len(pareto_solutions_list)), key=lambda i: pareto_solutions_list[i]['f1'])
    extreme_f1_solution = pareto_solutions_list[extreme_f1_idx]

    # 2. Çalışan Odaklı Uç Çözüm (Min f2)
    extreme_f2_idx = min(range(len(pareto_solutions_list)), key=lambda i: pareto_solutions_list[i]['f2'])
    extreme_f2_solution = pareto_solutions_list[extreme_f2_idx]

    # 3. Diz Noktası (Knee Point / Altın Denge)
    knee_idx = find_knee_point_index(pareto_pts)
    knee_solution = pareto_solutions_list[knee_idx]

    # Tüm Popülasyon Noktaları (Görsel Dağılım İçin)
    all_evaluated_solutions: List[Dict[str, Any]] = []
    for i, chrom in enumerate(population):
        all_evaluated_solutions.append({
            'f1': float(pop_objectives[i][0]),
            'f2': float(pop_objectives[i][1]),
            'total_score': float(pop_objectives[i][0] + pop_objectives[i][1]),
            'rank': int(final_ranks[i]),
            'crowding_distance': float(final_distances.get(i, 0.0))
        })

    exec_time_ms = round((time.time() - start_time) * 1000, 1)
    term_reason = f"🧬 Deb et al. (2002) NSGA-II tamamlandı. {generations} nesil sonunda {len(pareto_solutions_list)} adet baskın olmayan Pareto optimal çözüm üretildi."

    # Ana Kanonik Çözüm Olarak Diz Noktasını Belirle
    primary_solution = knee_solution

    # Standart Sözleşme Formatı Üretimi
    standard_result = build_standard_solver_result(
        schedule=primary_solution['schedule'],
        workers=workers,
        is_feasible=primary_solution['is_feasible'],
        hard_violations_count=primary_solution['hard_violations_count'],
        hard_violation_logs=primary_solution['hard_violation_logs'],
        final_score=primary_solution['total_score'],
        initial_score=int(pop_objectives[0][0] + pop_objectives[0][1]),
        improvement_rate=0.0,
        exec_time_ms=exec_time_ms,
        eval_count=eval_count,
        total_iterations=generations,
        termination_reason=term_reason,
        penalties=primary_solution['penalties'],
        request_details=primary_solution['request_details'],
        score_history=best_total_score_history,
        meta={
            'generations': generations,
            'population_size': pop_size,
            'pareto_front_size': len(pareto_solutions_list),
            'final_hypervolume': final_hv,
            'final_spread_delta': final_spread,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'ref_point': ref_point
        }
    )

    return {
        'standard_result': standard_result,
        'pareto_solutions': pareto_solutions_list,
        'knee_solution': knee_solution,
        'extreme_f1_solution': extreme_f1_solution,
        'extreme_f2_solution': extreme_f2_solution,
        'all_evaluated_solutions': all_evaluated_solutions,
        'metrics': {
            'hv_history': hv_history,
            'front_size_history': front_size_history,
            'min_f1_history': min_f1_history,
            'min_f2_history': min_f2_history,
            'avg_f1_history': avg_f1_history,
            'avg_f2_history': avg_f2_history,
            'final_hypervolume': final_hv,
            'final_spread_delta': final_spread,
            'total_evaluations': eval_count,
            'exec_time_ms': exec_time_ms,
            'generations': generations,
            'pop_size': pop_size,
            'ref_point': ref_point
        },
        'workers': workers,
        'n_workers': n_workers,
        'n_days': n_days,
        'shift_reqs': shift_reqs,
        'weights': weights
    }
