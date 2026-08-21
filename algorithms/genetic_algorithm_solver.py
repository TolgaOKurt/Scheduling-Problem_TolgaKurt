"""
================================================================================
  ALGORITHMS/GENETIC_ALGORITHM_SOLVER.PY - GENETİK ALGORİTMA EVRİMSEL ÇÖZÜCÜ
================================================================================
  Bu modül, Vardiya Çizelgeleme Problemi (NSP) için Genetik Algoritma (GA)
  stokastik evrimsel arama motorunu içerir. Canlı akış (real-time callback) destekler.
================================================================================
"""

import time
import random
import numpy as np
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm, get_best_greedy_initial_solution
from algorithms.penalty_calculator import calculate_full_penalties, check_swap_feasibility, build_worker_request_details, audit_all_hard_constraints
from algorithms.evolutionary_engine import initialize_population, tournament_selection
from algorithms.solver_contract import build_standard_solver_result


def run_genetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, 
                          pop_size=50, generations=100, crossover_rate=0.85, 
                          mutation_rate=0.05, elitism_count=2, seed=42, custom_workers=None,
                          callback=None, stream_interval=2, time_limit=None):
    """
    Genetik Algoritma (Evrimsel Arama) Vardiya Optimizasyon Motoru.
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # =========================================================================
    # 1. BAŞLANGIÇ POPÜLASYONU ÜRETİMİ (POPULATION INITIALIZATION & WARM-START)
    # =========================================================================
    # 1. Birey olarak en iyi Greedy çözümü alınır; kalan (pop_size - 1) birey
    # geçerli swap mutasyonları ile çeşitlendirilerek popülasyon oluşturulur.
    greedy_seed = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=custom_workers
    )
    greedy_sched = greedy_seed['schedule']
    workers = greedy_seed['workers']

    population = initialize_population(greedy_sched, pop_size, n_workers, n_days, workers, shift_reqs)

    # =========================================================================
    # 2. BAŞLANGIÇ UYGUNLUK (FITNESS) DEĞERLENDİRMESİ
    # =========================================================================
    eval_count = 0
    initial_scores = []
    for chrom in population:
        _, score = calculate_full_penalties(chrom, workers, n_workers, n_days, weights)
        eval_count += 1
        initial_scores.append(score)

    initial_baseline_score = int(np.mean(initial_scores))

    best_score_history = []
    avg_score_history = []
    diversity_history = []

    best_chromosome = population[np.argmin(initial_scores)].copy()
    best_score = min(initial_scores)

    if callback:
        callback(0, best_score, initial_baseline_score, f"| Pop: {pop_size}")

    # =========================================================================
    # 3. EVRİMSEL JENERASYON DÖNGÜSÜ (EVOLUTIONARY GENERATIONS LOOP)
    # =========================================================================
    effective_generations = 100000 if time_limit else generations
    for gen in range(effective_generations):
        if time_limit and (time.time() - start_time) >= time_limit:
            break
        # ---------------------------------------------------------------------
        # 3.1 TÜM POPÜLASYONUN DEĞERLENDİRİLMESİ (FITNESS EVALUATION)
        # ---------------------------------------------------------------------
        scores = []
        for chrom in population:
            _, score = calculate_full_penalties(chrom, workers, n_workers, n_days, weights)
            eval_count += 1
            scores.append(score)

        # Şampiyon kromozom kontrolü ve güncellemesi
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_chromosome = population[min_idx].copy()
            if callback:
                callback(gen + 1, best_score, float(np.mean(scores)), f"| Ort: {np.mean(scores):.0f}")

        best_score_history.append(best_score)
        avg_score_history.append(float(np.mean(scores)))
        diversity_history.append(float(np.std(scores)))

        if callback and (gen % max(1, stream_interval) == 0):
            callback(gen + 1, best_score, float(np.mean(scores)), f"| Ort: {np.mean(scores):.0f}")

        # Kromozomları başarı sırasına göre diz (Fitness Ranking)
        sorted_indices = np.argsort(scores)
        sorted_pop = [population[idx] for idx in sorted_indices]

        # ---------------------------------------------------------------------
        # 3.2 ELİTİZM OPERATÖRÜ (ELITISM SELECTION)
        # ---------------------------------------------------------------------
        # Popülasyonun en iyi 'elitism_count' adet şampiyonu hiçbir mutasyon veya
        # çaprazlamaya maruz kalmadan doğrudan bir sonraki nesle aktarılır.
        next_population = []
        for e in range(min(elitism_count, pop_size)):
            next_population.append(sorted_pop[e].copy())

        # ---------------------------------------------------------------------
        # 3.3 SEÇİM, ÇAPRAZLAMA VE MUTASYON DÖNGÜSÜ (CROSSOVER & MUTATION REPRODUCTION)
        # ---------------------------------------------------------------------
        while len(next_population) < pop_size:
            # Turnuva Seçimi: En uygun ebeveynler seçilir (Tournament Selection)
            p1 = tournament_selection(population, scores, k=3)
            p2 = tournament_selection(population, scores, k=3)

            child = p1.copy()

            # Gün Bazlı Çaprazlama (Day-wise Cut-and-Cross Crossover)
            if random.random() < crossover_rate:
                split_day = random.randint(1, n_days - 1)
                child_candidate = child.copy()
                child_candidate[:, split_day:] = p2[:, split_day:]
                # Çaprazlama sonrası sert kısıt uygunluk denetimi
                cand_feasible, _, _ = audit_all_hard_constraints(child_candidate, workers, n_workers, n_days, shift_reqs)
                if cand_feasible:
                    child = child_candidate

            # Takas Mutasyonu (Feasible Swap Mutation)
            if random.random() < mutation_rate:
                for _ in range(random.randint(1, 3)):
                    mut_day = random.randint(0, n_days - 1)
                    w1, w2 = random.sample(range(n_workers), 2)
                    if child[w1, mut_day] != child[w2, mut_day]:
                        temp_child = child.copy()
                        temp_child[w1, mut_day], temp_child[w2, mut_day] = temp_child[w2, mut_day], temp_child[w1, mut_day]
                        # Sert kısıt kontrolü
                        if check_swap_feasibility(temp_child, workers, mut_day, w1, w2, n_workers, n_days, shift_reqs):
                            child = temp_child

            next_population.append(child)

        # Yeni nesle geçiş
        population = next_population

    # =========================================================================
    # 4. NİHAİ EN İYİ ÇÖZÜMÜN DOĞRULANMASI VE RAPORLANMASI
    # =========================================================================
    final_penalties, final_score = calculate_full_penalties(best_chromosome, workers, n_workers, n_days, weights)
    eval_count += 1
    
    if callback:
        callback(generations, final_score, final_score, f"| Bitti")

    improvement_rate = 0.0
    if initial_baseline_score > final_score and initial_baseline_score > 0:
        improvement_rate = round(((initial_baseline_score - final_score) / initial_baseline_score) * 100, 2)

    exec_time_ms = round((time.time() - start_time) * 1000, 1)
    term_reason = f"🧬 Belirlenen {generations} jenerasyonluk evrimsel süreç tamamlandı. Popülasyon elitizasyonu ve çaprazlama ile en iyi kromozom elde edildi."

    request_details = build_worker_request_details(best_chromosome, workers, n_days, weights)
    is_feasible, hard_viols_count, hard_violation_logs = audit_all_hard_constraints(
        best_chromosome, workers, n_workers, n_days, shift_reqs
    )

    return build_standard_solver_result(
        schedule=best_chromosome,
        workers=workers,
        is_feasible=is_feasible,
        hard_violations_count=hard_viols_count,
        hard_violation_logs=hard_violation_logs,
        final_score=final_score,
        initial_score=initial_baseline_score,
        improvement_rate=improvement_rate,
        exec_time_ms=exec_time_ms,
        eval_count=eval_count,
        total_iterations=generations,
        termination_reason=term_reason,
        penalties=final_penalties,
        request_details=request_details,
        score_history=best_score_history,
        meta={
            'generations': generations,
            'population_size': pop_size,
            'avg_score_history': avg_score_history,
            'diversity_history': diversity_history,
            'seed_source': greedy_seed.get('meta', {}).get('seed_source', 'Greedy'),
            'is_csp_fallback': greedy_seed.get('meta', {}).get('is_csp_fallback', False),
            'fallback_reason': greedy_seed.get('meta', {}).get('fallback_reason', '')
        }
    )
