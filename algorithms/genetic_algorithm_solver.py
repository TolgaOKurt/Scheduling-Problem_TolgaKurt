"""
================================================================================
  ALGORITHMS/GENETIC_ALGORITHM_SOLVER.PY - GENETİK ALGORİTMA EVRİMSEL ÇÖZÜCÜ
================================================================================
  Bu modül, Vardiya Çizelgeleme Problemi (NSP) için Genetik Algoritma (GA)
  stokastik evrimsel arama motorunu içerir.
  
  Kromozom Temsili: İşçi x Gün boyutlu vardiya atama matrisi (0:OFF, 1:G, 2:A, 3:N)
  Evrimsel Adımlar: Popülasyon İlklendirmesi -> Uygunluk Değerlendirmesi -> Turnuva Seçimi ->
                    Gün Bazlı Çaprazlama -> Sert Kısıt Korumalı Mutasyon -> Elitizm
================================================================================
"""

import time
import random
import numpy as np
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm, get_best_greedy_initial_solution
from algorithms.penalty_calculator import calculate_full_penalties, check_swap_feasibility, build_worker_request_details
from algorithms.evolutionary_engine import initialize_population, tournament_selection


def run_genetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, 
                          pop_size=50, generations=100, crossover_rate=0.85, 
                          mutation_rate=0.05, elitism_count=2, seed=42, custom_workers=None):
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

    # Base Greedy Çözümü (Best-of-Heuristics ile En İyi Greedy Çözümü)
    greedy_sched, workers, _, _, _, _, _ = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=custom_workers
    )

    # 1. Popülasyon İlklendirmesi (Merkezi Evrimsel Motor)
    population = initialize_population(greedy_sched, pop_size, n_workers, n_days, workers, shift_reqs)

    # Popülasyonun Evrim Öncesi İlk Skorlarını Değerlendir
    eval_count = 0
    initial_scores = []
    for chrom in population:
        _, score = calculate_full_penalties(chrom, workers, n_workers, n_days, weights)
        eval_count += 1
        initial_scores.append(score)

    # Başlangıç referans skoru: Popülasyonun evrim öncesi ortalama/taban skoru
    initial_baseline_score = int(np.mean(initial_scores))

    best_score_history = []
    avg_score_history = []
    diversity_history = []

    best_chromosome = population[np.argmin(initial_scores)].copy()
    best_score = min(initial_scores)

    # 2. Evrimsel Jenerasyon Döngüsü (Generations Loop)
    for gen in range(generations):
        scores = []
        for chrom in population:
            _, score = calculate_full_penalties(chrom, workers, n_workers, n_days, weights)
            eval_count += 1
            scores.append(score)

        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_chromosome = population[min_idx].copy()

        best_score_history.append(best_score)
        avg_score_history.append(float(np.mean(scores)))
        diversity_history.append(float(np.std(scores)))

        sorted_indices = np.argsort(scores)
        sorted_pop = [population[idx] for idx in sorted_indices]

        # Elitizm (En iyi e bireyi koru)
        next_population = []
        for e in range(min(elitism_count, pop_size)):
            next_population.append(sorted_pop[e].copy())

        # Crossover & Mutation
        while len(next_population) < pop_size:
            p1 = tournament_selection(population, scores, k=3)
            p2 = tournament_selection(population, scores, k=3)

            child = p1.copy()

            # Gün bazlı Çaprazlama (Crossover) - sınır dinlenme kontrolü ile
            if random.random() < crossover_rate:
                split_day = random.randint(1, n_days - 1)
                child_candidate = child.copy()
                child_candidate[:, split_day:] = p2[:, split_day:]
                boundary_ok = True
                for _wid in range(n_workers):
                    prev = child_candidate[_wid, split_day - 1]
                    nxt  = child_candidate[_wid, split_day]
                    if (prev == 3 and nxt == 1) or (prev == 2 and nxt == 1):
                        boundary_ok = False
                        break
                if boundary_ok:
                    child = child_candidate

            # Mutasyon (Mutation)
            if random.random() < mutation_rate:
                for _ in range(random.randint(1, 3)):
                    mut_day = random.randint(0, n_days - 1)
                    w1, w2 = random.sample(range(n_workers), 2)
                    if child[w1, mut_day] != child[w2, mut_day]:
                        temp_child = child.copy()
                        temp_child[w1, mut_day], temp_child[w2, mut_day] = temp_child[w2, mut_day], temp_child[w1, mut_day]
                        if check_swap_feasibility(temp_child, workers, mut_day, w1, w2, n_workers, n_days, shift_reqs):
                            child = temp_child

            next_population.append(child)

        population = next_population

    final_penalties, final_score = calculate_full_penalties(best_chromosome, workers, n_workers, n_days, weights)
    eval_count += 1
    
    improvement_rate = 0.0
    if initial_baseline_score > final_score and initial_baseline_score > 0:
        improvement_rate = round(((initial_baseline_score - final_score) / initial_baseline_score) * 100, 2)

    exec_time_ms = round((time.time() - start_time) * 1000, 1)
    term_reason = f"🧬 Belirlenen {generations} jenerasyonluk evrimsel süreç tamamlandı. Popülasyon elitizasyonu ve çaprazlama ile en iyi kromozom elde edildi."

    return {
        'schedule': best_chromosome,
        'workers': workers,
        'initial_score': initial_baseline_score,
        'final_score': final_score,
        'improvement_rate': improvement_rate,
        'eval_count': eval_count,
        'exec_time_ms': exec_time_ms,
        'penalties': final_penalties,
        'best_score_history': best_score_history,
        'avg_score_history': avg_score_history,
        'diversity_history': diversity_history,
        'generations_run': generations,
        'pop_size': pop_size,
        'termination_reason': term_reason,
        'request_details': build_worker_request_details(best_chromosome, workers, n_days, weights)
    }
