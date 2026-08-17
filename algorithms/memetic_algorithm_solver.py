"""
================================================================================
  ALGORITHMS/MEMETIC_ALGORITHM_SOLVER.PY - MEMETİK ALGORİTMA (HYBRID GA + HC)
================================================================================
  Bu modül, Vardiya Çizelgeleme Problemi (NSP) için Memetik Algoritma (MA)
  hibrit evrimsel ve lokal arama motorunu içerir.
  
  Mimari: Genetik Algoritma (Global Popülasyon Araması) + Hill Climbing (Lokal Cerrahi Tamir)
================================================================================
"""

import time
import math
import random
import numpy as np
from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm, get_best_greedy_initial_solution
from algorithms.penalty_calculator import calculate_full_penalties, check_swap_feasibility, build_worker_request_details
from algorithms.evolutionary_engine import initialize_population, tournament_selection


def local_search_refinement(chromosome, workers, n_workers, n_days, shift_reqs, weights, depth=5, eval_tracker=None):
    """
    Memetik Bireysel İyileştirme (Hill Climbing Refinement):
    Çaprazlama sonrası oluşan çocuk kromozoma nokta atışı mikro takaslar uygulayarak 
    çaprazlama dikiş noktalarında kırılan sirkadiyen/posta kurallarını lokal olarak tamir eder.
    """
    current = chromosome.copy()
    _, current_score = calculate_full_penalties(current, workers, n_workers, n_days, weights)
    if eval_tracker is not None:
        eval_tracker[0] += 1
    
    for _ in range(depth):
        d = random.randint(0, n_days - 1)
        w1, w2 = random.sample(range(n_workers), 2)
        if current[w1, d] != current[w2, d]:
            cand = current.copy()
            cand[w1, d], cand[w2, d] = cand[w2, d], cand[w1, d]
            if check_swap_feasibility(cand, workers, d, w1, w2, n_workers, n_days, shift_reqs):
                _, cand_score = calculate_full_penalties(cand, workers, n_workers, n_days, weights)
                if eval_tracker is not None:
                    eval_tracker[0] += 1
                if cand_score < current_score:
                    current = cand
                    current_score = cand_score
                    
    return current

def run_memetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, 
                         pop_size=50, generations=100, crossover_rate=0.85, 
                         mutation_rate=0.05, local_search_depth=5, elitism_count=2, seed=42, custom_workers=None):
    """
    Memetik Algoritma (Hibrit Genetik + Tepeden Tırmanma Lokal Arama) Motoru.
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    eval_tracker = [0]

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
    base_population = initialize_population(greedy_sched, pop_size, n_workers, n_days, workers, shift_reqs)
    population = []
    for chrom in base_population:
        # İlk popülasyondaki bireylere hafif memetik lokal tamir uygula
        chrom_refined = local_search_refinement(chrom, workers, n_workers, n_days, shift_reqs, weights, depth=3, eval_tracker=eval_tracker)
        population.append(chrom_refined)

    initial_scores = []
    for chrom in population:
        _, score = calculate_full_penalties(chrom, workers, n_workers, n_days, weights)
        eval_tracker[0] += 1
        initial_scores.append(score)

    initial_baseline_score = int(np.mean(initial_scores))

    best_score_history = []
    avg_score_history = []
    diversity_history = []

    best_chromosome = population[np.argmin(initial_scores)].copy()
    best_score = min(initial_scores)

    # 2. Evrimsel & Memetik İyileştirme Döngüsü
    for gen in range(generations):
        scores = []
        for chrom in population:
            _, score = calculate_full_penalties(chrom, workers, n_workers, n_days, weights)
            eval_tracker[0] += 1
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

        # Crossover, Mutation & Memetic Local Search Refinement
        while len(next_population) < pop_size:
            p1 = tournament_selection(population, scores, k=3)
            p2 = tournament_selection(population, scores, k=3)

            child = p1.copy()

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

            if random.random() < mutation_rate:
                for _ in range(random.randint(1, 3)):
                    mut_day = random.randint(0, n_days - 1)
                    w1, w2 = random.sample(range(n_workers), 2)
                    if child[w1, mut_day] != child[w2, mut_day]:
                        temp_child = child.copy()
                        temp_child[w1, mut_day], temp_child[w2, mut_day] = temp_child[w2, mut_day], temp_child[w1, mut_day]
                        if check_swap_feasibility(temp_child, workers, mut_day, w1, w2, n_workers, n_days, shift_reqs):
                            child = temp_child

            # --- MEMETİK LOKAL ARAMA İYİLEŞTİRMESİ (HILL CLIMBING TAMİRİ) ---
            child = local_search_refinement(child, workers, n_workers, n_days, shift_reqs, weights, depth=local_search_depth, eval_tracker=eval_tracker)
            next_population.append(child)

        population = next_population

    final_penalties, final_score = calculate_full_penalties(best_chromosome, workers, n_workers, n_days, weights)
    eval_tracker[0] += 1
    
    improvement_rate = 0.0
    if initial_baseline_score > final_score and initial_baseline_score > 0:
        improvement_rate = round(((initial_baseline_score - final_score) / initial_baseline_score) * 100, 2)

    exec_time_ms = round((time.time() - start_time) * 1000, 1)
    term_reason = f"🧬 Memetik Algoritma {generations} nesillik evrim ve Hill Climbing yerel tamir adımlarını başarıyla tamamladı."

    return {
        'schedule': best_chromosome,
        'workers': workers,
        'initial_score': initial_baseline_score,
        'final_score': final_score,
        'improvement_rate': improvement_rate,
        'eval_count': eval_tracker[0],
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
