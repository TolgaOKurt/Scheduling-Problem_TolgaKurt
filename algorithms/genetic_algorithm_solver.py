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
            
    # MYK Zorunlu Sertifika Kontrolü (4 temel sertifika)
    required_skills = {'Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu'}
    for k in [1, 2, 3]:
        if not required_skills.issubset(shift_skills[k]):
            return False
            
    return True

def check_swap_feasibility(schedule, workers, day, w1_idx, w2_idx, n_workers, n_days, shift_reqs):
    """Takas adımı sonrası tüm sert kısıtların (Kadro min, MYK, 11h dinlenme ve 7 günlük izin) korunduğunu doğrular."""
    if not check_hard_constraints_single_day(schedule, workers, day, n_workers, shift_reqs):
        return False
        
    s1 = schedule[w1_idx, day]
    s2 = schedule[w2_idx, day]

    if day > 0 and schedule[w1_idx, day-1] == 3 and s1 == 1: return False
    if day < n_days - 1 and s1 == 3 and schedule[w1_idx, day+1] == 1: return False
    if day > 0 and schedule[w2_idx, day-1] == 3 and s2 == 1: return False
    if day < n_days - 1 and s2 == 3 and schedule[w2_idx, day+1] == 1: return False

    for wid in [w1_idx, w2_idx]:
        start_tau = max(0, day - 6)
        end_tau = min(n_days - 7, day)
        for tau in range(start_tau, end_tau + 1):
            if np.sum(schedule[wid, tau:tau+7] == 0) == 0:
                return False

    return True

def calculate_full_penalties(schedule, workers, n_workers, n_days, weights):
    """Verilen tam vardiya matrisinin tüm yumuşak kısıt ceza puanlarını hesaplar."""
    w_circ = weights.get('circadian', 50)
    w_night_imb = weights.get('night_imb', 25)
    w_exp = weights.get('exp_mix', 30)
    w_pref = weights.get('pref_off', 40)

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
    for t in range(n_days):
        for k in [1, 2, 3]:
            shift_wids = [w['id'] for w in workers if schedule[w['id'], t] == k]
            ustas = sum(1 for wid in shift_wids if workers[wid]['is_usta'])
            if len(shift_wids) > 0 and ustas == 0:
                penalties['Kıdem & MYK Sertifika Eksikliği'] += w_exp

    # 3. Posta Takım Bütünlüğü
    posta_members = {}
    for w in workers:
        p = w['posta']
        if p not in posta_members:
            posta_members[p] = []
        posta_members[p].append(w['id'])

    for p, wids in posta_members.items():
        if len(wids) > 1:
            for t in range(n_days):
                active_shifts = [schedule[wid, t] for wid in wids if schedule[wid, t] != 0]
                if len(active_shifts) > 1:
                    counts = [active_shifts.count(s) for s in set(active_shifts)]
                    majority = max(counts)
                    deviated = len(active_shifts) - majority
                    penalties['Posta Takım Bütünlüğü İhlali'] += deviated * weights.get('posta', weights.get('posta_unity', 15))

    return penalties, sum(penalties.values())

def run_genetic_algorithm(n_workers, n_days, r_day, r_eve, r_night, weights, 
                          pop_size=50, generations=100, crossover_rate=0.85, 
                          mutation_rate=0.05, elitism_count=2, seed=42, custom_workers=None):
    """
    Genetik Algorithma (Evrimsel Arama) Vardiya Optimizasyon Motoru.
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # Base Greedy Çözümü (1. Kromozom)
    greedy_sched, workers, _, _, _, _, _ = run_greedy_algorithm(
        n_workers, n_days, r_day, r_eve, r_night, weights, solver_mode="Akıllı Kademeli Greedy (İzinleri Günlere Yayan)", custom_workers=custom_workers
    )

    # 1. Popülasyon İlklendirmesi (Initial Population)
    population = []
    
    # 1. Birey: Akıllı Greedy Çözümü
    population.append(greedy_sched.copy())
    
    # Kalan popülasyonu çeşitlilik sunan rastgele geçerli matrisler ve Greedy türevleri ile doldur
    for idx in range(1, pop_size):
        chrom = greedy_sched.copy()
        # Her bireye rastgele farklı sayıda (3 ila 12) vardiya takası uygulayarak gerçek evrimsel çeşitlilik sağla
        num_swaps = random.randint(3, 12)
        for _ in range(num_swaps):
            d = random.randint(0, n_days - 1)
            w1, w2 = random.sample(range(n_workers), 2)
            if chrom[w1, d] != chrom[w2, d]:
                temp_chrom = chrom.copy()
                temp_chrom[w1, d], temp_chrom[w2, d] = temp_chrom[w2, d], temp_chrom[w1, d]
                if check_swap_feasibility(temp_chrom, workers, d, w1, w2, n_workers, n_days, shift_reqs):
                    chrom = temp_chrom
        population.append(chrom)

    # Popülasyonun Evrim Öncesi İlk Skorlarını Değerlendir
    initial_scores = []
    for chrom in population:
        _, score = calculate_full_penalties(chrom, workers, n_workers, n_days, weights)
        initial_scores.append(score)

    # Başlangıç referans skoru: Popülasyonun evrim öncesi ortalama/taban skoru
    initial_baseline_score = int(np.mean(initial_scores))

    best_score_history = []
    avg_score_history = []
    diversity_history = []

    best_chromosome = population[np.argmin(initial_scores)].copy()
    best_score = min(initial_scores)

    def tournament_selection(pop, scores, k=3):
        selected_indices = random.sample(range(len(pop)), k)
        best_idx = min(selected_indices, key=lambda idx: scores[idx])
        return pop[best_idx].copy()

    # 2. Evrimsel Jenerasyon Döngüsü (Generations Loop)
    for gen in range(generations):
        scores = []
        for chrom in population:
            _, score = calculate_full_penalties(chrom, workers, n_workers, n_days, weights)
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

            # Gün bazlı Çaprazlama (Crossover) - NumPy dilimleme ile anlık yıldırım hızında
            if random.random() < crossover_rate:
                split_day = random.randint(1, n_days - 1)
                child[:, split_day:] = p2[:, split_day:]

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
        'exec_time_ms': exec_time_ms,
        'penalties': final_penalties,
        'best_score_history': best_score_history,
        'avg_score_history': avg_score_history,
        'diversity_history': diversity_history,
        'generations_run': generations,
        'pop_size': pop_size,
        'termination_reason': term_reason
    }
