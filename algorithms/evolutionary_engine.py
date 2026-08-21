"""
================================================================================
  ALGORITHMS/EVOLUTIONARY_ENGINE.PY - ORTAK EVRİMSEL OPERATÖRLER MOTORU
================================================================================
  Bu modül, popülasyon tabanlı optimizasyon algoritmaları olan Genetik Algoritma (GA)
  ve Memetik Algoritma (MA) için ortak olan:
  - Popülasyon İlklendirmesi (Initial Population Seeding & Swaps)
  - Turnuva Seçim Operatörü (Tournament Selection)
  - Gün Bazlı Çaprazlama (Crossover)
  - Sert Kısıt Korumalı Takas Mutasyonu (Mutation)
  işlevlerini tek bir standart merkezden sunar.
================================================================================
"""

import random
import numpy as np
from algorithms.penalty_calculator import check_swap_feasibility


# =============================================================================
# 1. POPÜLASYON İLKLENDİRME OPERATÖRÜ (POPULATION INITIALIZATION & SEEDING)
# =============================================================================
def initialize_population(greedy_sched, pop_size, n_workers, n_days, workers, shift_reqs):
    """
    1. Birey olarak en iyi Greedy temel çözümünü alır (Warm-Start), kalan (pop_size - 1) bireyi
    3 ila 12 adet rastgele sert kısıt korumalı takas mutasyonu ile çeşitlendirerek (Diversification)
    yüksek kaliteli ve geçerli (%100 Feasible) bir başlangıç popülasyonu üretir.
    """
    population = [greedy_sched.copy()]

    for _ in range(1, pop_size):
        chrom = greedy_sched.copy()
        # Her birey için 3-12 arası geçerli swap yapılarak gen havuzu çeşitlendirilir
        num_swaps = random.randint(3, 12)
        for _ in range(num_swaps):
            d = random.randint(0, n_days - 1)
            w1, w2 = random.sample(range(n_workers), 2)
            if chrom[w1, d] != chrom[w2, d]:
                temp_chrom = chrom.copy()
                temp_chrom[w1, d], temp_chrom[w2, d] = temp_chrom[w2, d], temp_chrom[w1, d]
                # Sert kısıt kontrolü
                if check_swap_feasibility(temp_chrom, workers, d, w1, w2, n_workers, n_days, shift_reqs):
                    chrom = temp_chrom
        population.append(chrom)

    return population


# =============================================================================
# 2. TURNUVA SEÇİM OPERATÖRÜ (TOURNAMENT SELECTION)
# =============================================================================
def tournament_selection(population, scores, k=3):
    """
    Popülasyondan rastgele k adet birey seçer (Turnuva Havuzu); aralarından en düşük
    ceza puanına (en yüksek uygunluk / fitness) sahip olan şampiyon ebeveyni kopyalar.
    """
    selected_indices = random.sample(range(len(population)), k)
    best_idx = min(selected_indices, key=lambda idx: scores[idx])
    return population[best_idx].copy()


# =============================================================================
# 3. GÜN BAZLI ÇAPRAZLAMA OPERATÖRÜ (DAY-WISE ONE-POINT CROSSOVER)
# =============================================================================
def crossover_daywise(parent1, parent2, n_days, workers=None, n_workers=None, shift_reqs=None, crossover_rate=0.85):
    """
    İki ebeveyn kromozomu gün sütunları bazında rastgele bir kesim noktasından (cut point)
    birleştirerek çocuk kromozom üretir. Günlük vardiya toplamları bozulmaz.
    """
    if random.random() < crossover_rate:
        cut = random.randint(1, n_days - 1)
        child = np.hstack([parent1[:, :cut], parent2[:, cut:]])
        # Birleşim sonrası haftalık dinlenme ve sirkadiyen kısıtların denetimi
        if workers is not None and n_workers is not None and shift_reqs is not None:
            from algorithms.penalty_calculator import audit_all_hard_constraints
            is_ok, _, _ = audit_all_hard_constraints(child, workers, n_workers, n_days, shift_reqs)
            if is_ok:
                return child
            return parent1.copy()
        return child
    return parent1.copy()


# =============================================================================
# 4. TAKAS MUTASYONU OPERATÖRÜ (FEASIBLE SWAP MUTATION)
# =============================================================================
def mutate_swap(chromosome, n_workers, n_days, workers, shift_reqs, mutation_rate=0.05):
    """
    Belirli bir olasılıkla (mutation_rate) rastgele bir günde 2 işçinin vardiyalarını
    sert kısıtları bozmayacak şekilde takas ederek genetik çeşitliliği korur.
    """
    if random.random() < mutation_rate:
        d = random.randint(0, n_days - 1)
        w1, w2 = random.sample(range(n_workers), 2)
        if chromosome[w1, d] != chromosome[w2, d]:
            temp_chrom = chromosome.copy()
            temp_chrom[w1, d], temp_chrom[w2, d] = temp_chrom[w2, d], temp_chrom[w1, d]
            # Sert kısıt kontrolü
            if check_swap_feasibility(temp_chrom, workers, d, w1, w2, n_workers, n_days, shift_reqs):
                return temp_chrom
    return chromosome
