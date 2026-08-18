"""
================================================================================
  ALGORITHMS/PSO_SOLVER.PY - DISCRETE PARTICLE SWARM OPTIMIZATION (DPSO)
================================================================================
  Bu modül, Vardiya Çizelgeleme Problemi (NSP) için Kesikli Parçacık Sürü
  Optimizasyonu (Discrete / Swap-Sequence Binary PSO) metasezgisel motorunu içerir.
  
  Temel Parçacık Sürü Zekası İlkeleri (Kennedy & Eberhart 1995, Clerc 2002):
  - Bilişsel Hafıza (Cognitive Memory - p_best): Her parçacığın kişisel en iyi konumu
  - Sosyal Hafıza (Social Memory - g_best): Sürünün kolektif küresel en iyi konumu
  - Hız & Yön Güncellemesi (Velocity Update):
      V_i(t+1) = w * V_i(t) + c1*r1*(p_best_i - X_i) + c2*r2*(g_best - X_i)
  - Konum Güncellemesi (Position Update):
      X_i(t+1) = X_i(t) ⊕ V_i(t+1)
      
  Sert Kısıt Koruması:
  - Yapılan tüm hız ve takas operatörleri check_swap_feasibility ile %100 denetlenir.
================================================================================
"""

import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import get_best_greedy_initial_solution
from algorithms.penalty_calculator import (
    calculate_full_penalties,
    check_swap_feasibility
)
from algorithms.evolutionary_engine import initialize_population
from algorithms.solver_contract import finalize_solver_execution


def _extract_swap_sequence(source_mat: np.ndarray, target_mat: np.ndarray, n_workers: int, n_days: int) -> List[Tuple[int, int, int]]:
    """
    source_mat matrisini target_mat matrisine yaklaştırmak için gereken
    (gün, işçi1, işçi2) takas operatörleri listesini (Fark Dizisi - Difference Vector) çıkarır.
    """
    swaps = []
    temp_mat = source_mat.copy()

    for d in range(n_days):
        col_src = temp_mat[:, d]
        col_tgt = target_mat[:, d]

        diff_indices = [w for w in range(n_workers) if col_src[w] != col_tgt[w]]
        if not diff_indices:
            continue

        for i in range(len(diff_indices)):
            w1 = diff_indices[i]
            target_val = col_tgt[w1]
            if temp_mat[w1, d] == target_val:
                continue

            candidates = [w2 for w2 in diff_indices if temp_mat[w2, d] == target_val and col_tgt[w2] == temp_mat[w1, d]]
            if not candidates:
                candidates = [w2 for w2 in diff_indices if temp_mat[w2, d] == target_val]

            if candidates:
                w2 = candidates[0]
                temp_mat[w1, d], temp_mat[w2, d] = temp_mat[w2, d], temp_mat[w1, d]
                swaps.append((d, w1, w2))

    return swaps


class Particle:
    """Tek bir parçacığın konumunu, hızını ve kişisel hafızasını (p_best) temsil eder."""
    def __init__(self, initial_position: np.ndarray, initial_score: int):
        self.position = initial_position.copy()
        self.current_score = initial_score
        self.pbest_position = initial_position.copy()
        self.pbest_score = initial_score
        self.velocity: List[Tuple[int, int, int]] = []
        self.stagnation_count = 0


def run_particle_swarm_optimization(
    n_workers: int,
    n_days: int,
    r_day: int,
    r_eve: int,
    r_night: int,
    weights: Dict[str, int],
    swarm_size: int = 30,
    max_iterations: int = 100,
    w_inertia: float = 0.72,
    c1_cognitive: float = 1.49,
    c2_social: float = 1.49,
    seed: int = 42,
    custom_workers: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Any] = None,
    stream_interval: int = 5
) -> Dict[str, Any]:
    """
    Vardiya Çizelgeleme Problemi için Kesikli Parçacık Sürü Zekası (Discrete PSO) Çözücüsü.
    
    Parametreler:
    - swarm_size (int): Sürüdeki parçacık sayısı (Varsayılan: 30)
    - max_iterations (int): Maksimum sürü iterasyon sayısı (Varsayılan: 100)
    - w_inertia (float): Atalet ağırlığı (0.4 - 0.9)
    - c1_cognitive (float): Bilişsel çekim katsayısı (p_best çekimi)
    - c2_social (float): Sosyal çekim katsayısı (g_best çekimi)
    """
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)

    shift_reqs = {1: r_day, 2: r_eve, 3: r_night}

    if custom_workers is not None and len(custom_workers) == n_workers:
        workers = custom_workers
    else:
        workers = generate_worker_profiles(n_workers, n_days, randomize=False)

    # 1. En İyi Başlangıç Çözümünü Al (Greedy Seeding)
    greedy_seed = get_best_greedy_initial_solution(
        n_workers, n_days, r_day, r_eve, r_night, weights, custom_workers=workers
    )
    greedy_sched = greedy_seed['schedule']
    workers = greedy_seed['workers']

    # 2. Sürünün İlklendirilmesi (Initial Swarm Generation)
    raw_population = initialize_population(
        greedy_sched, swarm_size, n_workers, n_days, workers, shift_reqs
    )

    eval_count = 0
    particles: List[Particle] = []
    initial_scores: List[int] = []

    for init_sched in raw_population:
        _, score = calculate_full_penalties(init_sched, workers, n_workers, n_days, weights)
        eval_count += 1
        particles.append(Particle(init_sched, score))
        initial_scores.append(score)

    initial_baseline_score = int(np.mean(initial_scores))
    
    # Küresel En İyi (g_best) İlklendirmesi
    best_particle_idx = int(np.argmin(initial_scores))
    gbest_position = particles[best_particle_idx].position.copy()
    gbest_score = particles[best_particle_idx].current_score

    gbest_score_history: List[float] = [float(gbest_score)]
    avg_swarm_score_history: List[float] = [float(initial_baseline_score)]
    diversity_history: List[float] = [float(np.std(initial_scores))]
    velocity_swaps_count = 0
    turbulence_escapes = 0

    if callback:
        callback(0, gbest_score, initial_baseline_score, f"| Sürü: {swarm_size} Parçacık")

    # 3. PSO Sürü Arama Döngüsü (Swarm Optimization Loop)
    for it in range(1, max_iterations + 1):
        # Dinamik Atalet Ağırlığı (Linear Decreasing Inertia Weight)
        w_current = w_inertia - ((w_inertia - 0.4) * (it / max_iterations))

        current_scores = []

        for p_idx, particle in enumerate(particles):
            # Bilişsel ve Sosyal Fark Takas Dizilerini Çıkar
            cognitive_swaps = _extract_swap_sequence(particle.position, particle.pbest_position, n_workers, n_days)
            social_swaps = _extract_swap_sequence(particle.position, gbest_position, n_workers, n_days)

            # Yeni Hız Vektörünü Oluştur (Velocity Update)
            new_velocity: List[Tuple[int, int, int]] = []

            # 1. Atalet Bileşeni (Inertia)
            if particle.velocity and random.random() < w_current:
                sample_len = max(1, int(len(particle.velocity) * w_current))
                new_velocity.extend(random.sample(particle.velocity, min(sample_len, len(particle.velocity))))

            # 2. Bilişsel Çekim (Cognitive: p_best'e doğru)
            r1 = random.random()
            cog_prob = min(1.0, (c1_cognitive * r1) / 2.0)
            for swap in cognitive_swaps:
                if random.random() < cog_prob:
                    new_velocity.append(swap)

            # 3. Sosyal Çekim (Social: g_best'e doğru)
            r2 = random.random()
            soc_prob = min(1.0, (c2_social * r2) / 2.0)
            for swap in social_swaps:
                if random.random() < soc_prob:
                    new_velocity.append(swap)

            # Hız Kırpma (Velocity Clamping)
            max_vel_len = max(5, int(n_days * 3))
            if len(new_velocity) > max_vel_len:
                new_velocity = random.sample(new_velocity, max_vel_len)

            particle.velocity = new_velocity

            # Konum Güncellemesi (Position Update: X_i = X_i ⊕ V_i)
            new_position = particle.position.copy()
            applied_swaps = 0

            for d, w1, w2 in particle.velocity:
                if w1 < n_workers and w2 < n_workers and new_position[w1, d] != new_position[w2, d]:
                    temp_pos = new_position.copy()
                    temp_pos[w1, d], temp_pos[w2, d] = temp_pos[w2, d], temp_pos[w1, d]
                    if check_swap_feasibility(temp_pos, workers, d, w1, w2, n_workers, n_days, shift_reqs):
                        new_position = temp_pos
                        applied_swaps += 1

            velocity_swaps_count += applied_swaps

            # Durgunluktan Kaçış & Türbülans (Stagnation Turbulence / Chaos Mutation)
            if particle.stagnation_count >= 5:
                turbulence_escapes += 1
                particle.stagnation_count = 0
                for _ in range(random.randint(1, 3)):
                    d_rand = random.randint(0, n_days - 1)
                    w_a, w_b = random.sample(range(n_workers), 2)
                    if new_position[w_a, d_rand] != new_position[w_b, d_rand]:
                        temp_pos = new_position.copy()
                        temp_pos[w_a, d_rand], temp_pos[w_b, d_rand] = temp_pos[w_b, d_rand], temp_pos[w_a, d_rand]
                        if check_swap_feasibility(temp_pos, workers, d_rand, w_a, w_b, n_workers, n_days, shift_reqs):
                            new_position = temp_pos

            # Yeni Konumun Ceza Puanını Değerlendir
            _, new_score = calculate_full_penalties(new_position, workers, n_workers, n_days, weights)
            eval_count += 1
            particle.position = new_position
            particle.current_score = new_score
            current_scores.append(new_score)

            # Bilişsel Hafıza Güncellemesi (p_best Update)
            if new_score < particle.pbest_score:
                particle.pbest_score = new_score
                particle.pbest_position = new_position.copy()
                particle.stagnation_count = 0
            else:
                particle.stagnation_count += 1

            # Sosyal Hafıza Güncellemesi (g_best Update)
            if new_score < gbest_score:
                gbest_score = new_score
                gbest_position = new_position.copy()

        # İterasyon İstatistikleri
        mean_score = float(np.mean(current_scores))
        std_score = float(np.std(current_scores))
        gbest_score_history.append(float(gbest_score))
        avg_swarm_score_history.append(mean_score)
        diversity_history.append(std_score)

        # Canlı Akış Bildirimi
        if callback and (it % max(1, stream_interval) == 0 or it == max_iterations):
            callback(it, gbest_score, mean_score, f"| Sürü Ort: {mean_score:.0f} | Çeşitlilik σ: {std_score:.1f}")

    # Nihai Değerlendirme ve Sert Kısıt Denetimi
    term_reason = f"🐝 Belirlenen {max_iterations} sürü iterasyonu tamamlandı. {swarm_size} parçacığın kolektif bilişsel ve sosyal hafıza etkileşimiyle küresel en iyi çözüm (g_best) elde edildi."
    pbest_scores = [p.pbest_score for p in particles]

    meta = {
        'swarm_size': swarm_size,
        'iterations': max_iterations,
        'w_inertia': w_inertia,
        'c1_cognitive': c1_cognitive,
        'c2_social': c2_social,
        'pbest_scores': pbest_scores,
        'gbest_score_history': gbest_score_history,
        'avg_swarm_score_history': avg_swarm_score_history,
        'diversity_history': diversity_history,
        'velocity_swaps_count': velocity_swaps_count,
        'turbulence_escapes': turbulence_escapes,
        'accepted_moves': velocity_swaps_count,
        'total_iterations': max_iterations,
        'seed_source': greedy_seed.get('meta', {}).get('seed_source', 'Greedy'),
        'is_csp_fallback': greedy_seed.get('meta', {}).get('is_csp_fallback', False),
        'fallback_reason': greedy_seed.get('meta', {}).get('fallback_reason', '')
    }

    return finalize_solver_execution(
        best_schedule=gbest_position,
        workers=workers,
        initial_score=initial_baseline_score,
        start_time=start_time,
        eval_count=eval_count,
        total_iterations=max_iterations,
        weights=weights,
        shift_reqs=shift_reqs,
        n_workers=n_workers,
        n_days=n_days,
        termination_reason=term_reason,
        score_history=gbest_score_history,
        meta=meta
    )
