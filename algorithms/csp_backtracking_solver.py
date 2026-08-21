"""
================================================================================
  ALGORITHMS/CSP_BACKTRACKING_SOLVER.PY - BACKTRACKING / CSP SOLVER
================================================================================
  Bu modül, Yöneylem Araştırması & Yapay Zeka alanındaki Kısıt Tatmin Problemleri
  (Constraint Satisfaction Problem - CSP) ve Geri İzleme (Backtracking) arama
  algoritmasını içerir. Sert Kısıtlar %100 doğrulanırken, 4-posta takım bütünlüğü ve
  yumuşak kısıt sapmaları hesaplanıp raporlanır.
================================================================================
"""

import time
import numpy as np
import pandas as pd
from algorithms.worker_manager import generate_worker_profiles
from algorithms.penalty_calculator import calculate_full_penalties, build_worker_request_details, audit_all_hard_constraints
from algorithms.solver_contract import build_standard_solver_result

class CSPBacktrackingSolver:
    def __init__(self, n_workers, n_days, r_day, r_eve, r_night, max_backtracks=3000, custom_workers=None, weights=None):
        self.n_workers = n_workers
        self.n_days = n_days
        self.shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
        self.max_backtracks = max_backtracks
        self.weights = weights if weights is not None else {'posta': 15, 'circadian': 50, 'pref_off': 40, 'night_imb': 25, 'exp_mix': 60}
        
        self.backtrack_count = 0
        self.nodes_explored = 0
        self.start_time = 0
        self.execution_time = 0
        
        if custom_workers is not None and len(custom_workers) == n_workers:
            self.workers = custom_workers
        else:
            self.workers = generate_worker_profiles(n_workers, n_days, randomize=False)
            
        self.schedule = np.zeros((n_workers, n_days), dtype=int)

    # =========================================================================
    # 1. SERT KISIT UYGUNLUK DENETİMLERİ (HARD CONSTRAINT VALIDATION)
    # =========================================================================
    def is_valid_assignment(self, wid, day, shift):
        """Atamanın Sert Kısıtları ihlal edip etmediğini %100 denetler."""
        if shift == 0:
            return True
            
        # Sert Kısıt: Vardiyalar Arası Dinlenme (Gece çıkışı sabah girişi = 0 saat dinlenme YASAKTIR)
        if day > 0:
            prev_shift = self.schedule[wid, day-1]
            if prev_shift == 3 and shift == 1: # Gece sonrası Gündüz yazılamaz (Sert Kısıt)
                return False
            
        # Sert Kısıt: Kayan 7 günlük pencerede en az 1 OFF günü olmalı (Max 6 gün üst üste çalışma)
        if day >= 6:
            past_6_offs = sum(1 for tau in range(day-6, day) if self.schedule[wid, tau] == 0)
            if past_6_offs == 0 and shift != 0:
                return False
                
        return True

    def is_group_myk_certified(self, group):
        """Sert Kısıt: Vardiyaya atanan grupta 4 zorunlu MYK ehliyetinin tamamının varlığını denetler."""
        req_skills = {'Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu'}
        assigned_skills = set()
        for w in group:
            assigned_skills.update(w['skills'])
        return req_skills.issubset(assigned_skills)

    # =========================================================================
    # 2. ÇÖZÜCÜ ÇALIŞTIRMA VE SONUÇ DERLEME
    # =========================================================================
    def solve(self, callback=None, stream_interval=50):
        """Backtracking (Geri İzleme) Arama Algoritması."""
        self.start_time = time.time()
        self.backtrack_count = 0
        self.nodes_explored = 0
        
        if callback:
            callback(0, 0, 0, "| Başlangıç")
            
        success = self._backtrack(day=0, shift_idx=1, callback=callback, stream_interval=stream_interval)
        self.execution_time = (time.time() - self.start_time) * 1000 # ms
        
        if callback:
            callback(self.backtrack_count, 0, 0, "| Bitti")
        
        is_feas, hard_viols_cnt, hard_logs = audit_all_hard_constraints(
            self.schedule, self.workers, self.n_workers, self.n_days, self.shift_reqs
        )
        
        if success and is_feas:
            penalties = self._calculate_soft_penalties()
            final_score = sum(penalties.values())
            term_reason = "Tüm sert kısıtlar sağlanarak geçerli çizelge bulundu."
        elif self.backtrack_count >= self.max_backtracks:
            penalties = self._calculate_soft_penalties()
            final_score = 99999
            term_reason = f"Maksimum geri izleme limitine ({self.max_backtracks:,}) ulaşıldı."
        else:
            penalties = self._calculate_soft_penalties()
            final_score = 99999
            term_reason = "Arama ağacı tarandı ancak geçerli bir kombinasyon bulunamadı."
            
        request_details = build_worker_request_details(self.schedule, self.workers, self.n_days, self.weights)
        
        return build_standard_solver_result(
            schedule=self.schedule,
            workers=self.workers,
            is_feasible=bool(success and is_feas),
            hard_violations_count=hard_viols_cnt,
            hard_violation_logs=hard_logs,
            final_score=final_score,
            initial_score=final_score,
            improvement_rate=0.0,
            exec_time_ms=self.execution_time,
            eval_count=1,
            total_iterations=self.backtrack_count,
            termination_reason=term_reason,
            penalties=penalties,
            request_details=request_details,
            score_history=[float(final_score)],
            meta={
                'backtracks': self.backtrack_count,
                'nodes_explored': self.nodes_explored
            }
        )

    # =========================================================================
    # 3. ADAY GRUP DALLANMA VE SIRALAMA STRATEJİLERİ (HEURISTIC BRANCHING)
    # =========================================================================
    def _generate_candidate_groups(self, day, shift_idx, needed, target_posta):
        """Çeşitlemeli aday grupları türetir (Backtracking arama ağacı dallanması için)."""
        valid_candidates = [
            w for w in self.workers 
            if self.schedule[w['id'], day] == 0 and self.is_valid_assignment(w['id'], day, shift_idx)
        ]
        
        if len(valid_candidates) < needed:
            return []
            
        groups = []
        
        # Strateji 1: Posta Bütünlüğü Odaklı Aday Grubu
        c1 = sorted(valid_candidates, key=lambda x: (
            0 if x['posta'] == target_posta else 1,
            np.sum(self.schedule[x['id'], :] > 0)
        ))
        g1 = self._assemble_certified_group(c1, needed)
        if g1 and self.is_group_myk_certified(g1):
            groups.append(g1)
            
        # Strateji 2: Nöbet Adaleti Odaklı Aday Grubu (En az çalışan işçiler)
        c2 = sorted(valid_candidates, key=lambda x: (
            np.sum(self.schedule[x['id'], :] > 0),
            0 if x['posta'] == target_posta else 1
        ))
        g2 = self._assemble_certified_group(c2, needed)
        if g2 and self.is_group_myk_certified(g2):
            ids_g2 = set(w['id'] for w in g2)
            if not any(set(w['id'] for w in g) == ids_g2 for g in groups):
                groups.append(g2)

        # Strateji 3: İzin Sonrası Dinlenmiş Personel Odaklı Aday Grubu
        c3 = sorted(valid_candidates, key=lambda x: (
            0 if (day > 0 and self.schedule[x['id'], day-1] == 0) else 1,
            0 if x['posta'] == target_posta else 1
        ))
        g3 = self._assemble_certified_group(c3, needed)
        if g3 and self.is_group_myk_certified(g3):
            ids_g3 = set(w['id'] for w in g3)
            if not any(set(w['id'] for w in g) == ids_g3 for g in groups):
                groups.append(g3)

        return groups

    def _assemble_certified_group(self, candidates, needed):
        req_cert_list = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
        group = []
        
        for cert in req_cert_list:
            if len(group) >= needed:
                break
            if not any(cert in w['skills'] for w in group):
                for cand in candidates:
                    if cand not in group and cert in cand['skills']:
                        group.append(cand)
                        break
                        
        for cand in candidates:
            if len(group) >= needed:
                break
            if cand not in group:
                group.append(cand)
                
        return group if len(group) == needed else None

    # =========================================================================
    # 4. ÖZYİNELEMELİ GERİ İZLEME MOTORU (RECURSIVE BACKTRACKING & PRUNING)
    # =========================================================================
    def _backtrack(self, day, shift_idx, callback=None, stream_interval=50):
        # Durdurma Kriteri 1: Maksimum Geri İzleme Sınırı
        if self.backtrack_count >= self.max_backtracks:
            return False
            
        # Başarı Kriteri: Tüm günler başarıyla doldurulduğunda arama biter
        if day >= self.n_days:
            return True
            
        self.nodes_explored += 1
        
        # Gün tamamlandığında bir sonraki güne geç
        if shift_idx > 3:
            return self._backtrack(day + 1, shift_idx=1, callback=callback, stream_interval=stream_interval)
            
        needed = self.shift_reqs[shift_idx]
        postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
        target_posta = postas[(day + shift_idx - 1) % 4]
        
        # Bu vardiya için uygun aday grupları dallandır
        groups_to_try = self._generate_candidate_groups(day, shift_idx, needed, target_posta)
        
        # Hiçbir geçerli grup bulunamazsa bu dal budanır (Dead-end Pruning)
        if not groups_to_try:
            self.backtrack_count += 1
            if callback and self.backtrack_count % stream_interval == 0:
                callback(self.backtrack_count, 0, 0, f"| Gün: {day+1}/{self.n_days}")
            return False

        # Dalları tek tek dene
        for chosen_group in groups_to_try:
            # 1. Hamleyi Yap (İleri Atama / Forward Assignment)
            for w in chosen_group:
                self.schedule[w['id'], day] = shift_idx
                
            # 2. Özyineleme (Derinlemesine İlerle / DFS)
            if self._backtrack(day, shift_idx + 1, callback=callback, stream_interval=stream_interval):
                return True
                
            # 3. GERİ İZLEME (BACKTRACK / ROLLBACK): İleri dal başarısız olduysa hamleyi geri al
            self.backtrack_count += 1
            if callback and self.backtrack_count % stream_interval == 0:
                callback(self.backtrack_count, 0, 0, f"| Gün: {day+1}/{self.n_days}")
            for w in chosen_group:
                self.schedule[w['id'], day] = 0
                
        return False

    def _calculate_soft_penalties(self):
        penalties, _ = calculate_full_penalties(self.schedule, self.workers, self.n_workers, self.n_days, self.weights)
        return penalties
