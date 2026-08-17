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
from algorithms.penalty_calculator import calculate_full_penalties, build_worker_request_details

class CSPBacktrackingSolver:
    def __init__(self, n_workers, n_days, r_day, r_eve, r_night, max_backtracks=3000, custom_workers=None, weights=None):
        self.n_workers = n_workers
        self.n_days = n_days
        self.shift_reqs = {1: r_day, 2: r_eve, 3: r_night}
        self.max_backtracks = max_backtracks
        self.weights = weights if weights is not None else {'posta': 15, 'circadian': 50, 'pref_off': 40, 'night_imb': 25, 'exp_mix': 30}
        
        self.backtrack_count = 0
        self.nodes_explored = 0
        self.start_time = 0
        self.execution_time = 0
        
        if custom_workers is not None and len(custom_workers) == n_workers:
            self.workers = custom_workers
        else:
            self.workers = generate_worker_profiles(n_workers, n_days, randomize=False)
            
        self.schedule = np.zeros((n_workers, n_days), dtype=int)

    def is_valid_assignment(self, wid, day, shift):
        """Atamanın Sert Kısıtları ihlal edip etmediğini %100 denetler."""
        if shift == 0:
            return True
            
        # Sert Kısıt 3 & 4: Sirkadiyen Ritim & Dinlenme (Gece->Gündüz/Akşam ve Akşam->Gündüz YASAK)
        if day > 0:
            prev_shift = self.schedule[wid, day-1]
            if prev_shift == 3 and shift in [1, 2]: # Gece sonrası Gündüz/Akşam yazılamaz (Yetersiz dinlenme)
                return False
            if prev_shift == 2 and shift == 1: # Akşam sonrası Gündüz yazılamaz (Yetersiz dinlenme)
                return False
            
        # Sert Kısıt 5: Kayan 7 günlük pencerede en az 1 OFF günü olmalı (Max 6 gün üst üste çalışma)
        if day >= 6:
            past_6_offs = sum(1 for tau in range(day-6, day) if self.schedule[wid, tau] == 0)
            if past_6_offs == 0 and shift != 0:
                return False
                
        return True

    def is_group_myk_certified(self, group):
        """Sert Kısıt 2: Vardiyaya atanan grupta 4 zorunlu MYK ehliyetinin tamamının varlığını denetler."""
        req_skills = {'Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu'}
        assigned_skills = set()
        for w in group:
            assigned_skills.update(w['skills'])
        return req_skills.issubset(assigned_skills)

    def solve(self):
        """Backtracking (Geri İzleme) Arama Algoritması."""
        self.start_time = time.time()
        self.backtrack_count = 0
        self.nodes_explored = 0
        
        success = self._backtrack(day=0, shift_idx=1)
        self.execution_time = (time.time() - self.start_time) * 1000 # ms
        
        penalties = self._calculate_soft_penalties()
        request_details = build_worker_request_details(self.schedule, self.workers, self.n_days, self.weights)
        
        return {
            'success': success,
            'schedule': self.schedule,
            'workers': self.workers,
            'backtracks': self.backtrack_count,
            'nodes_explored': self.nodes_explored,
            'exec_time_ms': round(self.execution_time, 2),
            'penalties': penalties,
            'total_penalty': sum(penalties.values()),
            'final_score': sum(penalties.values()),
            'eval_count': 1,
            'request_details': request_details
        }

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

    def _backtrack(self, day, shift_idx):
        if self.backtrack_count >= self.max_backtracks:
            return False
            
        if day >= self.n_days:
            return True
            
        self.nodes_explored += 1
        
        if shift_idx > 3:
            return self._backtrack(day + 1, shift_idx=1)
            
        needed = self.shift_reqs[shift_idx]
        postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
        target_posta = postas[(day + shift_idx - 1) % 4]
        
        groups_to_try = self._generate_candidate_groups(day, shift_idx, needed, target_posta)
        
        if not groups_to_try:
            self.backtrack_count += 1
            return False

        for chosen_group in groups_to_try:
            for w in chosen_group:
                self.schedule[w['id'], day] = shift_idx
                
            if self._backtrack(day, shift_idx + 1):
                return True
                
            # GERİ İZLEME (BACKTRACK)
            self.backtrack_count += 1
            for w in chosen_group:
                self.schedule[w['id'], day] = 0
                
        return False

    def _calculate_soft_penalties(self):
        penalties, _ = calculate_full_penalties(self.schedule, self.workers, self.n_workers, self.n_days, self.weights)
        return penalties
