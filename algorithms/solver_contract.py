"""
================================================================================
  ALGORITHMS/SOLVER_CONTRACT.PY - STANDART ÇÖZÜCÜ SÖZLEŞMESİ & KANONİK VERİ MODELİ
================================================================================
  Bu modül, projede yer alan tüm çözücülerin
  ve tüm sekmelerin kesinlikle TEK ve STANDART bir veri formatı (1 Kavram = 1 Anahtar)
  üzerinden haberleşmesini garanti eder.
================================================================================
"""

from typing import List, Dict, Any, Optional
import numpy as np


def build_standard_solver_result(
    schedule: np.ndarray,
    workers: List[Dict[str, Any]],
    is_feasible: bool,
    hard_violations_count: int,
    hard_violation_logs: List[str],
    final_score: int,
    initial_score: Optional[int] = None,
    improvement_rate: float = 0.0,
    exec_time_ms: float = 0.0,
    eval_count: int = 1,
    total_iterations: int = 1,
    termination_reason: str = "",
    penalties: Optional[Dict[str, int]] = None,
    request_details: Optional[List[Dict[str, Any]]] = None,
    score_history: Optional[List[float]] = None,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Tüm optimizasyon çözücülerinin çıktılarını standartlaştıran kanonik fabrika fonksiyonu.
    
    Döndürülen Çekirdek Alanlar (Core Standard Fields):
      - schedule (np.ndarray): (N x D) boyutlu tamsayı vardiya matrisi
      - workers (list[dict]): Çalışan profilleri listesi
      - is_feasible (bool): Sert kısıtların tamamı sağlandı mı?
      - hard_violations_count (int): Toplam sert kısıt ihlali sayısı (0 ise geçerli)
      - hard_violation_logs (list[str]): Sert kısıt ihlallerinin metin logları
      - final_score (int): Minimize edilen nihai ceza skoru (Z)
      - initial_score (int): Başlangıç ceza skoru (Z_0)
      - improvement_rate (float): Yüzdesel iyileşme oranı (%)
      - exec_time_ms (float): Hesaplama süresi (milisaniye)
      - eval_count (int): Ceza fonksiyonu çağrı sayısı
      - total_iterations (int): Toplam iterasyon / nesil / adım sayısı
      - termination_reason (str): Algoritmanın durma gerekçesi
      - penalties (dict[str, int]): 5 alt yumuşak ceza puanı dağılımı
      - request_details (list[dict]): Kişisel izin talepleri dökümü
      - score_history (list[float]): Yakınsama ve skor geçmişi eğrisi
      - meta (dict): Algoritmaya özgü ek meta veriler (mip_gap, backtracks vb.)
    """
    if initial_score is None:
        initial_score = final_score
        
    if penalties is None:
        penalties = {
            'Posta Takım Bütünlüğü İhlali': 0,
            'Sirkadiyen Ritim İhlali (Akşam->Gündüz)': 0,
            'Gece Nöbeti Dengesizliği': 0,
            'Kıdem / Usta Eksikliği': 0,
            'Kişisel İzin İhlali': 0
        }
        
    if request_details is None:
        request_details = []
        
    if score_history is None:
        score_history = [float(final_score)]
        
    if meta is None:
        meta = {}

    return {
        'schedule': schedule,
        'workers': workers,
        'is_feasible': bool(is_feasible),
        'hard_violations_count': int(hard_violations_count),
        'hard_violation_logs': list(hard_violation_logs),
        'final_score': int(final_score),
        'initial_score': int(initial_score),
        'improvement_rate': float(round(improvement_rate, 2)),
        'exec_time_ms': float(round(exec_time_ms, 2)),
        'eval_count': int(eval_count),
        'total_iterations': int(total_iterations),
        'termination_reason': str(termination_reason),
        'penalties': dict(penalties),
        'request_details': list(request_details),
        'score_history': list(score_history),
        'meta': dict(meta)
    }


def finalize_solver_execution(
    best_schedule: np.ndarray,
    workers: List[Dict[str, Any]],
    initial_score: int,
    start_time: float,
    eval_count: int,
    total_iterations: int,
    weights: Dict[str, int],
    shift_reqs: Dict[int, int],
    n_workers: int,
    n_days: int,
    termination_reason: str = "",
    score_history: Optional[List[float]] = None,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Tüm metasezgisel çözücülerin bitişindeki kısıt denetimi, ceza kırılımı,
    izin talepleri dökümü ve sözleşme paketlemesini tek merkezden yürüten yardımcı fonksiyon.
    """
    import time
    from algorithms.penalty_calculator import (
        calculate_full_penalties,
        audit_all_hard_constraints,
        build_worker_request_details
    )

    exec_time_ms = round((time.time() - start_time) * 1000, 2)
    is_feasible, hard_viols_count, hard_violation_logs = audit_all_hard_constraints(
        best_schedule, workers, n_workers, n_days, shift_reqs
    )
    penalties_dict, final_total = calculate_full_penalties(
        best_schedule, workers, n_workers, n_days, weights
    )
    request_details = build_worker_request_details(best_schedule, workers, n_days, weights)
    
    improvement_rate = 0.0
    if initial_score > final_total and initial_score > 0:
        improvement_rate = round(((initial_score - final_total) / initial_score) * 100, 2)

    return build_standard_solver_result(
        schedule=best_schedule,
        workers=workers,
        is_feasible=is_feasible,
        hard_violations_count=hard_viols_count,
        hard_violation_logs=hard_violation_logs,
        final_score=final_total,
        initial_score=initial_score,
        improvement_rate=improvement_rate,
        exec_time_ms=exec_time_ms,
        eval_count=eval_count,
        total_iterations=total_iterations,
        termination_reason=termination_reason,
        penalties=penalties_dict,
        request_details=request_details,
        score_history=score_history,
        meta=meta
    )
