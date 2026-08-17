"""
================================================================================
  ALGORITHMS/SOLVER_CONTRACT.PY - STANDART ÇÖZÜCÜ SÖZLEŞMESİ & KANONİK VERİ MODELİ
================================================================================
  Bu modül, projede yer alan tüm 8 algoritmanın (Greedy, CSP, ILP, HC, SA, GA, MA, TS)
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
            'Kıdem & MYK Sertifika Eksikliği': 0,
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
