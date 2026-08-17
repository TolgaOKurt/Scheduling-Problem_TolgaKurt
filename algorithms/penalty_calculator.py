"""
================================================================================
  ALGORITHMS/PENALTY_CALCULATOR.PY - MERKEZİ CEZA PUANI & KISIT MOTORU
================================================================================
  Bu modül, tüm çizelgeleme algoritmalarının (Greedy, CSP, ILP, Hill Climbing,
  Simulated Annealing, Genetic Algorithm, Memetic Algorithm, Tabu Search)
  yumuşak kısıt ceza puanlarını ve sert kısıt uygunluklarını TEK BİR MERKEZDEN
  (Single Source of Truth) standart, adil ve hatasız hesaplamasını sağlar.
================================================================================
"""

import numpy as np

def calculate_full_penalties(schedule, workers, n_workers, n_days, weights):
    """
    Verilen tam vardiya matrisinin 5 yumuşak kısıt ceza puanını eksiksiz hesaplar.

    Döndürdüğü:
      - penalties (dict): 5 kuralın ayrı ayrı ceza puanları
      - total_score (int): Toplam ceza puanı (Z)
    """
    if weights is None:
        weights = {}

    w_posta = weights.get('posta', weights.get('posta_unity', 15))
    w_circ = weights.get('circadian', 50)
    w_night_imb = weights.get('night_imb', 25)
    w_exp = weights.get('exp_mix', 30)
    w_pref = weights.get('pref_off', 40)

    penalties = {
        'Posta Takım Bütünlüğü İhlali': 0,
        'Sirkadiyen Ritim İhlali (Akşam->Gündüz)': 0,
        'Gece Nöbeti Dengesizliği': 0,
        'Kıdem & MYK Sertifika Eksikliği': 0,
        'Kişisel İzin İhlali': 0
    }

    # 1. Posta Takım Bütünlüğü Kontrolü (A, B, C, D postalarının aynı vardiyada kalması)
    all_postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    for t in range(n_days):
        for p in all_postas:
            p_wids = [w['id'] for w in workers if w.get('posta') == p]
            active_shifts = [schedule[wid, t] for wid in p_wids if schedule[wid, t] != 0]
            if len(active_shifts) > 1:
                counts = [active_shifts.count(s) for s in set(active_shifts)]
                majority = max(counts)
                deviated = len(active_shifts) - majority
                penalties['Posta Takım Bütünlüğü İhlali'] += int(deviated * w_posta)

    # 2. Sirkadiyen Ritim & Gece Nöbeti Dengesi & Kişisel İzin İhlali
    night_counts = np.array([np.sum(schedule[i, :] == 3) for i in range(n_workers)])
    avg_night = np.mean(night_counts) if n_workers > 0 else 0

    for i in range(n_workers):
        # 2a. Sirkadiyen Ritim İhlali (Akşam 2 -> Ertesi gün Gündüz 1)
        for t in range(n_days - 1):
            if schedule[i, t] == 2 and schedule[i, t + 1] == 1:
                penalties['Sirkadiyen Ritim İhlali (Akşam->Gündüz)'] += int(w_circ)

        # 2b. Gece Nöbeti Dengesizliği
        diff = abs(night_counts[i] - avg_night)
        penalties['Gece Nöbeti Dengesizliği'] += int(diff * w_night_imb)

        # 2c. Kişisel İzin İhlali (pref_off: 1..n_days -> indeks: pref_off - 1)
        p_day_idx = workers[i].get('pref_off', 1) - 1
        if 0 <= p_day_idx < n_days and schedule[i, p_day_idx] != 0:
            penalties['Kişisel İzin İhlali'] += int(w_pref)

    # 3. Kıdem & MYK Sertifika Eksikliği (Her aktif vardiyada en az 1 Kıdemli Usta bulunması)
    worker_map = {w['id']: w for w in workers}
    for t in range(n_days):
        for k in (1, 2, 3):
            shift_wids = [w['id'] for w in workers if schedule[w['id'], t] == k]
            ustas = sum(1 for wid in shift_wids if worker_map[wid].get('is_usta', False))
            if len(shift_wids) > 0 and ustas == 0:
                penalties['Kıdem & MYK Sertifika Eksikliği'] += int(w_exp)

    total_score = sum(penalties.values())
    return penalties, total_score


def check_hard_constraints_single_day(schedule, workers, day, n_workers, shift_reqs):
    """
    Belirli bir günde Sert Kısıtların (Vardiya kotaları ve 4 MYK zorunlu sertifikası)
    sağlanıp sağlanmadığını doğrular.
    """
    shift_counts = {1: 0, 2: 0, 3: 0}
    shift_skills = {1: set(), 2: set(), 3: set()}

    for wid in range(n_workers):
        k = schedule[wid, day]
        if k > 0:
            shift_counts[k] += 1
            shift_skills[k].update(workers[wid].get('skills', []))

    # 1. Vardiya kotaları kontrolü
    for k in (1, 2, 3):
        if shift_counts[k] != shift_reqs[k]:
            return False

    # 2. 4 MYK Zorunlu Sertifika Kontrolü
    required_certs = {'Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu'}
    for k in (1, 2, 3):
        if not required_certs.issubset(shift_skills[k]):
            return False

    return True


def check_swap_feasibility(schedule, workers, day, w1_idx, w2_idx, n_workers, n_days, shift_reqs):
    """
    İki işçi arasında day gününde yapılan takas sonrası tüm sert kısıtların
    (Kadro min kotaları, 4 MYK zorunlu sertifikası, 11h dinlenme ve 7 günlük izin)
    sağlanıp sağlanmadığını %100 doğrular.
    """
    if not check_hard_constraints_single_day(schedule, workers, day, n_workers, shift_reqs):
        return False

    s1 = schedule[w1_idx, day]
    s2 = schedule[w2_idx, day]

    # Gece (3) -> Ertesi gün Gündüz (1) dinlenme ihlali kontrolü
    if day > 0 and schedule[w1_idx, day - 1] == 3 and s1 == 1: return False
    if day < n_days - 1 and s1 == 3 and schedule[w1_idx, day + 1] == 1: return False
    if day > 0 and schedule[w2_idx, day - 1] == 3 and s2 == 1: return False
    if day < n_days - 1 and s2 == 3 and schedule[w2_idx, day + 1] == 1: return False

    # Akşam (2) -> Ertesi gün Gündüz (1) dinlenme ihlali kontrolü
    if day > 0 and schedule[w1_idx, day - 1] == 2 and s1 == 1: return False
    if day < n_days - 1 and s1 == 2 and schedule[w1_idx, day + 1] == 1: return False
    if day > 0 and schedule[w2_idx, day - 1] == 2 and s2 == 1: return False
    if day < n_days - 1 and s2 == 2 and schedule[w2_idx, day + 1] == 1: return False

    # 7 günlük kayan pencerede en az 1 gün OFF (dinlenme) kuralı
    for wid in (w1_idx, w2_idx):
        start_tau = max(0, day - 6)
        end_tau = min(max(0, n_days - 7), day)
        for tau in range(start_tau, end_tau + 1):
            if np.sum(schedule[wid, tau:tau + 7] == 0) == 0:
                return False

    return True


def build_worker_request_details(schedule, workers, n_days, weights):
    """
    Tüm çalışanların kişisel izin tercihlerinin (pref_off) karşılanma durumunu
    ve ceza puanı dökümünü standart bir tablo sözlük listesi olarak üretir.
    """
    if weights is None:
        weights = {}
    pref_penalty = weights.get('pref_off', 40)

    request_details = []
    for w in workers:
        wid = w['id']
        pref_day = int(w.get('pref_off', 1))
        p_day_idx = pref_day - 1
        assigned_shift = schedule[wid, p_day_idx] if (0 <= p_day_idx < n_days) else 0
        is_fulfilled = (assigned_shift == 0)

        shift_str = "OFF (İzin)" if is_fulfilled else ("Gündüz" if assigned_shift == 1 else ("Akşam" if assigned_shift == 2 else "Gece"))
        durum_str = "✅ Karşılandı (OFF verildi)" if is_fulfilled else "❌ İhlal Edildi (Vardiyaya Yazıldı)"
        ceza_val = 0 if is_fulfilled else pref_penalty

        request_details.append({
            'id': wid,
            'name': w['name'],
            'posta': w.get('posta', '-'),
            'unvan': "Kıdemli Usta" if w.get('is_usta', False) else "Operatör/İşçi",
            'talep_gun': pref_day,
            'atandi_vardiya': shift_str,
            'durum': durum_str,
            'ceza_puani': ceza_val
        })

    return request_details
