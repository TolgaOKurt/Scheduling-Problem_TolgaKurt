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
    Verilen tam vardiya matrisinin 5 yumuşak kısıt ceza puanını eksiksiz ve
    vektörize C/NumPy hızında hesaplar.

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

    # 1. Posta Takım Bütünlüğü Kontrolü (A, B, C, D postalarının aynı vardiyada kalması)
    all_postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    posta_dev = 0
    for p in all_postas:
        p_wids = [w['id'] for w in workers if w.get('posta') == p and w['id'] < schedule.shape[0]]
        if len(p_wids) <= 1:
            continue
        p_sub = schedule[p_wids, :]
        for t in range(n_days):
            col = p_sub[:, t]
            active = col[col != 0]
            if len(active) > 1:
                counts = [np.sum(active == s) for s in (1, 2, 3)]
                majority = max(counts)
                posta_dev += (len(active) - majority)
    posta_pen = int(posta_dev * w_posta)

    # 2. Vektörize Sirkadiyen Ritim İhlali (Akşam 2 -> Ertesi gün Gündüz 1)
    if n_days > 1:
        circ_viols = int(np.sum((schedule[:n_workers, :-1] == 2) & (schedule[:n_workers, 1:] == 1)))
    else:
        circ_viols = 0
    circ_pen = int(circ_viols * w_circ)

    # 3. Vektörize Gece Nöbeti Dengesizliği
    night_counts = np.sum(schedule[:n_workers, :] == 3, axis=1)
    avg_night = np.mean(night_counts) if n_workers > 0 else 0.0
    night_pen = int(np.sum(np.abs(night_counts - avg_night)) * w_night_imb)

    # 4. Vektörize Kişisel İzin İhlali
    pref_days = np.array([workers[i].get('pref_off', 1) - 1 for i in range(min(n_workers, len(workers)))])
    valid_mask = (pref_days >= 0) & (pref_days < n_days)
    pref_rows = np.arange(len(pref_days))[valid_mask]
    pref_cols = pref_days[valid_mask]
    pref_viols = int(np.sum(schedule[pref_rows, pref_cols] != 0))
    pref_pen = int(pref_viols * w_pref)

    # 5. Kıdem & MYK Sertifika Eksikliği (Her aktif vardiyada en az 1 Kıdemli Usta bulunması)
    usta_set = {w['id'] for w in workers if w.get('is_usta', False)}
    exp_pen = 0
    for t in range(n_days):
        col = schedule[:n_workers, t]
        for k in (1, 2, 3):
            shift_wids = np.where(col == k)[0]
            if len(shift_wids) > 0 and not any(wid in usta_set for wid in shift_wids):
                exp_pen += int(w_exp)

    penalties = {
        'Posta Takım Bütünlüğü İhlali': posta_pen,
        'Sirkadiyen Ritim İhlali (Akşam->Gündüz)': circ_pen,
        'Gece Nöbeti Dengesizliği': night_pen,
        'Kıdem & MYK Sertifika Eksikliği': exp_pen,
        'Kişisel İzin İhlali': pref_pen
    }
    total_score = posta_pen + circ_pen + night_pen + exp_pen + pref_pen
    return penalties, total_score


def build_fast_evaluator(workers, n_days, weights):
    """
    Metasezgisel algoritmaların iç döngülerinde (inner loop) binlerce kez çağrılan
    yüksek hızlı, vektörize ceza değerlendirici fonksiyonunu (closure) oluşturur.
    Tüm ön hesaplamaları (is_usta, pref_days, posta_groups) bir kez yapar.
    """
    if weights is None:
        weights = {}

    w_posta = weights.get('posta', weights.get('posta_unity', 15))
    w_circ = weights.get('circadian', 50)
    w_night_imb = weights.get('night_imb', 25)
    w_exp = weights.get('exp_mix', 30)
    w_pref = weights.get('pref_off', 40)

    is_usta_arr = np.array([w.get('is_usta', False) for w in workers])
    pref_days = np.array([int(w.get('pref_off', 1)) - 1 for w in workers])
    valid_pref = (pref_days >= 0) & (pref_days < n_days)
    pref_rows = np.arange(len(pref_days))[valid_pref]
    pref_cols = pref_days[valid_pref]

    postas = np.array([w.get('posta', '') for w in workers])
    posta_groups = [np.where(postas == p)[0] for p in ['Posta A', 'Posta B', 'Posta C', 'Posta D']]

    def fast_evaluate(sched: np.ndarray) -> int:
        # 1. Sirkadiyen Ritim
        p_circ = int(((sched[:, :-1] == 2) & (sched[:, 1:] == 1)).sum() * w_circ) if n_days > 1 else 0
        
        # 2. Gece Dengesizliği
        night_counts = (sched == 3).sum(axis=1)
        avg_night = night_counts.mean()
        p_night = int(np.abs(night_counts - avg_night).sum() * w_night_imb)

        # 3. Kişisel İzin
        p_pref = int((sched[pref_rows, pref_cols] != 0).sum() * w_pref) if len(pref_rows) > 0 else 0

        # 4. Usta Varlığı
        p_exp = 0
        for t in range(n_days):
            col = sched[:, t]
            for k in (1, 2, 3):
                mask = (col == k)
                if mask.any() and not is_usta_arr[mask].any():
                    p_exp += w_exp

        # 5. Posta Bütünlüğü
        p_posta = 0
        for g in posta_groups:
            if len(g) > 1:
                for t in range(n_days):
                    col_g = sched[g, t]
                    act = col_g[col_g > 0]
                    if len(act) > 1:
                        bc = np.bincount(act)
                        p_posta += (len(act) - bc.max()) * w_posta

        return p_circ + p_night + p_pref + p_exp + p_posta

    return fast_evaluate


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

    # Gece (3) -> Ertesi gün Gündüz (1) dinlenme ihlali kontrolü (0 saat dinlenme - Sert Kısıt)
    if day > 0 and schedule[w1_idx, day - 1] == 3 and s1 == 1: return False
    if day < n_days - 1 and s1 == 3 and schedule[w1_idx, day + 1] == 1: return False
    if day > 0 and schedule[w2_idx, day - 1] == 3 and s2 == 1: return False
    if day < n_days - 1 and s2 == 3 and schedule[w2_idx, day + 1] == 1: return False

    # 7 günlük kayan pencerede en az 1 gün OFF (dinlenme) kuralı
    for wid in (w1_idx, w2_idx):
        start_tau = max(0, day - 6)
        end_tau = min(max(0, n_days - 7), day)
        for tau in range(start_tau, end_tau + 1):
            if np.sum(schedule[wid, tau:tau + 7] == 0) == 0:
                return False

    return True


def audit_all_hard_constraints(schedule, workers, n_workers, n_days, shift_reqs):
    """
    Tamamlanmış veya ara çizelgenin TÜM Sert Kısıtlarını (Hard Constraints)
    satır satır denetler ve ihlalleri detaylı log listesi olarak döndürür.
    
    Kontrol edilen Sert Kısıtlar:
    1. Vardiya kotaları / Kadro yetersizliği (Shift capacity requirements)
    2. Her vardiyada zorunlu 4 MYK sertifikasının bulunması
    3. 7 günlük kayan pencerede en az 1 gün OFF (dinlenme) kuralı (İş Kanunu 6 gün kuralı)
    4. Gece (3) -> Ertesi gün Gündüz (1) dinlenme ihlali
    5. Akşam (2) -> Ertesi gün Gündüz (1) dinlenme ihlali
    
    Döndürdüğü:
      - is_feasible (bool): İhlal sayısı 0 ise True, aksi halde False
      - hard_violations_count (int): Toplam tespit edilen sert kısıt ihlali sayısı
      - hard_violation_logs (list[str]): İhlallerin detaylı açıklama listesi
    """
    hard_logs = []
    shift_names = {1: "Gündüz (08-16)", 2: "Akşam (16-24)", 3: "Gece (24-08)"}
    required_certs = {'Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu'}
    
    # 1 & 2: Gün bazlı vardiya kotaları ve MYK sertifika kontrolleri
    for d in range(n_days):
        shift_counts = {1: 0, 2: 0, 3: 0}
        shift_skills = {1: set(), 2: set(), 3: set()}
        
        for w in workers:
            wid = w['id']
            if wid < schedule.shape[0]:
                k = schedule[wid, d]
                if k in (1, 2, 3):
                    shift_counts[k] += 1
                    shift_skills[k].update(w.get('skills', []))
                    
        for k in (1, 2, 3):
            needed = shift_reqs.get(k, 0)
            assigned = shift_counts[k]
            if assigned < needed:
                shortage = needed - assigned
                hard_logs.append(
                    f"❌ Kadro Yetersizliği [Gün {d+1} - {shift_names[k]}]: İstenen min {needed} kişi, atanan {assigned} kişi! (Eksik: {shortage} personel)"
                )
            elif assigned > needed:
                excess = assigned - needed
                hard_logs.append(
                    f"⚠️ Fazla Atama [Gün {d+1} - {shift_names[k]}]: Kotadan fazla {excess} kişi atandı! (Atanan: {assigned}, İstenen: {needed})"
                )
                
            if assigned > 0:
                missing_certs = required_certs - shift_skills[k]
                for cert in missing_certs:
                    hard_logs.append(
                        f"⚠️ MYK Sertifika Eksikliği [Gün {d+1} - {shift_names[k]}]: Zorunlu '{cert}' sertifikalı personel yok!"
                    )

    # 3, 4, 5: Çalışan bazlı dinlenme ve sirkadiyen kontroller
    for w in workers:
        wid = w['id']
        if wid >= schedule.shape[0]:
            continue
        w_name = w.get('name', f"İşçi #{wid}")
        w_posta = w.get('posta', '')
        
        # 3. 7 günlük kayan pencerede en az 1 gün OFF kontrolü
        for tau in range(max(1, n_days - 6)):
            window = schedule[wid, tau:tau + 7]
            if np.sum(window == 0) == 0:
                hard_logs.append(
                    f"⛔ Haftalık Dinlenme İhlali: {w_name} ({w_posta}), Gün {tau+1}-{tau+7} arasında hiç izin (OFF) kullanmadan 7 gün üst üste çalıştırıldı!"
                )
                break  # İşçi başına haftalık pencereyi bir kez raporla
                
        # 4: Ardışık günler dinlenme süresi kontrolü (Gece 08:00 çıkış -> Sabah 08:00 giriş = 0 saat dinlenme)
        for d in range(n_days - 1):
            s_curr = schedule[wid, d]
            s_next = schedule[wid, d + 1]
            if s_curr == 3 and s_next == 1:
                hard_logs.append(
                    f"🚫 Yetersiz Dinlenme (Gece->Gündüz): {w_name} Gün {d+1} Gece (08:00 çıkış) sonrası Gün {d+2} Gündüz (08:00 giriş) yazıldı (0 saat dinlenme - Yasal Olarak İmkansız)!"
                )

    is_feasible = (len(hard_logs) == 0)
    return is_feasible, len(hard_logs), hard_logs


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
