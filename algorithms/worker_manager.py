"""
  ALGORITHMS/WORKER_MANAGER.PY - PERSONEL KADROSU VE SERTİFİKA YÖNETİMİ
================================================================================
  Bu modül, deterministik, rastgele (randomized) veya kullanıcı tarafından
  elle düzenlenmiş (st.data_editor) personel profillerini ve MYK sertifikalarını
  yönetir. Posta takımlarının her birine otonom tam sertifika seti atanarak
  gerçek 4-posta bütünlüğü sağlanır.
"""

import random
import pandas as pd

def generate_worker_profiles(n_workers, n_days, randomize=False, seed=42, mode="Düzenli Kadro"):
    """
    Personel listesi ve yetkinliklerini üretir.
    - Düzenli Kadro: Her Posta (A, B, C, D) kendi içinde TAM SERTİFİKA SETİNE (Vinç, Potacı, Döküm, Gaz)
      sahip otonom bir ekip olarak kurulur.
    - Mükemmel Kadro: Her çalışan 'Kıdemli Usta' unvanına ve 4 kritik MYK ehliyetine eksiksiz sahiptir.
      İzin talepleri 1'den D'ye sırayla (işçi i -> (i % D) + 1) dağıtılır.
    - Rastgele Kadro: Nitelikler, postalar ve izin talepleri rastgele üretilir.
    """
    is_perfect = (mode == "🌟 Mükemmel Kadro" or mode == "Mükemmel Kadro" or (isinstance(mode, str) and "Mükemmel" in mode))
    
    if randomize and not is_perfect:
        random.seed(seed)
        
    workers = []
    postas = ['Posta A', 'Posta B', 'Posta C', 'Posta D']
    all_certs = ['Vinç Operatörü', 'Potacı', 'Sıcak Metal Döküm Uzmanı', 'Gaz İzleme Sorumlusu']
    
    for i in range(n_workers):
        # İşçinin bağlı olduğu Posta (0: Posta A, 1: Posta B, 2: Posta C, 3: Posta D)
        posta_idx = i % 4
        posta_name = postas[posta_idx]
        
        # Posta içindeki sıra indeksi (0, 1, 2, 3, 4, 5...)
        within_posta_idx = i // 4
        
        if is_perfect:
            # MÜKEMMEL KADRO:
            # 1. Herkes Kıdemli Usta
            is_usta = True
            # 2. Herkes 4 kritik sertifikanın tümüne sahip (+ Kıdemli Usta)
            skills = set(all_certs) | {'Kıdemli Usta'}
            # 3. Herkes sırayla izin talep ediyor: İşçi 0 -> Gün 1, İşçi 1 -> Gün 2, ..., İşçi i -> (i % D) + 1
            pref_off = (i % n_days) + 1
        elif not randomize:
            is_usta = (within_posta_idx == 0) # Her postanın ilk elemanı Kıdemli Usta
            skills = set()
            
            # HER POSTA KENDİ İÇİNDE 4 MYK EHLİYETİNİ EKSİKSİZ BARINDIRIR:
            # 0. eleman: Vinç Operatörü (+ Usta ise Kıdemli Usta)
            # 1. eleman: Potacı
            # 2. eleman: Sıcak Metal Döküm Uzmanı
            # 3. eleman: Gaz İzleme Sorumlusu
            # 4. eleman: Vinç Operatörü + Potacı (Yedek)
            # 5. eleman: Döküm Uzmanı + Gaz İzleme (Yedek)
            
            cert_assignment = within_posta_idx % 4
            skills.add(all_certs[cert_assignment])
            
            # Yedek nitelikler ekle
            if within_posta_idx >= 4:
                skills.add(all_certs[(cert_assignment + 1) % 4])
                
            if is_usta:
                skills.add('Kıdemli Usta')
                skills.add(all_certs[(cert_assignment + 2) % 4]) # Usta çift ehliyetli
                
            # Postalar farklı günlerde izin talep etsin (1..n_days)
            pref_off = ((posta_idx * 2 + within_posta_idx) % n_days) + 1
        else:
            is_usta = random.choice([True, False, False])
            skills_count = random.randint(1, 3)
            skills = set(random.sample(all_certs, skills_count))
            if is_usta:
                skills.add('Kıdemli Usta')
            pref_off = random.randint(1, n_days)
            posta_name = random.choice(postas)
            
        workers.append({
            'id': i,
            'name': f"İşçi #{i+1:02d}",
            'posta': posta_name,
            'is_usta': is_usta,
            'skills': skills,
            'pref_off': int(pref_off)
        })
        
    return workers

def normalize_posta_name(raw_val, default_idx=0):
    """Kullanıcının girdiği her türlü posta ifadesini (örn: 'a', 'A', 'posta a', 'POSTA-B', '2', vb.) 'Posta A' formatına standardize eder."""
    if raw_val is None:
        return f"Posta {['A','B','C','D'][default_idx % 4]}"
    s = str(raw_val).strip().upper()
    if not s or s in ("NAN", "NONE"):
        return f"Posta {['A','B','C','D'][default_idx % 4]}"
    
    import re
    cleaned = re.sub(r'POSTA', '', s).strip()
    
    if 'D' in cleaned or s.endswith('D') or '4' in cleaned: return 'Posta D'
    if 'C' in cleaned or s.endswith('C') or '3' in cleaned: return 'Posta C'
    if 'B' in cleaned or s.endswith('B') or '2' in cleaned: return 'Posta B'
    if 'A' in cleaned or s.endswith('A') or '1' in cleaned: return 'Posta A'
    
    return f"Posta {['A','B','C','D'][default_idx % 4]}"


def parse_edited_dataframe_to_workers(df_edited, n_days):
    """st.data_editor tarafından düzenlenen veya CSV/JSON'dan yüklenen Dataframe'i solver formatına dönüştürür."""
    def _is_truthy(val):
        if isinstance(val, bool): return val
        if isinstance(val, (int, float)): return val > 0
        if isinstance(val, str):
            return val.strip().lower() in ['true', '1', 'evet', 'yes', 't', '+', 'var', 'x']
        return False

    workers = []
    for idx, row in df_edited.iterrows():
        skills = set()
        if _is_truthy(row.get('Vinç Operatörü', False)): skills.add('Vinç Operatörü')
        if _is_truthy(row.get('Potacı', False)): skills.add('Potacı')
        if _is_truthy(row.get('Sıcak Metal Döküm Uzmanı', False)): skills.add('Sıcak Metal Döküm Uzmanı')
        if _is_truthy(row.get('Gaz İzleme Sorumlusu', False)): skills.add('Gaz İzleme Sorumlusu')
        
        unvan_str = str(row.get('Unvan', '')).strip().lower()
        is_usta = (unvan_str in ["kıdemli usta", "kidemli usta", "usta", "senior", "master", "true", "1"])
        if is_usta:
            skills.add('Kıdemli Usta')
            
        pref_off_val = row.get('Talep Edilen İzin (Gün)', row.get('Talep Edilen İzin', row.get('pref_off', 1)))
        try:
            import re
            digits = re.findall(r'\d+', str(pref_off_val))
            if digits:
                pref_off = int(digits[0])
            else:
                pref_off = 1
            pref_off = max(1, min(pref_off, n_days))
        except:
            pref_off = 1
            
        posta_raw = row.get('Posta', row.get('posta', None))
        posta_val = normalize_posta_name(posta_raw, idx)

        workers.append({
            'id': idx,
            'name': str(row.get('İşçi Adı', row.get('name', f"İşçi #{idx+1:02d}"))),
            'posta': posta_val,
            'is_usta': is_usta,
            'skills': skills,
            'pref_off': int(pref_off)
        })
    return workers
