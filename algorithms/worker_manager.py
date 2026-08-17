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

def generate_worker_profiles(n_workers, n_days, randomize=False, seed=42):
    """
    Personel listesi ve yetkinliklerini üretir.
    - Her Posta (A, B, C, D) kendi içinde TAM SERTİFİKA SETİNE (Vinç, Potacı, Döküm, Gaz)
      sahip otonom bir ekip olarak kurulur.
    """
    if randomize:
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
        
        if not randomize:
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

def parse_edited_dataframe_to_workers(df_edited, n_days):
    """st.data_editor tarafından düzenlenen Dataframe'i solver formatına dönüştürür."""
    workers = []
    for idx, row in df_edited.iterrows():
        skills = set()
        if row.get('Vinç Operatörü', False): skills.add('Vinç Operatörü')
        if row.get('Potacı', False): skills.add('Potacı')
        if row.get('Sıcak Metal Döküm Uzmanı', False): skills.add('Sıcak Metal Döküm Uzmanı')
        if row.get('Gaz İzleme Sorumlusu', False): skills.add('Gaz İzleme Sorumlusu')
        
        is_usta = (row.get('Unvan', '') == "Kıdemli Usta")
        if is_usta:
            skills.add('Kıdemli Usta')
            
        pref_off_val = row.get('Talep Edilen İzin (Gün)', row.get('Talep Edilen İzin', 1))
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
            
        workers.append({
            'id': idx,
            'name': str(row.get('İşçi Adı', f"İşçi #{idx+1:02d}")),
            'posta': str(row.get('Posta', 'Posta A')),
            'is_usta': is_usta,
            'skills': skills,
            'pref_off': int(pref_off)
        })
    return workers
