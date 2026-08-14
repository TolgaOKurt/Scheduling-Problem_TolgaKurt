"""
================================================================================
  GLOBAL_STATE.PY - ORTAK YÖNETİM VE GLOBAL MODEL YAPILANDIRMASI
================================================================================
  Bu modül, tüm sekmeler için ortak olan
  Personel Kadrosunu, Sertifikaları, Vardiya İhtiyaçlarını ve Ceza Ağırlıklarını
  Streamlit Sidebar üzerinde merkezi olarak yönetir.
================================================================================
"""

import streamlit as st
import pandas as pd
from algorithms.worker_manager import generate_worker_profiles, parse_edited_dataframe_to_workers

def render_global_sidebar():
    """
    Tüm algoritmalara (Sekme 4, 5, 6, 7) ortak veri sağlayan merkezi Sidebar kontrol paneli.
    """
    st.sidebar.markdown("## ⚙️ Merkezi Model & Kadro Yönetimi")
    st.sidebar.caption("Burada yaptığınız tüm değişiklikler tüm algoritmalar için eş zamanlı olarak geçerli olur.")

    # 1. PERSONEL VE PERİYOT AYARLARI
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👥 1. Personel & Periyot")
    
    n_workers = st.sidebar.slider("Toplam Personel Sayısı (N)", min_value=16, max_value=48, value=24, step=4, key="g_n")
    n_days = st.sidebar.slider("Planlama Periyodu / Gün (D)", min_value=7, max_value=28, value=14, step=7, key="g_d")

    # 2. VARDIYA BAŞI MİNİMUM İHTİYAÇLAR
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏭 2. Vardiya Min Kadro İhtiyaçları")
    
    r_day = st.sidebar.number_input("Gündüz (08-16) Min İhtiyaç", min_value=1, max_value=15, value=5, key="g_rd")
    r_eve = st.sidebar.number_input("Akşam (16-24) Min İhtiyaç", min_value=1, max_value=15, value=5, key="g_re")
    r_night = st.sidebar.number_input("Gece (24-08) Min İhtiyaç", min_value=1, max_value=15, value=5, key="g_rn")

    # 3. AMAÇ FONKSİYONU CEZA AĞIRLIKLARI
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ 3. Ortak Ceza Ağırlıkları")
    
    w_circadian = st.sidebar.slider("Sirkadiyen Ritim Cezası", 10, 100, 50, 10, key="g_wc")
    w_night_imb = st.sidebar.slider("Gece Dengesizlik Cezası", 5, 50, 25, 5, key="g_wn")
    w_exp_mix = st.sidebar.slider("Kıdem (Usta Yoksa) Cezası", 10, 100, 30, 5, key="g_we")
    w_pref_off = st.sidebar.slider("Kişisel İzin İhlal Cezası", 10, 100, 40, 10, key="g_wp")
    w_posta = st.sidebar.slider(
        "Posta Bölünme Cezası (Kişi Başı)",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        key="g_wposta",
        help="Postası bölünen grupta ana topluluktan ayrılan HER İŞÇİ İÇİN kesilecek kişi başı birim ceza puanı (Varsayılan: 15 Puan/Kişi). 1 kişi ayrılırsa 15p, 4 kişi ayrılırsa 60p ceza verilir."
    )

    weights = {
        'circadian': w_circadian,
        'night_imb': w_night_imb,
        'exp_mix': w_exp_mix,
        'pref_off': w_pref_off,
        'posta': w_posta
    }

    # 4. ORTAK KADRO VE SERTİFİKA DÜZENLEYİCİ
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🪪 4. Ortak Personel Kadrosu")
    
    kadro_mode = st.sidebar.radio(
        "Kadro Üretim Modu:",
        ["Düzenli Kadro", "🎲 Rastgele Kadro"],
        help="Düzenli Kadro: İzin talepleri günlere rotasyonlu/dengeli yayılır.\nRastgele Kadro: İzin talepleri, postalar ve sertifikalar TAMAMEN RASTGELE (Total Random) üretilir ve izin yığılmaları oluşur.",
        key="g_km"
    )
    rand_seed = 42
    if "Rastgele" in kadro_mode:
        rand_seed = st.sidebar.number_input("Rastgele Tohum (Seed)", 1, 1000, 42, key="g_seed")

    base_workers = generate_worker_profiles(n_workers, n_days, randomize=("Rastgele" in kadro_mode), seed=rand_seed)

    df_roster_input = pd.DataFrame([
        {
            "İşçi Adı": w['name'],
            "Posta": w['posta'],
            "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi",
            "Vinç Operatörü": "Vinç Operatörü" in w['skills'],
            "Potacı": "Potacı" in w['skills'],
            "Sıcak Metal Döküm Uzmanı": "Sıcak Metal Döküm Uzmanı" in w['skills'],
            "Gaz İzleme Sorumlusu": "Gaz İzleme Sorumlusu" in w['skills'],
            "Talep Edilen İzin": f"Gün {w['pref_off'] + 1}"
        }
        for w in base_workers
    ])

    with st.sidebar.expander("✏️ Kadroyu & Ehliyetleri Elle Düzenle", expanded=False):
        df_edited_roster = st.data_editor(df_roster_input, num_rows="fixed", key="g_editor")
        custom_workers = parse_edited_dataframe_to_workers(df_edited_roster, n_days)

    return {
        'n_workers': n_workers,
        'n_days': n_days,
        'r_day': r_day,
        'r_eve': r_eve,
        'r_night': r_night,
        'weights': weights,
        'custom_workers': custom_workers
    }
