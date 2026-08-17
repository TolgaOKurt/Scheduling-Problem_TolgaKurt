"""
================================================================================
  GLOBAL_STATE.PY - ORTAK YÖNETİM VE GLOBAL MODEL YAPILANDIRMASI
================================================================================
  Bu modül, tüm sekmeler için ortak olan
  Personel Kadrosunu, Sertifikaları, Vardiya İhtiyaçlarını, Ceza Ağırlıklarını
  ve Canlı Akış (Live Streaming) parametrelerini merkezi bir sekme üzerinden
  kalıcı (persistent) olarak yönetir.
================================================================================
"""

import streamlit as st
import pandas as pd
from algorithms.worker_manager import generate_worker_profiles, parse_edited_dataframe_to_workers

# ================================================================================
# MERKEZİ MODEL VARSAYILANLARI (Single Source of Truth)
# ================================================================================
GLOBAL_DEFAULTS = {
    # 1. Personel & Periyot
    'n_workers': 28,
    'n_days': 7,

    # 2. Vardiya Başı Minimum Kadro İhtiyaçları
    'r_day': 9,
    'r_eve': 7,
    'r_night': 5,

    # 3. Amaç Fonksiyonu Ceza Ağırlıkları
    'penalties': {
        'circadian': 50,
        'night_imb': 25,
        'exp_mix': 30,
        'pref_off': 40,
        'posta': 15
    },

    # 4. Kadro Üretim Modu ve Rastgelelik Tohumu
    'kadro_mode': "Düzenli Kadro",
    'rand_seed': 42,

    # 5. Canlı Optimizasyon Akışı & İzleme Ayarları
    'live_stream_enabled': True,
    'live_stream_interval': 50
}

DEFAULT_PENALTIES = GLOBAL_DEFAULTS['penalties']


def _init_persistent_config():
    """Streamlit sekme geçişlerinde silinmeyen kalıcı durum deposunu başlatır."""
    if "persistent_global_config" not in st.session_state:
        init_workers = generate_worker_profiles(
            GLOBAL_DEFAULTS['n_workers'],
            GLOBAL_DEFAULTS['n_days'],
            randomize=(GLOBAL_DEFAULTS['kadro_mode'] != "Düzenli Kadro"),
            seed=GLOBAL_DEFAULTS['rand_seed']
        )
        st.session_state["persistent_global_config"] = {
            'n_workers': GLOBAL_DEFAULTS['n_workers'],
            'n_days': GLOBAL_DEFAULTS['n_days'],
            'r_day': GLOBAL_DEFAULTS['r_day'],
            'r_eve': GLOBAL_DEFAULTS['r_eve'],
            'r_night': GLOBAL_DEFAULTS['r_night'],
            'circadian': DEFAULT_PENALTIES['circadian'],
            'night_imb': DEFAULT_PENALTIES['night_imb'],
            'exp_mix': DEFAULT_PENALTIES['exp_mix'],
            'pref_off': DEFAULT_PENALTIES['pref_off'],
            'posta': DEFAULT_PENALTIES['posta'],
            'kadro_mode': GLOBAL_DEFAULTS['kadro_mode'],
            'rand_seed': GLOBAL_DEFAULTS['rand_seed'],
            'custom_workers': init_workers,
            '_last_kadro_sig': (
                GLOBAL_DEFAULTS['n_workers'],
                GLOBAL_DEFAULTS['n_days'],
                GLOBAL_DEFAULTS['kadro_mode'],
                GLOBAL_DEFAULTS['rand_seed']
            ),
            'editor_version': 0,
            'live_stream_enabled': GLOBAL_DEFAULTS['live_stream_enabled'],
            'live_stream_interval': GLOBAL_DEFAULTS['live_stream_interval']
        }


def _sync_state_from_ui():
    """Streamlit arayüz widget'larında değişen en güncel değerleri kalıcı depoya anında senkronize eder."""
    _init_persistent_config()
    cfg = st.session_state["persistent_global_config"]
    mapping = {
        'n_workers': 'ui_cfg_n',
        'n_days': 'ui_cfg_d',
        'r_day': 'ui_cfg_rd',
        'r_eve': 'ui_cfg_re',
        'r_night': 'ui_cfg_rn',
        'circadian': 'ui_cfg_wc',
        'night_imb': 'ui_cfg_wn',
        'exp_mix': 'ui_cfg_we',
        'pref_off': 'ui_cfg_wp',
        'posta': 'ui_cfg_wposta',
        'kadro_mode': 'ui_cfg_km',
        'rand_seed': 'ui_cfg_seed',
        'live_stream_enabled': 'ui_cfg_live_enabled',
        'live_stream_interval': 'ui_cfg_live_interval'
    }
    for cfg_key, ui_key in mapping.items():
        if ui_key in st.session_state:
            cfg[cfg_key] = st.session_state[ui_key]


def get_global_params():
    """Tüm sekmelerin ve çözücülerin ortak kullandığı parametre setini döndürür."""
    _sync_state_from_ui()
    cfg = st.session_state["persistent_global_config"]

    n_workers = cfg['n_workers']
    n_days = cfg['n_days']
    r_day = cfg['r_day']
    r_eve = cfg['r_eve']
    r_night = cfg['r_night']
    kadro_mode = cfg['kadro_mode']
    rand_seed = cfg['rand_seed']

    weights = {
        'circadian': cfg['circadian'],
        'night_imb': cfg['night_imb'],
        'exp_mix': cfg['exp_mix'],
        'pref_off': cfg['pref_off'],
        'posta': cfg['posta']
    }

    call_cost_ms = round(0.001492 * (n_workers * n_days) + 0.1670, 3)

    # Kadro boyutu veya mod/tohum değişimi doğrulaması
    curr_sig = (n_workers, n_days, kadro_mode, rand_seed)
    last_sig = cfg.get('_last_kadro_sig')
    custom_workers = cfg.get('custom_workers')

    if curr_sig != last_sig or custom_workers is None or len(custom_workers) != n_workers:
        custom_workers = generate_worker_profiles(
            n_workers,
            n_days,
            randomize=("Rastgele" in kadro_mode),
            seed=rand_seed
        )
        cfg['custom_workers'] = custom_workers
        cfg['_last_kadro_sig'] = curr_sig
        cfg['editor_version'] = cfg.get('editor_version', 0) + 1

    return {
        'n_workers': n_workers,
        'n_days': n_days,
        'r_day': r_day,
        'r_eve': r_eve,
        'r_night': r_night,
        'weights': weights,
        'custom_workers': custom_workers,
        'call_cost_ms': call_cost_ms,
        'live_stream_enabled': cfg.get('live_stream_enabled', True),
        'live_stream_interval': cfg.get('live_stream_interval', 50)
    }


def render_global_config_tab():
    """Merkezi Model & Kadro Yönetimi sekmesini çizer (Geniş ekran görünümü)."""
    _sync_state_from_ui()
    cfg = st.session_state["persistent_global_config"]

    st.markdown("## ⚙️ Merkezi Model Yapılandırması & Kadro Yönetimi")
    st.markdown("""
    <div style="background-color: #eff6ff; border-left: 5px solid #2563eb; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; color: #1e3a8a;">
        📌 <b>Global Parametre Merkezi:</b> Bu sayfada belirlediğiniz <b>Personel Sayısı, Vardiya Gereksinimleri, Ceza Katsayıları, Canlı Akış Sıklığı ve Kadro Yetkinlikleri</b> tüm optimizasyon sekmelerine (Sekme 4 - 11 ve Benchmark) anında ve kalıcı olarak aktarılır.
    </div>
    """, unsafe_allow_html=True)

    # 1. SATIR: ÖZET METRİK KARTLARI
    total_daily_req = cfg['r_day'] + cfg['r_eve'] + cfg['r_night']
    call_cost_ms = round(0.001492 * (cfg['n_workers'] * cfg['n_days']) + 0.1670, 3)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam Personel (N)</div>
        <div class="metric-value" style="color: #2563eb;">{cfg['n_workers']} İşçi</div>
        </div>""", unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Planlama Periyodu (D)</div>
        <div class="metric-value" style="color: #059669;">{cfg['n_days']} Gün</div>
        </div>""", unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Günlük Min Kadro İhtiyacı</div>
        <div class="metric-value" style="color: #d97706;">{total_daily_req} Kişi/Gün</div>
        </div>""", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Çağrı Başı Ortalama Maliyet</div>
        <div class="metric-value" style="color: #6366f1;">{call_cost_ms:.3f} ms</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # 2. SATIR: 3 KOLONLU PARAMETRE DÜZENLEME PANELİ
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div class="card-box" style="border-top: 4px solid #2563eb; padding: 16px;">
        <h4 style="color: #1d4ed8; margin-top: 0;">👥 1. Personel & Periyot</h4>
        </div>""", unsafe_allow_html=True)
        
        n_workers_in = st.slider("Toplam Personel Sayısı (N)", min_value=16, max_value=60, value=cfg['n_workers'], step=4, key="ui_cfg_n")
        n_days_in = st.slider("Planlama Periyodu / Gün (D)", min_value=7, max_value=28, value=cfg['n_days'], step=7, key="ui_cfg_d")
        
        st.caption(f"📐 **Matris Boyutu:** {n_workers_in} × {n_days_in} = {n_workers_in * n_days_in} hücre")
        
        max_weekly_capacity = n_workers_in * 6
        needed_weekly_shifts = total_daily_req * 7
        if needed_weekly_shifts > max_weekly_capacity:
            st.warning(f"⚠️ **Kadro Yetersizliği Riski:** Haftalık {needed_weekly_shifts} vardiya ihtiyacına karşılık maksimum kapasite {max_weekly_capacity} vardiyadır.")

    with col2:
        st.markdown("""<div class="card-box" style="border-top: 4px solid #059669; padding: 16px;">
        <h4 style="color: #047857; margin-top: 0;">🏭 2. Vardiya Min İhtiyaçları</h4>
        </div>""", unsafe_allow_html=True)
        
        r_day_in = st.number_input("Gündüz (08-16) Min İhtiyaç", min_value=1, max_value=15, value=cfg['r_day'], key="ui_cfg_rd")
        r_eve_in = st.number_input("Akşam (16-24) Min İhtiyaç", min_value=1, max_value=15, value=cfg['r_eve'], key="ui_cfg_re")
        r_night_in = st.number_input("Gece (24-08) Min İhtiyaç", min_value=1, max_value=15, value=cfg['r_night'], key="ui_cfg_rn")

    with col3:
        st.markdown("""<div class="card-box" style="border-top: 4px solid #d97706; padding: 16px;">
        <h4 style="color: #b45309; margin-top: 0;">⚖️ 3. Ortak Ceza Ağırlıkları</h4>
        </div>""", unsafe_allow_html=True)
        
        def reset_penalty_weights():
            _init_persistent_config()
            cfg_target = st.session_state["persistent_global_config"]
            for k in ['circadian', 'night_imb', 'exp_mix', 'pref_off', 'posta']:
                cfg_target[k] = DEFAULT_PENALTIES[k]
            st.session_state["ui_cfg_wc"] = DEFAULT_PENALTIES['circadian']
            st.session_state["ui_cfg_wn"] = DEFAULT_PENALTIES['night_imb']
            st.session_state["ui_cfg_we"] = DEFAULT_PENALTIES['exp_mix']
            st.session_state["ui_cfg_wp"] = DEFAULT_PENALTIES['pref_off']
            st.session_state["ui_cfg_wposta"] = DEFAULT_PENALTIES['posta']

        st.button("🔄 Cezaları Varsayılana Sıfırla", on_click=reset_penalty_weights, use_container_width=True, key="btn_reset_penalties")

        w_circadian_in = st.slider("Sirkadiyen Ritim Cezası", 10, 100, cfg['circadian'], 10, key="ui_cfg_wc")
        w_night_imb_in = st.slider("Gece Dengesizlik Cezası", 5, 50, cfg['night_imb'], 5, key="ui_cfg_wn")
        w_exp_mix_in = st.slider("Kıdem (Usta Yoksa) Cezası", 10, 100, cfg['exp_mix'], 5, key="ui_cfg_we")
        w_pref_off_in = st.slider("Kişisel İzin İhlal Cezası", 10, 100, cfg['pref_off'], 10, key="ui_cfg_wp")
        w_posta_in = st.slider(
            "Posta Bölünme Cezası (Kişi Başı)",
            min_value=5,
            max_value=50,
            value=cfg['posta'],
            step=5,
            key="ui_cfg_wposta",
            help=f"Postası bölünen grupta ana topluluktan ayrılan HER İŞÇİ İÇİN kesilecek kişi başı birim ceza puanı (Varsayılan: {DEFAULT_PENALTIES['posta']} Puan/Kişi)."
        )

    # Parametre güncelleme
    cfg['n_workers'] = n_workers_in
    cfg['n_days'] = n_days_in
    cfg['r_day'] = r_day_in
    cfg['r_eve'] = r_eve_in
    cfg['r_night'] = r_night_in
    cfg['circadian'] = w_circadian_in
    cfg['night_imb'] = w_night_imb_in
    cfg['exp_mix'] = w_exp_mix_in
    cfg['pref_off'] = w_pref_off_in
    cfg['posta'] = w_posta_in

    st.divider()

    # 3. SATIR: CANLI AKIŞ & İZLEME AYARLARI
    st.markdown("### ⚡ 4. Canlı Optimizasyon Akışı & İzleme Ayarları (Live Streaming)")
    s_col1, s_col2 = st.columns([1, 2])
    with s_col1:
        live_enabled_in = st.toggle(
            "⚡ Canlı Akışı Göster",
            value=cfg.get('live_stream_enabled', True),
            key="ui_cfg_live_enabled",
            help="Algoritmalar (Genetik, SA, HC, Tabu, CSP) çalışırken ceza grafiğini ve arama hızını canlı adım adım gösterir."
        )
    with s_col2:
        live_interval_in = st.slider(
            "Grafik Yenileme Sıklığı (Her X İterasyonda Bir Güncelle)",
            min_value=10,
            max_value=150,
            value=cfg.get('live_stream_interval', 50),
            step=10,
            key="ui_cfg_live_interval",
            help="Optimizasyon çalışırken ekranın kaç adımda bir tazeleneceğini belirler (Varsayılan: 50 adım = ~60 FPS akıcı görselleştirme)."
        )
    
    cfg['live_stream_enabled'] = live_enabled_in
    cfg['live_stream_interval'] = live_interval_in

    st.divider()

    # 4. SATIR: ORTAK KADRO VE SERTİFİKA DÜZENLEYİCİ
    st.markdown("### 🪪 5. Ortak Personel Kadrosu ve Yetkinlik Düzenleyici")
    
    k_col1, k_col2, k_col3 = st.columns([2, 1, 1.2])
    with k_col1:
        kadro_options = ["Düzenli Kadro", "🎲 Rastgele Kadro"]
        curr_mode_idx = kadro_options.index(cfg['kadro_mode']) if cfg['kadro_mode'] in kadro_options else 0
        kadro_mode_in = st.radio(
            "Kadro Üretim Modu:",
            kadro_options,
            index=curr_mode_idx,
            horizontal=True,
            help="Düzenli Kadro: İzin talepleri günlere rotasyonlu/dengeli yayılır.\nRastgele Kadro: İzin talepleri, postalar ve sertifikalar TAMAMEN RASTGELE üretilir.",
            key="ui_cfg_km"
        )
    with k_col2:
        rand_seed_in = cfg['rand_seed']
        if "Rastgele" in kadro_mode_in:
            rand_seed_in = st.number_input("Rastgele Tohum (Seed)", 1, 1000, cfg['rand_seed'], key="ui_cfg_seed")
        else:
            st.caption("ℹ️ Düzenli kadroda deterministik 4-Posta MYK kuralları uygulanır.")

    with k_col3:
        st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        def force_regenerate_roster():
            _init_persistent_config()
            cfg_target = st.session_state["persistent_global_config"]
            fresh = generate_worker_profiles(
                cfg_target['n_workers'],
                cfg_target['n_days'],
                randomize=("Rastgele" in cfg_target['kadro_mode']),
                seed=cfg_target['rand_seed']
            )
            cfg_target['custom_workers'] = fresh
            cfg_target['_last_kadro_sig'] = (
                cfg_target['n_workers'],
                cfg_target['n_days'],
                cfg_target['kadro_mode'],
                cfg_target['rand_seed']
            )
            cfg_target['editor_version'] = cfg_target.get('editor_version', 0) + 1

        st.button("🔄 Kadroyu Sıfırla / Yeniden Üret", on_click=force_regenerate_roster, use_container_width=True, key="btn_regen_roster")

    # Kadro üretim tetikleyicisi kontrolü
    curr_sig = (n_workers_in, n_days_in, kadro_mode_in, rand_seed_in)
    last_sig = cfg.get('_last_kadro_sig')

    if curr_sig != last_sig or cfg.get('custom_workers') is None or len(cfg['custom_workers']) != n_workers_in:
        base_workers = generate_worker_profiles(
            n_workers_in,
            n_days_in,
            randomize=("Rastgele" in kadro_mode_in),
            seed=rand_seed_in
        )
        cfg['custom_workers'] = base_workers
        cfg['_last_kadro_sig'] = curr_sig
        cfg['editor_version'] = cfg.get('editor_version', 0) + 1
    else:
        base_workers = cfg['custom_workers']

    cfg['kadro_mode'] = kadro_mode_in
    cfg['rand_seed'] = rand_seed_in

    df_roster_input = pd.DataFrame([
        {
            "İşçi Adı": w['name'],
            "Posta": w['posta'],
            "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi",
            "Vinç Operatörü": "Vinç Operatörü" in w['skills'],
            "Potacı": "Potacı" in w['skills'],
            "Sıcak Metal Döküm Uzmanı": "Sıcak Metal Döküm Uzmanı" in w['skills'],
            "Gaz İzleme Sorumlusu": "Gaz İzleme Sorumlusu" in w['skills'],
            "Talep Edilen İzin (Gün)": int(w.get('pref_off', 1))
        }
        for w in base_workers
    ])

    st.markdown("##### ✏️ Personel Kadrosunu & MYK Ehliyetlerini Düzenle:")
    editor_key = f"ui_roster_editor_{n_workers_in}_{n_days_in}_{cfg.get('editor_version', 0)}"
    df_edited_roster = st.data_editor(
        df_roster_input,
        num_rows="fixed",
        use_container_width=True,
        key=editor_key
    )
    
    # Düzenlenen kadroyu parse edip kalıcı depoya kaydet
    parsed_workers = parse_edited_dataframe_to_workers(df_edited_roster, n_days_in)
    cfg['custom_workers'] = parsed_workers

    st.success("✅ **Model Güncel:** Yapılan tüm ayarlar kaydedildi ve diğer sekmelerdeki algoritmalara anında aktarıldı.")


# Geriye dönük uyumluluk için alias
render_global_sidebar = get_global_params
