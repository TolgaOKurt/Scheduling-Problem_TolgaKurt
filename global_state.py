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

import json
import streamlit as st
import pandas as pd
from algorithms.worker_manager import generate_worker_profiles, parse_edited_dataframe_to_workers, normalize_posta_name

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
        'workload_imb': 20,
        'exp_mix': 60,
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
            randomize=("Rastgele" in GLOBAL_DEFAULTS['kadro_mode']),
            seed=GLOBAL_DEFAULTS['rand_seed'],
            mode=GLOBAL_DEFAULTS['kadro_mode']
        )
        st.session_state["persistent_global_config"] = {
            'n_workers': GLOBAL_DEFAULTS['n_workers'],
            'n_days': GLOBAL_DEFAULTS['n_days'],
            'r_day': GLOBAL_DEFAULTS['r_day'],
            'r_eve': GLOBAL_DEFAULTS['r_eve'],
            'r_night': GLOBAL_DEFAULTS['r_night'],
            'circadian': DEFAULT_PENALTIES['circadian'],
            'night_imb': DEFAULT_PENALTIES['night_imb'],
            'workload_imb': DEFAULT_PENALTIES['workload_imb'],
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
    """Kalıcı depoyu doğrular ve bekleyen güncellemeleri uygular."""
    _init_persistent_config()
    cfg = st.session_state["persistent_global_config"]

    # Bekleyen gecikmeli widget güncellemelerini state'e uygula
    if "_pending_ui_updates" in st.session_state:
        pending = st.session_state.pop("_pending_ui_updates")
        for k, v in pending.items():
            st.session_state[k] = v
            if k == "w_n_workers":
                cfg['n_workers'] = v


def clear_all_solver_caches():
    """Tüm çözücülerin oturum önbelleklerini temizler."""
    for res_k in [
        'res_t4_myopic', 'res_t4_staggered', 'res_t4_mrv', 'res_t4',
        'res_t5', 'res_t6', 'res_t7', 'res_t8', 'res_t9', 'res_t10',
        'res_t11', 'res_t12', 'res_t13', 'res_t14', 'res_t15'
    ]:
        if res_k in st.session_state:
            del st.session_state[res_k]


def _compute_full_config_signature(cfg, custom_workers=None):
    """Tüm model parametrelerinin, ceza ağırlıklarının ve işçi yetkinliklerinin benzersiz imzasını hesaplar."""
    workers_list = custom_workers if custom_workers is not None else cfg.get('custom_workers', [])
    w_sig = tuple(
        (w.get('id'), w.get('name'), w.get('posta'), w.get('is_usta'), tuple(sorted(list(w.get('skills', [])))), w.get('pref_off'))
        for w in workers_list
    ) if workers_list else ()
    
    return (
        cfg.get('n_workers'),
        cfg.get('n_days'),
        cfg.get('r_day'),
        cfg.get('r_eve'),
        cfg.get('r_night'),
        cfg.get('circadian'),
        cfg.get('night_imb'),
        cfg.get('workload_imb'),
        cfg.get('exp_mix'),
        cfg.get('pref_off'),
        cfg.get('posta'),
        cfg.get('kadro_mode'),
        cfg.get('rand_seed'),
        w_sig
    )


def get_global_params():
    """Tüm sekmelerin ve çözücülerin ortak kullandığı parametre setini döndürür."""
    _init_persistent_config()
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
        'workload_imb': cfg.get('workload_imb', DEFAULT_PENALTIES['workload_imb']),
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
            seed=rand_seed,
            mode=kadro_mode
        )
        cfg['custom_workers'] = custom_workers
        cfg['_last_kadro_sig'] = curr_sig
        cfg['editor_version'] = cfg.get('editor_version', 0) + 1
        clear_all_solver_caches()

    # Bütüncül model ve kadro değişiklik kontrolü (Herhangi bir parametre değiştiği an eski sonuçları temizle)
    curr_full_sig = _compute_full_config_signature(cfg, custom_workers)
    last_full_sig = cfg.get('_last_full_model_sig')
    if last_full_sig is not None and curr_full_sig != last_full_sig:
        clear_all_solver_caches()
    cfg['_last_full_model_sig'] = curr_full_sig

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
    _init_persistent_config()
    cfg = st.session_state["persistent_global_config"]

    # Widget'lar render edilmeden önce session_state anahtarlarını kalıcı değerlerle senkronize et
    widget_sync = [
        ("w_n_workers", cfg['n_workers']),
        ("w_n_days", cfg['n_days']),
        ("w_r_day", cfg['r_day']),
        ("w_r_eve", cfg['r_eve']),
        ("w_r_night", cfg['r_night']),
        ("w_circadian", cfg['circadian']),
        ("w_night_imb", cfg['night_imb']),
        ("w_workload_imb", cfg.get('workload_imb', DEFAULT_PENALTIES['workload_imb'])),
        ("w_exp_mix", cfg['exp_mix']),
        ("w_pref_off", cfg['pref_off']),
        ("w_posta", cfg['posta']),
        ("w_live_enabled", cfg.get('live_stream_enabled', True)),
        ("w_live_interval", cfg.get('live_stream_interval', 50)),
        ("w_kadro_mode", cfg['kadro_mode']),
        ("w_rand_seed", cfg['rand_seed'])
    ]
    for w_key, w_val in widget_sync:
        if w_key not in st.session_state:
            st.session_state[w_key] = w_val

    st.markdown("## ⚙️ Merkezi Model Yapılandırması & Kadro Yönetimi")
    st.markdown("""
    <div style="background-color: #eff6ff; border-left: 5px solid #2563eb; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; color: #1e3a8a;">
        📌 <b>Global Parametre Merkezi:</b> Bu sayfada belirlediğiniz <b>Personel Sayısı, Vardiya Gereksinimleri, Ceza Katsayıları, Canlı Akış Sıklığı ve Kadro Yetkinlikleri</b> tüm optimizasyon sekmelerine anında ve kalıcı olarak aktarılır.
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
        <div class="metric-label">Günlük Net Vardiya Talebi</div>
        <div class="metric-value" style="color: #d97706;">{total_daily_req} Kişi/Gün</div>
        </div>""", unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Çağrı Başı Ortalama Maliyet</div>
        <div class="metric-value" style="color: #6366f1;">{call_cost_ms:.3f} ms</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # HIZLI ÇELİK TESİSİ & VARDİYA ÖN AYARLARI (PRESETS)
    st.markdown("### 🎛️ Çelik Sanayisi Hızlı Tesis & Vardiya Ön Ayarları (Presets)")
    st.markdown("""
    <div style="font-size: 0.92rem; color: #334155; margin-bottom: 12px;">
        Tek tıkla fabrikanın operasyonel büyüklüğüne göre <b>Gündüz, Akşam, Gece vardiya taleplerini</b> ve <b>4-Posta kuralına (<i>N = ⁴⁄₃ × Günlük İhtiyaç</i>)</b> uygun gerekli <b>Toplam Personel Sayısını (N)</b> otomatik ayarlar:
    </div>
    """, unsafe_allow_html=True)

    def set_shift_preset(n_w, r_d, r_e, r_n):
        _init_persistent_config()
        cfg_target = st.session_state["persistent_global_config"]
        cfg_target['n_workers'] = n_w
        cfg_target['r_day'] = r_d
        cfg_target['r_eve'] = r_e
        cfg_target['r_night'] = r_n
        
        st.session_state["w_n_workers"] = n_w
        st.session_state["w_r_day"] = r_d
        st.session_state["w_r_eve"] = r_e
        st.session_state["w_r_night"] = r_n
        
        fresh_workers = generate_worker_profiles(
            n_w,
            cfg_target['n_days'],
            randomize=("Rastgele" in cfg_target['kadro_mode']),
            seed=cfg_target['rand_seed'],
            mode=cfg_target['kadro_mode']
        )
        cfg_target['custom_workers'] = fresh_workers
        cfg_target['_last_kadro_sig'] = (
            n_w,
            cfg_target['n_days'],
            cfg_target['kadro_mode'],
            cfg_target['rand_seed']
        )
        cfg_target['editor_version'] = cfg_target.get('editor_version', 0) + 1
        clear_all_solver_caches()

    pr1, pr2, pr3, pr4, pr5, pr6 = st.columns(6)
    with pr1:
        if st.button("🏭 Standart Çelikhane\n\n9-7-5 | N:28", width="stretch", key="btn_pr_std", help="Standart 1 Yüksek Fırın + 1 Sürekli Döküm Hattı (Günlük: 21 kişi / Kadro: 28)"):
            set_shift_preset(28, 9, 7, 5)
            st.rerun()
    with pr2:
        if st.button("⚡ Kompakt Haddehane\n\n6-5-4 | N:20", width="stretch", key="btn_pr_compact", help="Küçük ark ocağı veya bağımsız haddehane (Günlük: 15 kişi / Kadro: 20)"):
            set_shift_preset(20, 6, 5, 4)
            st.rerun()
    with pr3:
        if st.button("🔥 Çift Döküm Hattı\n\n11-9-7 | N:36", width="stretch", key="btn_pr_dual", help="2 Sürekli Döküm Makinesi + Pota Ocağı (Günlük: 27 kişi / Kadro: 36)"):
            set_shift_preset(36, 11, 9, 7)
            st.rerun()
    with pr4:
        if st.button("🏗️ Çoklu Döküm Tesisi\n\n15-12-9 | N:48", width="stretch", key="btn_pr_mega", help="Büyük Çelikhane & Çoklu Hat Kompleksi (Günlük: 36 kişi / Kadro: 48)"):
            set_shift_preset(48, 15, 12, 9)
            st.rerun()
    with pr5:
        if st.button("⚖️ Kok & Gaz Arıtma\n\n6-6-6 | N:24", width="stretch", key="btn_pr_coke", help="7/24 Kesintisiz Kimyasal Proses (Günlük: 18 kişi / Kadro: 24)"):
            set_shift_preset(24, 6, 6, 6)
            st.rerun()
    with pr6:
        if st.button("🛠️ Fırın Uyutma / Bakım\n\n4-4-4 | N:16", width="stretch", key="btn_pr_banking", help="Planlı Duruş / 4 MYK Zorunlu Asgari Kadro (Günlük: 12 kişi / Kadro: 16)"):
            set_shift_preset(16, 4, 4, 4)
            st.rerun()

    st.divider()

    # Widget On-Change Callbacks (Değişiklik anında kalıcı depoya yazar)
    def _on_n_workers_change():
        st.session_state["persistent_global_config"]['n_workers'] = st.session_state["w_n_workers"]
        clear_all_solver_caches()

    def _on_n_days_change():
        st.session_state["persistent_global_config"]['n_days'] = st.session_state["w_n_days"]
        clear_all_solver_caches()

    def _on_r_day_change():
        st.session_state["persistent_global_config"]['r_day'] = st.session_state["w_r_day"]
        clear_all_solver_caches()

    def _on_r_eve_change():
        st.session_state["persistent_global_config"]['r_eve'] = st.session_state["w_r_eve"]
        clear_all_solver_caches()

    def _on_r_night_change():
        st.session_state["persistent_global_config"]['r_night'] = st.session_state["w_r_night"]
        clear_all_solver_caches()

    def _on_circadian_change():
        st.session_state["persistent_global_config"]['circadian'] = st.session_state["w_circadian"]
        clear_all_solver_caches()

    def _on_night_imb_change():
        st.session_state["persistent_global_config"]['night_imb'] = st.session_state["w_night_imb"]
        clear_all_solver_caches()

    def _on_workload_imb_change():
        st.session_state["persistent_global_config"]['workload_imb'] = st.session_state["w_workload_imb"]
        clear_all_solver_caches()

    def _on_exp_mix_change():
        st.session_state["persistent_global_config"]['exp_mix'] = st.session_state["w_exp_mix"]
        clear_all_solver_caches()

    def _on_pref_off_change():
        st.session_state["persistent_global_config"]['pref_off'] = st.session_state["w_pref_off"]
        clear_all_solver_caches()

    def _on_posta_change():
        st.session_state["persistent_global_config"]['posta'] = st.session_state["w_posta"]
        clear_all_solver_caches()

    def _on_live_enabled_change():
        st.session_state["persistent_global_config"]['live_stream_enabled'] = st.session_state["w_live_enabled"]

    def _on_live_interval_change():
        st.session_state["persistent_global_config"]['live_stream_interval'] = st.session_state["w_live_interval"]

    def _on_kadro_mode_change():
        st.session_state["persistent_global_config"]['kadro_mode'] = st.session_state["w_kadro_mode"]
        clear_all_solver_caches()

    def _on_rand_seed_change():
        st.session_state["persistent_global_config"]['rand_seed'] = st.session_state["w_rand_seed"]
        clear_all_solver_caches()

    # 3. SATIR: 3 KOLONLU PARAMETRE DÜZENLEME PANELİ
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""<div class="card-box" style="border-top: 4px solid #2563eb; padding: 16px;">
        <h4 style="color: #1d4ed8; margin-top: 0;">👥 1. Personel & Periyot</h4>
        </div>""", unsafe_allow_html=True)
        
        min_w = 12
        max_w = max(100, int(cfg['n_workers']) + 20)
        step_w = 4 if int(cfg['n_workers']) % 4 == 0 else 1
        n_workers_in = st.slider("Toplam Personel Sayısı (N)", min_value=min_w, max_value=max_w, step=step_w, key="w_n_workers", on_change=_on_n_workers_change)
        n_days_in = st.slider("Planlama Periyodu / Gün (D)", min_value=7, max_value=28, step=7, key="w_n_days", on_change=_on_n_days_change)
        
        st.caption(f"📐 **Matris Boyutu:** {n_workers_in} × {n_days_in} = {n_workers_in * n_days_in} hücre")
        
        max_weekly_capacity = n_workers_in * 6
        needed_weekly_shifts = total_daily_req * 7
        if needed_weekly_shifts > max_weekly_capacity:
            st.warning(f"⚠️ **Kadro Yetersizliği Riski:** Haftalık {needed_weekly_shifts} vardiya ihtiyacına karşılık maksimum kapasite {max_weekly_capacity} vardiyadır.")

    with col2:
        st.markdown("""<div class="card-box" style="border-top: 4px solid #059669; padding: 16px;">
        <h4 style="color: #047857; margin-top: 0;">🏭 2. Vardiya Personel Talepleri (Zorunlu)</h4>
        </div>""", unsafe_allow_html=True)
        
        r_day_in = st.number_input("Gündüz (08-16) Talebi", min_value=1, max_value=30, key="w_r_day", on_change=_on_r_day_change)
        r_eve_in = st.number_input("Akşam (16-24) Talebi", min_value=1, max_value=30, key="w_r_eve", on_change=_on_r_eve_change)
        r_night_in = st.number_input("Gece (24-08) Talebi", min_value=1, max_value=30, key="w_r_night", on_change=_on_r_night_change)

    with col3:
        st.markdown("""<div class="card-box" style="border-top: 4px solid #d97706; padding: 16px;">
        <h4 style="color: #b45309; margin-top: 0;">⚖️ 3. Ortak Ceza Ağırlıkları</h4>
        </div>""", unsafe_allow_html=True)
        
        def reset_penalty_weights():
            _init_persistent_config()
            cfg_target = st.session_state["persistent_global_config"]
            for k in ['circadian', 'night_imb', 'workload_imb', 'exp_mix', 'pref_off', 'posta']:
                cfg_target[k] = DEFAULT_PENALTIES[k]
            st.session_state["w_circadian"] = DEFAULT_PENALTIES['circadian']
            st.session_state["w_night_imb"] = DEFAULT_PENALTIES['night_imb']
            st.session_state["w_workload_imb"] = DEFAULT_PENALTIES['workload_imb']
            st.session_state["w_exp_mix"] = DEFAULT_PENALTIES['exp_mix']
            st.session_state["w_pref_off"] = DEFAULT_PENALTIES['pref_off']
            st.session_state["w_posta"] = DEFAULT_PENALTIES['posta']

        st.button("🔄 Cezaları Varsayılana Sıfırla", on_click=reset_penalty_weights, width="stretch", key="btn_reset_penalties")

        w_circadian_in = st.slider("Sirkadiyen Ritim Cezası", min_value=10, max_value=100, step=10, key="w_circadian", on_change=_on_circadian_change)
        w_night_imb_in = st.slider("Gece Dengesizlik Cezası", min_value=5, max_value=50, step=5, key="w_night_imb", on_change=_on_night_imb_change)
        w_workload_in = st.slider(
            "Toplam İş Yükü Dengesizlik Cezası",
            min_value=5,
            max_value=50,
            step=5,
            key="w_workload_imb",
            on_change=_on_workload_imb_change,
            help=f"Çalışanlar arasındaki toplam aktif vardiya sayısı dengesizliği için kesilecek birim ceza puanı (Varsayılan: {DEFAULT_PENALTIES['workload_imb']} Puan/Vardiya)."
        )
        w_exp_mix_in = st.slider("Kıdem (Usta Yoksa) Cezası", min_value=10, max_value=100, step=5, key="w_exp_mix", on_change=_on_exp_mix_change)
        w_pref_off_in = st.slider("Kişisel İzin İhlal Cezası", min_value=10, max_value=100, step=10, key="w_pref_off", on_change=_on_pref_off_change)
        w_posta_in = st.slider(
            "Posta Bölünme Cezası (Kişi Başı)",
            min_value=5,
            max_value=50,
            step=5,
            key="w_posta",
            on_change=_on_posta_change,
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
    cfg['workload_imb'] = w_workload_in
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
            key="w_live_enabled",
            on_change=_on_live_enabled_change,
            help="Algoritmalar (Genetik, SA, HC, Tabu, CSP) çalışırken ceza grafiğini ve arama hızını canlı adım adım gösterir."
        )
    with s_col2:
        live_interval_in = st.slider(
            "Grafik Yenileme Sıklığı (Her X İterasyonda Bir Güncelle)",
            min_value=10,
            max_value=150,
            step=10,
            key="w_live_interval",
            on_change=_on_live_interval_change,
            help="Optimizasyon çalışırken ekranın kaç adımda bir tazeleneceğini belirler (Varsayılan: 50 adım = ~60 FPS akıcı görselleştirme)."
        )
    
    cfg['live_stream_enabled'] = live_enabled_in
    cfg['live_stream_interval'] = live_interval_in

    st.divider()

    # 4. SATIR: ORTAK KADRO VE SERTİFİKA DÜZENLEYİCİ
    st.markdown("### 🪪 5. Ortak Personel Kadrosu ve Yetkinlik Düzenleyici")
    
    k_col1, k_col2, k_col3 = st.columns([2.2, 1.2, 1.2])
    with k_col1:
        kadro_options = ["Düzenli Kadro", "🌟 Mükemmel Kadro", "🎲 Rastgele Kadro"]
        kadro_mode_in = st.radio(
            "Kadro Üretim Modu:",
            kadro_options,
            horizontal=True,
            help="Düzenli Kadro: İzin talepleri günlere rotasyonlu/dengeli yayılır.\nMükemmel Kadro: Herkes Kıdemli Usta, her 4 MYK sertifikasına sahip ve izinler sırayla talep edilir.\nRastgele Kadro: İzin talepleri, postalar ve sertifikalar TAMAMEN RASTGELE üretilir.",
            key="w_kadro_mode",
            on_change=_on_kadro_mode_change
        )
    with k_col2:
        rand_seed_in = cfg['rand_seed']
        if "Rastgele" in kadro_mode_in:
            rand_seed_in = st.number_input("Rastgele Tohum (Seed)", min_value=1, max_value=1000, key="w_rand_seed", on_change=_on_rand_seed_change)
        elif "Mükemmel" in kadro_mode_in:
            st.caption("✨ Tüm personel Kıdemli Usta, 4 MYK sertifikalı ve sıralı izinli.")
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
                seed=cfg_target['rand_seed'],
                mode=cfg_target['kadro_mode']
            )
            cfg_target['custom_workers'] = fresh
            cfg_target['_last_kadro_sig'] = (
                cfg_target['n_workers'],
                cfg_target['n_days'],
                cfg_target['kadro_mode'],
                cfg_target['rand_seed']
            )
            cfg_target['editor_version'] = cfg_target.get('editor_version', 0) + 1
            clear_all_solver_caches()

        st.button("🔄 Kadroyu Sıfırla / Yeniden Üret", on_click=force_regenerate_roster, width="stretch", key="btn_regen_roster")

    # Kadro üretim tetikleyicisi kontrolü
    curr_sig = (n_workers_in, n_days_in, kadro_mode_in, rand_seed_in)
    last_sig = cfg.get('_last_kadro_sig')

    if curr_sig != last_sig or cfg.get('custom_workers') is None or len(cfg['custom_workers']) != n_workers_in:
        base_workers = generate_worker_profiles(
            n_workers_in,
            n_days_in,
            randomize=("Rastgele" in kadro_mode_in),
            seed=rand_seed_in,
            mode=kadro_mode_in
        )
        cfg['custom_workers'] = base_workers
        cfg['_last_kadro_sig'] = curr_sig
        cfg['editor_version'] = cfg.get('editor_version', 0) + 1
        clear_all_solver_caches()
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
        width="stretch",
        key=editor_key
    )
    
    # Düzenlenen kadroyu parse edip kalıcı depoya kaydet
    parsed_workers = parse_edited_dataframe_to_workers(df_edited_roster, n_days_in)
    old_parsed_sig = tuple(
        (w.get('id'), w.get('name'), w.get('posta'), w.get('is_usta'), tuple(sorted(list(w.get('skills', [])))), w.get('pref_off'))
        for w in (cfg.get('custom_workers') or [])
    )
    new_parsed_sig = tuple(
        (w.get('id'), w.get('name'), w.get('posta'), w.get('is_usta'), tuple(sorted(list(w.get('skills', [])))), w.get('pref_off'))
        for w in parsed_workers
    )
    if old_parsed_sig != new_parsed_sig:
        clear_all_solver_caches()
    cfg['custom_workers'] = parsed_workers

    st.success("✅ **Model Güncel:** Yapılan tüm ayarlar kaydedildi ve diğer sekmelerdeki algoritmalara anında aktarıldı.")

    # ==============================================================================
    # 5.1 PERSONEL KADROSU INPUT / OUTPUT (İÇE & DIŞA AKTARMA)
    # ==============================================================================
    st.divider()
    st.markdown("##### 📁 Kadro Dosyası Yönetimi: Dışa Aktar (Output) & İçe Aktar (Input)")

    io_col1, io_col2 = st.columns(2)

    with io_col1:
        st.markdown("""
        <div style="background-color: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
            <div style="font-weight: 700; color: #1e293b; font-size: 0.95rem; margin-bottom: 4px;">
                📤 Kadroyu Dışa Aktar (Output / İndir)
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px; line-height: 1.4;">
                Mevcut tablodaki personel yetkinliklerini ve izin taleplerini <b>Excel uyumlu CSV</b> veya <b>JSON</b> formatında cihazınıza indirin:
            </div>
        </div>
        """, unsafe_allow_html=True)

        # CSV Export (UTF-8 BOM ile Excel'de Türkçe karakter hatası vermez)
        csv_bytes = df_edited_roster.to_csv(index=False).encode('utf-8-sig')

        # JSON Export
        json_export_list = [
            {
                "id": w["id"],
                "name": w["name"],
                "posta": w["posta"],
                "is_usta": w["is_usta"],
                "skills": sorted(list(w["skills"])),
                "pref_off": w["pref_off"]
            }
            for w in parsed_workers
        ]
        json_bytes = json.dumps(json_export_list, ensure_ascii=False, indent=2).encode('utf-8')

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="📥 CSV İndir (.csv)",
                data=csv_bytes,
                file_name=f"kadro_{n_workers_in}kisi_{n_days_in}gun.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_dl_roster_csv",
                help="Excel veya LibreOffice ile açıp düzenleyebileceğiniz CSV dosyası."
            )
        with d_col2:
            st.download_button(
                label="📥 JSON İndir (.json)",
                data=json_bytes,
                file_name=f"kadro_{n_workers_in}kisi_{n_days_in}gun.json",
                mime="application/json",
                use_container_width=True,
                key="btn_dl_roster_json",
                help="Tam sistem profil yapısını barındıran JSON formatı."
            )

    with io_col2:
        st.markdown("""
        <div style="background-color: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-bottom: 10px;">
            <div style="font-weight: 700; color: #1e293b; font-size: 0.95rem; margin-bottom: 4px;">
                📥 Kadro Yükle (Input / İçe Aktar)
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 12px; line-height: 1.4;">
                Daha önce dışa aktardığınız veya Excel'de hazırladığınız <b>CSV</b> veya <b>JSON</b> kadro dosyasını yükleyin:
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_roster = st.file_uploader(
            "Kadro Dosyası Seç (.csv veya .json)",
            type=["csv", "json"],
            key="ui_roster_file_uploader",
            help="İşçi Adı, Posta, Unvan, MYK Ehliyetleri ve İzin Günü içeren dosya yükleyiniz."
        )

        if uploaded_roster is not None:
            # Dosyanın her sayfa etkileşiminde tekrar tekrar okunmasını önleyen tek-seferlik imza kontrolü
            file_id = f"{uploaded_roster.name}_{uploaded_roster.size}"
            if st.session_state.get("_last_processed_file_id") != file_id:
                try:
                    imported_workers = []
                    file_name = uploaded_roster.name.lower()

                    if file_name.endswith(".csv"):
                        df_up = pd.read_csv(uploaded_roster)
                        imported_workers = parse_edited_dataframe_to_workers(df_up, n_days_in)
                    elif file_name.endswith(".json"):
                        raw_json = json.loads(uploaded_roster.getvalue().decode('utf-8'))
                        if isinstance(raw_json, list):
                            for idx, item in enumerate(raw_json):
                                skills = set(item.get("skills", []))
                                if item.get("is_usta", False):
                                    skills.add("Kıdemli Usta")
                                pref_val = int(item.get("pref_off", (idx % n_days_in) + 1))
                                pref_val = max(1, min(n_days_in, pref_val))
                                imported_workers.append({
                                    "id": idx,
                                    "name": str(item.get("name", item.get("İşçi Adı", f"İşçi #{idx+1:02d}"))),
                                    "posta": normalize_posta_name(item.get("posta", item.get("Posta", None)), idx),
                                    "is_usta": bool(item.get("is_usta", False) or str(item.get("Unvan", "")).strip().lower() in ["kıdemli usta", "kidemli usta", "usta"]),
                                    "skills": skills,
                                    "pref_off": pref_val
                                })

                    if imported_workers:
                        n_imp = len(imported_workers)
                        cfg['n_workers'] = n_imp
                        cfg['custom_workers'] = imported_workers
                        cfg['_last_kadro_sig'] = (n_imp, n_days_in, cfg['kadro_mode'], cfg['rand_seed'])
                        cfg['editor_version'] = cfg.get('editor_version', 0) + 1
                        st.session_state["_last_processed_file_id"] = file_id

                        # Widget'lar henüz oluşturulmadan bir sonraki çalıştırmada uygulanmak üzere pending'e kaydet
                        if "_pending_ui_updates" not in st.session_state:
                            st.session_state["_pending_ui_updates"] = {}
                        st.session_state["_pending_ui_updates"]["w_n_workers"] = n_imp
                        st.session_state["w_n_workers"] = n_imp

                        clear_all_solver_caches()
                        st.success(f"🎉 **{n_imp} Kişilik Kadro Başarıyla İçe Aktarıldı!** Tablo ve N={n_imp} personel sayısı eşitlendi.")
                        st.rerun()
                    else:
                        st.error("⚠️ Yüklenen dosyada geçerli işçi kaydı ayrıştırılamadı.")
                except Exception as e:
                    st.error(f"❌ Kadro dosyası okunurken hata oluştu: {e}")
        else:
            # Dosya kaldırıldığında veya temizlendiğinde hafızadaki dosya kilidini sıfırla
            if "_last_processed_file_id" in st.session_state:
                del st.session_state["_last_processed_file_id"]


# Geriye dönük uyumluluk için alias
render_global_sidebar = get_global_params
