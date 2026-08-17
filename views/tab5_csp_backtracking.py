"""
================================================================================
  VIEWS/TAB5_CSP_BACKTRACKING.PY - SEKME 5: BACKTRACKING / CSP SOLVER & SİMÜLASYONU
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.csp_backtracking_solver import CSPBacktrackingSolver
from views.common_components import (
    render_schedule_heatmap,
    render_workload_chart,
    render_penalties_chart,
    render_posta_load_chart,
    render_shift_posta_stacked_chart,
    render_schedule_matrix_table,
    render_request_details_expander,
    render_worker_profiles_table,
    render_evaluator_cost_badge,
    LiveStreamTracker,
    render_live_stream_summary
)

def render_tab5(params):
    """Sekme 5 içeriğini çizer: Backtracking / CSP Teorisi, Simülasyonu, Grafikler ve Karşılaştırma Matrisi."""
    st.markdown("## 🔍 Backtracking / Constraint Satisfaction Problem (CSP) Yaklaşımı")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #059669;">
<div class="card-title" style="color: #047857;">🧩 CSP Çerçevesi (Değişkenler & Kümeler)</div>
<ul>
<li><b>Değişkenler (X<sub>i,t</sub>):</b> <i>i</i> çalışanının <i>t</i> günündeki vardiya ataması.</li>
<li><b>Etki Alanı (Domain D<sub>i,t</sub>):</b> {0: OFF, 1: Gündüz, 2: Akşam, 3: Gece}</li>
<li><b>Geri İzleme (Backtracking):</b> Bir vardiyada kadro eksik kaldığında algoritma durmaz; önceki günlere dönerek (backtrack) alternatif işçi kombinasyonlarını dener.</li>
<li><b>Pruning (Budama):</b> Sert kısıtları ihlal eden mantıksız arama dalları önceden budanarak arama ağacı küçültülür.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb;">
<div class="card-title" style="color: #1d4ed8;">⚡ Sezgisel Yönlendirme (Heuristics)</div>
<ul>
<li><b>MRV (Minimum Remaining Values):</b> En çok kısıtlanmış ve atama seçeneği en az kalan gün/işçiden aramaya başlar.</li>
<li><b>LCV (Least Constraining Value):</b> Gelecekteki günlerin seçeneklerini en az kısıtlayan vardiya atamasına öncelik verir.</li>
<li><b>Kadro Boşalması ve İzin Birikmesini Önleme:</b> Backtracking mekanizması sayesinde haftanın son günlerinde kadro yetersizliği yaşanmaz; haftalık izinler tüm haftaya %100 dengeli dağıtılır.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🎛️ Backtracking / CSP Arama Limiti Kontrolü")
    max_backtracks = st.number_input("Maksimum Backtrack Limiti", min_value=500, max_value=10000, value=3000, step=500, key="csp_mb")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # HESAPLAMA YÜKÜ & ÖN-ANALİZ PANELİ
    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    t5_time_str = f"{cost_ms:.2f} ms"

    st.markdown("#### 🧮 Arama Ağacı Boyutu & Ceza Değerlendirici Tahmin Paneli (Ön-Analiz)")
    c_est1, c_est2 = st.columns(2)
    with c_est1:
        render_evaluator_cost_badge(1, cost_ms, label="Tahmini Ceza Değerlendirme")
    with c_est2:
        st.metric(label="🌳 Maksimum Budama Limiti", value=f"{max_backtracks:,} Adım", help="Sonsuz döngüyü önleyen emniyet tavan sınırı.")

    st.divider()

    run_btn = st.button("🚀 CSP Backtracking Çözücüsünü Çalıştır", type="primary", width="stretch", key="btn_run_t5")

    if not run_btn and "res_t5" not in st.session_state:
        st.warning("👈 Arama ağacını çalıştırmak için yukarıdaki **'🚀 CSP Backtracking Çözücüsünü Çalıştır'** butonuna basınız.")
        return

    if run_btn:
        with st.spinner("⏳ CSP Backtracking Arama Ağacı Taranıyor..."):
            solver = CSPBacktrackingSolver(
                n_workers=num_workers,
                n_days=num_days,
                r_day=req_day,
                r_eve=req_eve,
                r_night=req_night,
                max_backtracks=max_backtracks,
                custom_workers=custom_workers,
                weights=weights
            )
            results = solver.solve()
            st.session_state["res_t5"] = results
    else:
        results = st.session_state["res_t5"]

    meta = results.get('meta', {})
    backtracks_cnt = meta.get('backtracks', results.get('backtracks', 0))
    nodes_cnt = meta.get('nodes_explored', results.get('nodes_explored', 0))
    is_feas = results.get('is_feasible', results.get('success', False))
    exec_time = results.get('exec_time_ms', 0.0)

    # --- CANLI ARAMA AĞACI DURUM BİLGİSİ ---
    st.markdown("#### ⚡ CSP Backtracking Arama Ağacı Taraması (Kısıt Tatmin Durumu)")
    if is_feas:
        st.progress(1.0, text=f"✅ %100 Tamamlandı | Geri İzleme (Backtrack): {backtracks_cnt:,} / {max_backtracks:,} | Gezilen Düğüm: {nodes_cnt:,} | Toplam Süre: {exec_time:.2f} ms | ✅ Geçerli Çözüm Bulundu")
    elif backtracks_cnt >= max_backtracks:
        st.progress(1.0, text=f"⏱️ Limit Aşıldı | Geri İzleme (Backtrack): {backtracks_cnt:,} / {max_backtracks:,} | Gezilen Düğüm: {nodes_cnt:,} | Toplam Süre: {exec_time:.2f} ms | ⏱️ Maksimum Adım Sınırına Ulaşıldı")
    else:
        st.progress(1.0, text=f"❌ Çözüm Yok | Geri İzleme (Backtrack): {backtracks_cnt:,} | Gezilen Düğüm: {nodes_cnt:,} | Toplam Süre: {exec_time:.2f} ms | ❌ Kısıt Çıkmazı")

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        if is_feas:
            status_str = "✅ BAŞARILI"
            status_clr = "#059669"
        elif backtracks_cnt >= max_backtracks:
            status_str = "⏱️ LİMİT AŞILDI"
            status_clr = "#dc2626"
        else:
            status_str = "❌ ÇÖZÜM YOK"
            status_clr = "#dc2626"

        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">CSP Durumu</div>
        <div class="metric-value" style="color: {status_clr}; font-size: 1.05rem;">{status_str}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Geri İzleme (Backtrack)</div>
        <div class="metric-value" style="color: #d97706;">{backtracks_cnt}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Gezilen Düğüm</div>
        <div class="metric-value" style="color: #2563eb;">{nodes_cnt}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        eval_c = results.get('eval_count', 1)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Ceza Çağrısı (Evaluator)</div>
        <div class="metric-value" style="color: #6366f1;">{eval_c} Adet</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Çözüm Süresi</div>
        <div class="metric-value" style="color: #059669;">{results['exec_time_ms']} ms</div>
        </div>""", unsafe_allow_html=True)

    with m6:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam Ceza Puanı</div>
        <div class="metric-value" style="color: #dc2626;">{results['final_score']}</div>
        </div>""", unsafe_allow_html=True)

    if not is_feas:
        if backtracks_cnt >= max_backtracks:
            st.error(f"⏱️ **Maksimum Backtrack Limiti ({max_backtracks:,} Adım) Aşıldı:** Arama algoritması belirlenen adım sınırı içinde geçerli bir çizelge bulamadı. Yukarıdaki arama limitini artırmayı deneyebilirsiniz.")
        else:
            st.error(f"❌ **Matematiksel Olarak Geçerli Çözüm Bulunamadı (Kadro / Kısıt Çıkmazı):** Arama ağacı ({backtracks_cnt:,} geri izleme ile) tamamen tarandı ve sert kısıtları %100 sağlayan hiçbir kombinasyon bulunamadı. Toplam personel ({num_workers}) ve günlük vardiya ihtiyaçları ({req_day} Gündüz + {req_eve} Akşam + {req_night} Gece = {req_day+req_eve+req_night} işçi/gün) haftalık zorunlu izin (max 6 gün çalışma) ve dinlenme kuralları ile çelişmektedir.")

    st.divider()

    # --- SERT VE YUMUŞAK KISIT DEĞERLENDİRME VE ZAYIFLIK ANALİZ UYARISI ---
    st.warning("""
    💡 **Yöneylem Araştırması Analizi (CSP / Backtracking Yaklaşımının Zayıf Yönleri ve Başarımı):**  
    Saf **Backtracking (CSP)** algoritması temel olarak **Sert Kısıtları %100 doğrulayan ilk geçerli çözümü (Satisfiability)** bulmayı hedefler. 
    Bu durum şu sonuçları doğurur:
    - **Güçlü Yönü:** Sert kısıtları (Pazar izin çöküşü, MYK ehliyet eksiklikleri, 11 saat dinlenme süreleri) %100 Kusursuz Garanti Eder.
    - **Kötü Yaptığı / İhmal Ettiği Yön:** İlk bulduğu geçerli çözüme kilitlendiği için Yumuşak Kısıtları (Örn: Gece nöbetinin postalar arasında tam adil dağılımı veya sirkadiyen vardiya dönüşleri) bir **MILP (Tam Sayılı Optimizasyon)** kadar mükemmel minimize etmeyebilir. Aşağıdaki grafiklerden kalan yumuşak ceza puanlarını gözlemleyebilirsiniz.
    """)

    st.divider()

    # --- PERSONEL YETKİNLİK VE MYK SERTİFİKA KADRO TABLOSU ---
    render_worker_profiles_table(results['workers'], title="🪪 Personel Yetkinlik ve MYK Sertifika Kadro Listesi")

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Backtracking CSP Çözüm Grafikleri & Ceza Analizi")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        render_schedule_heatmap(results['schedule'], results['workers'], num_days, key_prefix="t5_csp", title="1️⃣ CSP Vardiya Dağılım Isı Haritası (Heatmap)")

    with g_col2:
        render_workload_chart(results['schedule'], results['workers'], key_prefix="t5_csp", title="2️⃣ CSP Personel Vardiya & Gece Nöbet Dağılımı", color_seq=["#059669", "#dc2626"])

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        render_penalties_chart(results['penalties'], key_prefix="t5_csp", title="3️⃣ CSP Yumuşak Kısıt Ceza Puanı Dağılımı")

    with g_col4:
        render_posta_load_chart(results['schedule'], results['workers'], key_prefix="t5_csp", title="4️⃣ CSP Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")

    # 5. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI
    render_shift_posta_stacked_chart(results['schedule'], results['workers'], num_days, key_prefix="t5_csp", title="5️⃣ Gün ve Vardiya Bazında Posta Dağılımı (Gündüz, Akşam ve Gece Vardiyalarında Hangi Postadan Kaç Kişi Var?)")

    st.divider()

    # --- FULL SCHEDULE MATRIX TABLE ---
    render_schedule_matrix_table(results['schedule'], results['workers'], num_days, title="🗓️ Backtracking / CSP Tarafından Üretilen Tam Vardiya Çizelgesi")
    
    # --- KİŞİSEL İZİN TALEPLERİ DETAY RAPORU ---
    if 'request_details' in results:
        render_request_details_expander(results['request_details'])
