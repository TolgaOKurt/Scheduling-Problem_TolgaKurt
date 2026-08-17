"""
================================================================================
  VIEWS/TAB7_HILL_CLIMBING.PY - SEKME 7: HILL CLIMBING / LOCAL SEARCH (TIRMANMA)
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.hill_climbing_solver import run_hill_climbing
from views.common_components import (
    render_schedule_heatmap,
    render_workload_chart,
    render_penalties_chart,
    render_posta_load_chart,
    render_shift_posta_stacked_chart,
    render_schedule_matrix_table,
    render_request_details_expander,
    render_hard_constraints_status_card,
    render_evaluator_cost_badge,
    render_metaheuristic_metric_cards,
    render_standard_schedule_analytics,
    LiveStreamTracker,
    render_live_stream_summary
)

def render_tab7(params):
    """Sekme 7 içeriğini çizer: Hill Climbing / Local Search Metasezgisel Çözücüsü ve Grafikler."""
    st.markdown("## 🏔️ Hill Climbing / Local Search (Tepeden Tırmanma & Yöresel Arama)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #d97706;">
<div class="card-title" style="color: #b45309;">🔄 Komşuluk Operatörleri (Neighborhood Search)</div>
<ul>
<li><b>Vardiya Takası (Swap Move):</b> Aynı gün içerisinde çalışan iki işçinin vardiyaları takas edilir.</li>
<li><b>Kabul Kriteri (Iterative Descent):</b> Yapılan takas Sert Kısıtları bozmuyorsa VE toplam ceza puanını <b>düşürüyorsa</b> hamle kabul edilir (&Delta;Z < 0).</li>
<li><b>Yöresel Arama:</b> Arama uzayında adım adım ilerleyerek daha adil nöbet ve posta bütünlüğü arar.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #7c3aed;">
<div class="card-title" style="color: #6d28d9;">📈 Yakınsama & Yerel Optimum (Local Optimum)</div>
<ul>
<li><b>Aşamalı İyileşme:</b> İlk Greedy çizelgenin eksiklerini (sirkadiyen ihlaller, posta bölünmeleri) binlerce iterasyonda düzeltir.</li>
<li><b>Yakınsama Eğrisi:</b> Algoritmanın iterasyonlar boyunca ceza puanını nasıl adım adım düşürdüğü canlı grafikte izlenir.</li>
<li><b>Limit:</b> Çevresinde daha iyi bir komşu kalmadığında veya maksimum iterasyon bittiğinde durur.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- METASEZGİSEL (HILL CLIMBING) ÜSTÜNLÜKLERİ PANOLARI ---
    st.markdown("### 🚀 Metasezgisel (Hill Climbing) Yönteminin ILP / MILP'e Göre 4 Temel Üstünlüğü")

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        st.markdown("""<div style="background-color: #f0fdf4; border: 2px solid #059669; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #047857; margin-top: 0;">1. 🌐 Devasa Tesislerde Ölçeklenebilirlik (Scalability)</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
MILP problemleri NP-Hard'dır. 500+ işçi ve 365 günlük devasa tesislerde MILP denklem sayısı milyonlara ulaşır ve RAM/süre kilitlenmesi yaşanır. <b>Hill Climbing ise hafızayı şişirmeden saniyeler içinde çözüme ulaşır.</b>
</p>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background-color: #eff6ff; border: 2px solid #2563eb; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #1e40af; margin-top: 0;">2. 🧩 Doğrusal Olmayan (Non-Linear) Kısıt Esnekliği</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
ILP'de tüm kurallar doğrusal (linear) olmak zorundadır. Hill Climbing'de ise <b>if-else mantığı, yapay zeka sinir ağları, olasılıksal/stokastik kurallar veya karmaşık Python fonksiyonları serbestçe kullanılabilir.</b>
</p>
</div>""", unsafe_allow_html=True)

    with adv_col2:
        st.markdown("""<div style="background-color: #fef3c7; border: 2px solid #d97706; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #b45309; margin-top: 0;">3. ⏱️ Anlık Vardiya Tamiri (Real-Time Re-Scheduling)</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
Aniden 2 işçi hastalanıp gelmediğinde MILP tüm modeli sıfırdan kurup çözer. <b>Hill Climbing ise var olan planı başlangıç alarak 50 ms içinde lokal takaslarla anında tamir eder (Repair Heuristic).</b>
</p>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background-color: #faf5ff; border: 2px solid #7c3aed; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #6d28d9; margin-top: 0;">4. ⚙️ Düşük Donanım & Çözücü Bağımsızlığı</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
Pahalı veya ticari MILP çözücülere (Gurobi, CPLEX) ihtiyaç duymaz. <b>Tamamen açık kaynaklı ve her türlü cihazda hafif çalışan bir kural motoruna sahiptir.</b>
</p>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🎛️ Hill Climbing İterasyon Kontrolü")
    max_iter = st.slider("Maksimum İterasyon Sayısı (K)", 2000, 10000, 3000, 1000, key="hc_iter")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # HESAPLAMA YÜKÜ & ÖN-ANALİZ PANELİ
    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    # Komşuluk takaslarında farklı vardiya ve sert kısıt uygunluk olasılığı ~%28'dir
    hc_est_calls = int(max_iter * 0.28)
    hc_total_ms = hc_est_calls * cost_ms
    hc_time_str = f"{hc_total_ms/1000:.2f} sn" if hc_total_ms >= 1000 else f"{hc_total_ms:.0f} ms"

    st.markdown("#### 🧮 İterasyon & Ceza Değerlendirici Tahmin Paneli (Ön-Analiz)")
    c_est1, c_est2 = st.columns(2)
    with c_est1:
        render_evaluator_cost_badge(hc_est_calls, cost_ms, label="Tahmini Ceza Değerlendirme")
    with c_est2:
        st.metric(label="🎯 Maksimum İterasyon Limiti", value=f"{max_iter:,} Adım", help="Yerel aramada denenecek maksimum komşuluk sayısı.")

    st.divider()

    run_btn = st.button("🚀 Hill Climbing Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t7")

    if not run_btn and "res_t7" not in st.session_state:
        st.warning("👈 Optimizasyonu başlatmak için yukarıdaki **'🚀 Hill Climbing Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = params.get('live_stream_enabled', True)
        stream_interval = params.get('live_stream_interval', 50)
        
        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="Hill Climbing Yerel Arama",
            max_steps=max_iter,
            unit_name="İterasyon",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t7_tracker"
        )
        
        results = run_hill_climbing(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            weights=weights,
            max_iterations=max_iter,
            seed=42,
            custom_workers=custom_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=stream_interval
        )
        tracker.finish(results.get('total_iterations', max_iter), results.get('final_score'))
        results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t7"] = results
    else:
        results = st.session_state["res_t7"]
        if params.get('live_stream_enabled', True) and 'stream_data' in results:
            render_live_stream_summary(results['stream_data'])

    # --- BİTİRME NEDENİ VE SERT KISIT UYGUNLUK BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")
    render_hard_constraints_status_card(
        results.get('is_feasible', True),
        results.get('hard_violations_count', 0),
        results.get('hard_violation_logs', []),
        solver_name="Hill Climbing (Tepeden Tırmanma)"
    )

    # --- METRİK KARTLARI ---
    render_metaheuristic_metric_cards(results, move_label="Kabul Edilen Hamle")

    st.divider()

    # --- 3 GELİŞMİŞ YÖNTEMİN KARŞILAŞTIRMA MATRİSİ ---
    st.markdown("### 📊 Gelişmiş Yaklaşımların Bütüncül Karşılaştırma Matrisi (CSP vs. ILP vs. Hill Climbing)")

    df_3way_hc = pd.DataFrame({
        "Karşılaştırma Özelliği": [
            "Algoritma Sınıfı",
            "Sert Kısıt Garantisi (%100 Feasibility)",
            "Ceza İyileştirme Mekanizması",
            "Arama Yöntemi",
            "En Uygun (Optimal) Çözüm Garantisi",
            "Hesaplama Esnekliği"
        ],
        "Sekme 5: CSP Backtracking": ["Kısıt Tatmin (AI)", "✅ %100 Kusursuz Garanti (İmkansızsa çözüm yok der)", "❌ Yok (İlk çizelgede kalır)", "Arama Ağacı (DFS)", "❌ Yok (Yalnız Feasible)", "⏱️ Milisaniyeler"],
        "Sekme 6: ILP / MILP": ["Matematiksel Programlama", "✅ %100 Kusursuz Garanti (İmkansızsa Infeasible der)", "🏆 Tam Optimizasyon (min Z*)", "Dal-Sınır (Branch & Bound)", "🏆 MATEMATİKSEL GARANTİ", "⏱️ Saniyeler"],
        "Sekme 7: Hill Climbing": ["Metasezgisel (Local Search)", "⚠️ Sezgisel Bağımlı (İmkansızsa İhlal İçerebilir)", "📈 İteratif Hamle İyileştirmesi", "Komşuluk Araması (Swap Move)", "⚠️ Yerel Optimum (Local Min)", "⚡ Çok Hızlı (İteratif)"]
    })

    st.dataframe(df_3way_hc, width="stretch", hide_index=True)

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Hill Climbing Local Search Grafikleri & Yakınsama Analizi")

    # 1. İTERASYON BAZLI CEZA PUANI DÜŞÜŞ YAKINSAMA GRAFİĞİ
    st.markdown("##### 1️⃣ İterasyon Bazlı Ceza Puanı Düşüş & Yakınsama Eğrisi (Convergence Curve)")
    score_hist = results.get('score_history', results.get('history', []))
    df_history = pd.DataFrame({
        "İterasyon": list(range(len(score_hist))),
        "Toplam Ceza Puanı (Z)": score_hist
    })

    fig_conv = px.line(
        df_history,
        x="İterasyon",
        y="Toplam Ceza Puanı (Z)",
        labels={"İterasyon": "Arama İterasyonu (K)", "Toplam Ceza Puanı (Z)": "Ceza Puanı (Skor)"},
        color_discrete_sequence=["#d97706"]
    )
    fig_conv.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
    st.plotly_chart(fig_conv, width="stretch", key="t7_fig_conv")

    # 2-6 STANDART ÇİZELGE ANALİTİKLERİ VE MATRİS TABLOSU
    render_standard_schedule_analytics(
        results, num_days, key_prefix="t7_hc",
        solver_name="Hill Climbing", start_chart_num=2
    )
