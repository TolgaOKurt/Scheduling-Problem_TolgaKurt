"""
================================================================================
  VIEWS/TAB9_GENETIC_ALGORITHM.PY - SEKME 9: GENETİK ALGORİTMA (EVRİMSEL OPTİMİZASYON)
================================================================================
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.genetic_algorithm_solver import run_genetic_algorithm
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

def render_tab9(params):
    """Sekme 9 içeriğini çizer: Genetik Algoritma (GA) Evrimsel Çözücüsü ve Analitik Grafikler."""
    st.markdown("## 🧬 Genetic Algorithm (Genetik Algoritma & Evrimsel Arama)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #8b5cf6;">
<div class="card-title" style="color: #6d28d9;">🧬 Biyolojik Evrim Analojisi & Kromozom Yapısı</div>
<ul>
<li><b>Kromozom (Genotip):</b> Vardiya matrisi (İşçi Sayısı × Gün Sayısı) boyutlu bir kromozom olarak dizilenir. Her gen, bir çalışanın o günkü vardiyasını temsil eder.</li>
<li><b>Uygunluk Fonksiyonu (Fitness):</b> Popülasyondaki her kromozomun ceza puanı (Z) hesaplanır. Düşük ceza puanı = Yüksek Evrimsel Uygunluk (Fitness).</li>
<li><b>Turnuva Seçimi (Selection):</b> Bir sonraki jenerasyona ebeveyn olmak üzere en kaliteli kromozomlar rastgele turnuvalarla seçilir.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #ec4899;">
<div class="card-title" style="color: #be185d;">🧬 Çaprazlama (Crossover), Mutasyon & Elitizm</div>
<ul>
<li><b>Çaprazlama (Rekombinasyon):</b> Seçilen iki başarılı ebeveynin vardiya blokları rastgele belirlenen bir gün kesim noktasından birleştirilerek çocuk kromozom üretilir.</li>
<li><b>Mutasyon (Genetik Çeşitlilik):</b> Belirlenen mutasyon olasılığı ile rastgele günlerde personel vardiyaları takas edilerek yerel tuzaklardan kaçılır.</li>
<li><b>Elitizm:</b> En düşük ceza puanına sahip seçkin şampiyon kromozomlar bozulmadan doğrudan bir sonraki jenerasyona aktarılır.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- PARAMETRE KONTROL PANELİ ---
    st.markdown("### 🎛️ Genetik Algoritma Evrimsel Parametre Kontrol Paneli")
    
    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        pop_size = st.slider("Popülasyon Büyüklüğü (P)", 50, 200, 60, 10, key="ga_pop")
    with p_col2:
        generations = st.slider("Jenerasyon Sayısı (G)", 20, 600, 100, 10, key="ga_gen")
    with p_col3:
        crossover_rate = st.slider("Çaprazlama Oranı (p_c)", 0.50, 1.00, 0.85, 0.05, format="%.2f", key="ga_pc")
    with p_col4:
        mutation_rate = st.slider("Mutasyon Oranı (p_m)", 0.01, 0.30, 0.05, 0.01, format="%.2f", key="ga_pm")
    with p_col5:
        elitism_count = st.slider("Elitizm Sayısı (e)", 1, 10, 2, 1, key="ga_elitism")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # --- MATEMATİKSEL ÖN-ANALİZ TAHMİN PANELİ ---
    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    # Her nesilde tüm popülasyon (P) + başlangıç (P) + bitiş (1) değerlendirilir
    total_evaluations = (generations + 1) * pop_size + 1
    ga_total_ms = total_evaluations * cost_ms
    ga_time_str = f"{ga_total_ms/1000:.2f} sn" if ga_total_ms >= 1000 else f"{ga_total_ms:.0f} ms"

    st.markdown("#### 🧮 Evrimsel Hesaplama & Ceza Değerlendirici Tahmin Paneli (Ön-Analiz)")

    pred_c1, pred_c2 = st.columns(2)
    with pred_c1:
        render_evaluator_cost_badge(total_evaluations, cost_ms, label="Tahmini Ceza Değerlendirme (P × (1+G))")
    with pred_c2:
        st.metric(
            label="Elitizm Koruma Oranı",
            value=f"%{round((elitism_count / pop_size) * 100, 1)}",
            help="Her nesilde korunup doğrudan aktarılan şampiyon kromozom oranı."
        )

    st.divider()

    run_btn = st.button("🚀 Genetik Algoritma Evrimini Başlat", type="primary", width="stretch", key="btn_run_t9")

    if not run_btn and "res_t9" not in st.session_state:
        st.warning("👈 Evrimsel aramayı başlatmak için yukarıdaki **'🚀 Genetik Algoritma Evrimini Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = params.get('live_stream_enabled', True)
        stream_interval = max(1, int(params.get('live_stream_interval', 50) / 25))
        
        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="Genetik Algoritma Evrimsel Arama",
            max_steps=generations,
            unit_name="Nesil",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t9_tracker"
        )
        
        results = run_genetic_algorithm(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            weights=weights,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_count=elitism_count,
            seed=42,
            custom_workers=custom_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=stream_interval
        )
        tracker.finish(generations, results.get('final_score'))
        results['generations_run'] = generations
        results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t9"] = results
    else:
        results = st.session_state["res_t9"]
        if params.get('live_stream_enabled', True) and 'stream_data' in results:
            render_live_stream_summary(results['stream_data'])

    # --- BİTİRME NEDENİ VE SERT KISIT UYGUNLUK BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")
    render_hard_constraints_status_card(
        results.get('is_feasible', True),
        results.get('hard_violations_count', 0),
        results.get('hard_violation_logs', []),
        solver_name="Genetik Algoritma (GA)"
    )

    # --- METRİK KARTLARI ---
    render_metaheuristic_metric_cards(results, move_label="Jenerasyon", is_population=True)

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Genetik Algoritma Evrimsel Yakınsama & Analitik Grafikler")

    score_hist = results.get('score_history', results.get('best_score_history', []))
    gens = list(range(1, len(score_hist) + 1))
    meta = results.get('meta', {})
    avg_scores = meta.get('avg_score_history', results.get('avg_score_history', score_hist))
    div_scores = meta.get('diversity_history', results.get('diversity_history', [0.0] * len(gens)))
    
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ Jenerasyon Bazlı Evrimsel Yakınsama Grafiği (Best vs Avg Score)")
        fig_ga_conv = go.Figure()
        fig_ga_conv.add_trace(go.Scatter(
            x=gens, y=score_hist,
            name="En İyi Birey Skoru Z_best",
            line=dict(color="#059669", width=2.5)
        ))
        fig_ga_conv.add_trace(go.Scatter(
            x=gens, y=avg_scores,
            name="Popülasyon Ortalama Skoru Z_avg",
            line=dict(color="#8b5cf6", width=2, dash="dash")
        ))
        fig_ga_conv.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            xaxis=dict(title="Jenerasyon (Nesil)"),
            yaxis=dict(title="Toplam Ceza Skoru Z"),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_ga_conv, width="stretch", key="t9_fig_conv")

    with g_col2:
        st.markdown("##### 2️⃣ Popülasyon Genetik Çeşitlilik İndeksi (Standart Sapma)")
        fig_div = px.line(
            x=gens, y=div_scores,
            labels={"x": "Jenerasyon", "y": "Popülasyon Çeşitliliği (Std Dev)"},
            line_shape="linear"
        )
        fig_div.update_traces(line_color="#ec4899", line_width=2.5)
        fig_div.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_div, width="stretch", key="t9_fig_div")

    # 3-7 STANDART ÇİZELGE ANALİTİKLERİ VE MATRİS TABLOSU
    render_standard_schedule_analytics(
        results, num_days, key_prefix="t9_ga",
        solver_name="Genetik Algoritma", start_chart_num=3
    )
