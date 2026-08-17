"""
================================================================================
  VIEWS/TAB10_MEMETIC_ALGORITHM.PY - SEKME 10: MEMETİK ALGORİTMA (HIBRİT OPTİMİZASYON)
================================================================================
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.memetic_algorithm_solver import run_memetic_algorithm
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

def render_tab10(params):
    """Sekme 10 içeriğini çizer: Memetik Algoritma (MA) Hibrit Evrimsel Çözücü ve Analitik Grafikler."""
    st.markdown("## 🧬 Sekme 10: Memetic Algorithm (Hibrit Genetik + Yerel Tırmanma Çözücüsü)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb;">
<div class="card-title" style="color: #1d4ed8;">🧠 Genetik Arama + Lokal Cerrahi Tamir (Memetik Mantığı)</div>
<ul style="line-height: 1.6;">
<li><b>Global Arama (Genetik Algoritma):</b> Popülasyon tabanlı evrimsel arama ile geniş çözüm uzayındaki küresel minimum potansiyeline sahip bölgeleri keşfeder.</li>
<li><b>Lokal Cerrahi Tamir (Hill Climbing):</b> Çaprazlama operatörünün dikiş günlerinde kırdığı sirkadiyen ritim ve posta bütünlüğü kurallarını <i>nokta atışı mikro takaslar</i> ile tamir eder.</li>
<li><b>Evrimsel Uygunluk (Fitness):</b> Her kromozomun ceza puanı (Z) hesaplanır. Lokal tamir gören bireyler daha yüksek evrimsel uygunluğa kavuşur.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #059669;">
<div class="card-title" style="color: #047857;">🏆 Neden Saf Genetik Algoritmadan Çok Daha Üstündür?</div>
<ul style="line-height: 1.6;">
<li><b>Çaprazlama Tahribatını Önleme:</b> Saf GA'nın kromozomları rastgele kesip birleştirmesinden doğan dikiş hataları Memetik Lokal Arama ile anında giderilir.</li>
<li><b>Erken Yakınsamayı Engelleme:</b> Popülasyon yerel tuzaklara takılmadan sürekli iyileşmeye devam eder.</li>
<li><b>En Düşük Ceza Skoru:</b> Hem Hill Climbing'in yerel tamir gücünü hem de GA'nın küresel arama genişliğini tek potada birleştirir.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- PARAMETRE KONTROL PANELİ ---
    st.markdown("### 🎛️ Memetik Algoritma Hibrit Parametre Kontrol Paneli")
    
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
    with p_col1:
        pop_size = st.slider("Popülasyon Büyüklüğü (P)", 20, 200, 50, 10, key="ma_pop")
    with p_col2:
        generations = st.slider("Jenerasyon Sayısı (G)", 20, 300, 100, 10, key="ma_gen")
    with p_col3:
        crossover_rate = st.slider("Çaprazlama Oranı (p_c)", 0.50, 1.00, 0.85, 0.05, format="%.2f", key="ma_pc")
    with p_col4:
        mutation_rate = st.slider("Mutasyon Oranı (p_m)", 0.01, 0.30, 0.05, 0.01, format="%.2f", key="ma_pm")
    with p_col5:
        local_depth = st.slider("Lokal Tamir Derinliği (k)", 1, 20, 5, 1, key="ma_depth", help="Çaprazlama sonrası çocuğa uygulanacak Hill Climbing mikro-takas adım sayısı.")
    with p_col6:
        elitism_count = st.slider("Elitizm Sayısı (e)", 1, 10, 2, 1, key="ma_elitism")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # --- HESAPLAMA VE SÜRE TAHMİNİ PANELİ ---
    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    # Her jenerasyonda: Popülasyon skorlaması (P) + Çocuk başına lokal tırmanma [(P - E) * (1 + 0.35 * Derinlik)]
    total_evaluations = int(pop_size + generations * (pop_size + (pop_size - elitism_count) * (1.0 + local_depth * 0.35)))
    ma_est_calls = total_evaluations
    ma_total_ms = ma_est_calls * cost_ms
    ma_time_str = f"{ma_total_ms/1000:.2f} sn" if ma_total_ms >= 1000 else f"{ma_total_ms:.0f} ms"

    st.markdown("#### 🧮 Evrimsel & Hibrit Değerlendirme Tahmin Paneli (Ön-Analiz)")

    pred_c1, pred_c2 = st.columns(2)
    with pred_c1:
        render_evaluator_cost_badge(ma_est_calls, cost_ms, label="Tahmini Ceza Değerlendirme (GA + Tamir)")
    with pred_c2:
        st.metric(
            label="Elitizm Koruma Oranı",
            value=f"%{round((elitism_count / pop_size) * 100, 1)}",
            help="Her nesilde korunup doğrudan aktarılan şampiyon kromozom oranı."
        )

    st.divider()

    run_btn = st.button("🚀 Memetik Algoritma Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t10")

    if not run_btn and "res_t10" not in st.session_state:
        st.warning("👈 Hibrit aramayı başlatmak için yukarıdaki **'🚀 Memetik Algoritma Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = params.get('live_stream_enabled', True)
        stream_interval = max(1, int(params.get('live_stream_interval', 50) / 25))
        
        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="Memetik Algoritma Hibrit Arama",
            max_steps=generations,
            unit_name="Nesil",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t10_tracker"
        )
        
        results = run_memetic_algorithm(
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
            local_search_depth=local_depth,
            elitism_count=elitism_count,
            seed=42,
            custom_workers=custom_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=stream_interval
        )
        tracker.finish(generations, results.get('final_score'))
        results['generations_run'] = generations
        results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t10"] = results
    else:
        results = st.session_state["res_t10"]
        if params.get('live_stream_enabled', True) and 'stream_data' in results:
            render_live_stream_summary(results['stream_data'])

    # --- BİTİRME NEDENİ VE SERT KISIT UYGUNLUK BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")
    render_hard_constraints_status_card(
        results.get('is_feasible', True),
        results.get('hard_violations_count', 0),
        results.get('hard_violation_logs', []),
        solver_name="Memetik Algoritma (MA)"
    )

    # --- METRİK KARTLARI ---
    render_metaheuristic_metric_cards(results, move_label="Jenerasyon", is_population=True)

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Memetik Algoritma Evrimsel Yakınsama & Analitik Grafikler")

    score_hist = results.get('score_history', results.get('best_score_history', []))
    gens = list(range(1, len(score_hist) + 1))
    meta = results.get('meta', {})
    avg_scores = meta.get('avg_score_history', results.get('avg_score_history', score_hist))
    div_scores = meta.get('diversity_history', results.get('diversity_history', [0.0] * len(gens)))
    
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ Jenerasyon Bazlı Memetik Yakınsama Grafiği (Best vs Avg Score)")
        fig_ma_conv = go.Figure()
        fig_ma_conv.add_trace(go.Scatter(
            x=gens, y=score_hist,
            name="Memetik En İyi Birey Skoru Z_best",
            line=dict(color="#059669", width=2.5)
        ))
        fig_ma_conv.add_trace(go.Scatter(
            x=gens, y=avg_scores,
            name="Popülasyon Ortalama Skoru Z_avg",
            line=dict(color="#2563eb", width=2, dash="dash")
        ))
        fig_ma_conv.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            xaxis=dict(title="Jenerasyon (Nesil)"),
            yaxis=dict(title="Toplam Ceza Skoru Z"),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_ma_conv, width="stretch", key="t10_fig_conv")

    with g_col2:
        st.markdown("##### 2️⃣ Popülasyon Genetik Çeşitlilik İndeksi (Standart Sapma)")
        fig_div = px.line(
            x=gens, y=div_scores,
            labels={"x": "Jenerasyon", "y": "Popülasyon Çeşitliliği (Std Dev)"},
            line_shape="linear"
        )
        fig_div.update_traces(line_color="#2563eb", line_width=2.5)
        fig_div.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_div, width="stretch", key="t10_fig_div")

    # 3-7 STANDART ÇİZELGE ANALİTİKLERİ VE MATRİS TABLOSU
    render_standard_schedule_analytics(
        results, num_days, key_prefix="t10_ma",
        solver_name="Memetik Algoritma", start_chart_num=3
    )
