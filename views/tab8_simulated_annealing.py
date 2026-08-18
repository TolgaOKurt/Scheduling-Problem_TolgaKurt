"""
================================================================================
  VIEWS/TAB8_SIMULATED_ANNEALING.PY - SEKME 8: SIMULATED ANNEALING (TAVLAMA BENZETİMİ)
================================================================================
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.simulated_annealing_solver import run_simulated_annealing
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

def render_tab8(params):
    """Sekme 8 içeriğini çizer: Simulated Annealing (Tavlama Benzetimi) Metasezgisel Çözücüsü ve Grafikler."""
    st.markdown("## ♨️ Sekme 8: Simulated Annealing (Tavlama Benzetimi & Stokastik Arama)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #ea580c;">
<div class="card-title" style="color: #c2410c;">🔥 Metalurjik Tavlama Analojisi & Sıcaklık (T)</div>
<ul>
<li><b>Yüksek Sıcaklıkta Esneklik (Exploration):</b> Yüksek sıcaklıklarda (T &gt;&gt; 0) algoritma arama uzayını genişçe keşfeder.</li>
<li><b>Metropolis Kriteri:</b> P = exp(-&Delta;Z / T) formülü sayesinde <b>kötüleşen (ceza puanını artıran) hamleleri bile olasılıksal olarak kabul eder.</b></li>
<li><b>Tuzaklardan Kaçış:</b> Bu stokastik sıçramalar sayesinde Hill Climbing'in kilitlendiği yerel minimum (local optimum) çukurlarından kurtulur.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #0284c7;">
<div class="card-title" style="color: #0369a1;">❄️ Geometrik Soğuma (&alpha;) & Kararlı Yerleşim</div>
<ul>
<li><b>Aşamalı Soğuma (Cooling Schedule):</b> Sıcaklık her adımda T<sub>k+1</sub> = &alpha; &middot; T<sub>k</sub> formülü ile yavaşça düşürülür (&alpha; &isin; [0.85, 0.999]).</li>
<li><b>Düşük Sıcaklıkta Kararlılık (Exploitation):</b> Sıcaklık sıfıra yaklaştıkça (T &rarr; 0) kötü hamle kabul olasılığı sıfırlanır ve algoritma Hill Climbing'e dönüşerek en iyi vadiye yerleşir.</li>
<li><b>Denge:</b> Keşif (Exploration) ve Odaklanma (Exploitation) arasındaki mükemmel dengeyi kurar.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- PARAMETRE KONTROL PANELİ ---
    st.markdown("### 🎛️ Simulated Annealing Parametre Kontrol Paneli")
    
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        t_start = st.slider("Başlangıç Sıcaklığı (T_start)", 100.0, 5000.0, 1000.0, 100.0, key="sa_tstart")
    with p_col2:
        t_min = st.select_slider("Minimum Sıcaklık (T_min)", options=[0.1, 0.05, 0.01, 0.005, 0.001], value=0.01, key="sa_tmin")
    with p_col3:
        cooling_rate = st.slider("Soğuma Katsayısı (α)", 0.9800, 0.9999, 0.9900, 0.0005, format="%.4f", key="sa_alpha")
    with p_col4:
        max_iter = st.slider("Maksimum İterasyon (K)", 2000, 31000, 3000, 1000, key="sa_iter")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # --- MATEMATİKSEL İTERASYON TAHMİN HESAPLAYICISI (ADIM ADIM KESİN FORMÜL) ---
    t_start_val = float(t_start)
    t_min_val = float(t_min)
    cooling_val = float(cooling_rate)

    # Matematiksel Denklem: k_soğuma = ceil( ln(T_min / T_start) / ln(alpha) )
    num_ratio = t_min_val / t_start_val
    log_num = float(np.log(num_ratio))
    log_den = float(np.log(cooling_val))
    
    k_cooling_needed = int(np.ceil(log_num / log_den))
    est_cooling_cpu_ms = round(k_cooling_needed * 1.5, 1)

    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    # Komşuluk takaslarında farklı vardiya ve sert kısıt uygunluk olasılığı ~%28'dir
    sa_est_calls = int(min(k_cooling_needed, max_iter) * 0.28)
    sa_total_ms = sa_est_calls * cost_ms
    sa_time_str = f"{sa_total_ms/1000:.2f} sn" if sa_total_ms >= 1000 else f"{sa_total_ms:.0f} ms"

    st.markdown("#### 🧮 Sıcaklık Parametrelerine Göre İterasyon Tahmin Paneli (Ön-Analiz)")

    pred_c1, pred_c2, pred_c3 = st.columns(3)
    with pred_c1:
        st.metric(
            label="Gerekli Soğuma Adımı",
            value=f"{k_cooling_needed:,} İterasyon",
            help="Sıcaklığın T_start değerinden T_min değerine düşmesi için gereken matematiksel net adım sayısı."
        )
    with pred_c2:
        render_evaluator_cost_badge(sa_est_calls, cost_ms, label="Tahmini Ceza Değerlendirme")
    with pred_c3:
        st.metric(
            label="İterasyon Sınırı (K_max)",
            value=f"{max_iter:,} İterasyon",
            delta=f"{max_iter - k_cooling_needed:+,} Adım Marj",
            delta_color="normal" if k_cooling_needed <= max_iter else "inverse"
        )

    with st.expander("📐 Matematiksel Formül ve Adım Adım İterasyon Hesabı Detayı", expanded=False):
        st.latex(r"k_{\text{soğuma}} = \left\lceil \frac{\ln\left(\frac{T_{\text{min}}}{T_{\text{başlangıç}}}\right)}{\ln(\alpha)} \right\rceil")
        st.write(f"- **Başlangıç Sıcaklığı (T_başlangıç):** `{t_start_val}`")
        st.write(f"- **Hedef Bitiş Sıcaklığı (T_min):** `{t_min_val}`")
        st.write(f"- **Soğuma Katsayısı (α):** `{cooling_val}`")
        st.write(f"- **Pay ln(T_min / T_başlangıç):** `ln({num_ratio})` = `{log_num:.6f}`")
        st.write(f"- **Payda ln(α):** `ln({cooling_val})` = `{log_den:.6f}`")
        st.write(f"- **Hesaplanan Net Adım Sayısı:** `{log_num:.6f} / {log_den:.6f}` = `{log_num/log_den:.2f}` → **{k_cooling_needed:,} İterasyon**")

    if k_cooling_needed <= max_iter:
        st.success(f"✅ **Soğuma Başarıyla Tamamlanacak:** Seçilen sıcaklık parametreleri ile algoritma **{k_cooling_needed:,}. adımda** hedef T <= {t_min} sıcaklığına ulaşarak soğumayı bitirecektir. Girilen K={max_iter:,} limiti soğumaya izin vermektedir.")
    else:
        st.warning(f"⚠️ **İterasyon Sınırı Sıcaklığı Yarıda Kesecek:** Tam soğuma için **{k_cooling_needed:,} adım** gerekirken, K={max_iter:,} limiti seçildiği için algoritma tam soğumadan {max_iter}. adımda kesilecektir! (Tam soğuma için K >= {k_cooling_needed:,} yapabilirsiniz).")

    st.divider()

    run_btn = st.button("🚀 Simulated Annealing Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t8")

    if not run_btn and "res_t8" not in st.session_state:
        st.warning("👈 Optimizasyonu başlatmak için yukarıdaki **'🚀 Simulated Annealing Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = params.get('live_stream_enabled', True)
        stream_interval = params.get('live_stream_interval', 50)
        
        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="Simulated Annealing Stokastik Tavlama",
            max_steps=max_iter,
            unit_name="İterasyon",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t8_tracker"
        )
        
        results = run_simulated_annealing(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            weights=weights,
            t_start=t_start,
            t_min=t_min,
            cooling_rate=cooling_rate,
            max_iterations=max_iter,
            seed=42,
            custom_workers=custom_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=stream_interval
        )
        tracker.finish(results.get('total_iterations', max_iter), results.get('final_score'))
        results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t8"] = results
    else:
        results = st.session_state["res_t8"]
        if params.get('live_stream_enabled', True) and 'stream_data' in results:
            render_live_stream_summary(results['stream_data'])

    # --- BİTİRME NEDENİ VE SERT KISIT UYGUNLUK BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")
    render_hard_constraints_status_card(
        results.get('is_feasible', True),
        results.get('hard_violations_count', 0),
        results.get('hard_violation_logs', []),
        solver_name="Simulated Annealing (Tavlama Benzetimi)",
        meta=results.get('meta')
    )

    meta = results.get('meta', {})

    # --- METRİK KARTLARI ---
    render_metaheuristic_metric_cards(results, move_label="Kabul Edilen Kötü Hamle")

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Simulated Annealing İterasyon, Sıcaklık & Yakınsama Grafikleri")

    # 1. ÇİFT EKSENLİ İTERASYON BAZLI CEZA PUANI & SICAKLIK DÜŞÜŞ GRAFİĞİ
    st.markdown("##### 1️⃣ İterasyon Bazlı Anlık Skor, En İyi Skor ve Sıcaklık T(k) Yakınsama Grafiği (Double Y-Axis)")
    
    score_hist = results.get('score_history', results.get('best_score_history', []))
    curr_scores = meta.get('curr_score_history', results.get('curr_score_history', score_hist))
    temp_hist = meta.get('temp_history', results.get('temp_history', [1000.0] * len(curr_scores)))
    iters = list(range(len(curr_scores)))
    fig_sa_conv = go.Figure()

    fig_sa_conv.add_trace(go.Scatter(
        x=iters,
        y=curr_scores,
        mode="lines",
        name="Anlık Aday Skor Z_curr",
        line=dict(color="#94a3b8", width=1)
    ))

    fig_sa_conv.add_trace(go.Scatter(
        x=iters,
        y=score_hist,
        mode="lines",
        name="Tarihi En İyi Skor Z_best",
        line=dict(color="#059669", width=2.5)
    ))

    fig_sa_conv.add_trace(go.Scatter(
        x=iters,
        y=temp_hist,
        mode="lines",
        name="Sıcaklık T(k)",
        line=dict(color="#0284c7", width=2, dash="dash"),
        yaxis="y2"
    ))

    fig_sa_conv.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=400,
        xaxis=dict(title=dict(text="Arama İterasyonu (K)")),
        yaxis=dict(title=dict(text="Toplam Ceza Skoru Z(k)", font=dict(color="#059669")), tickfont=dict(color="#059669")),
        yaxis2=dict(title=dict(text="Sıcaklık T(k)", font=dict(color="#0284c7")), tickfont=dict(color="#0284c7"), overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15)
    )
    st.plotly_chart(fig_sa_conv, width="stretch", key="t8_fig_sa_conv")

    # 2. SICAKLIK SOĞUMA EĞRİSİ VE METROPOLIS KABUL ORANI DİNAMİĞİ
    st.markdown("##### 2️⃣ Sıcaklık Soğuma ve Dinamik Metropolis Kabul Oranı Eğrisi (Temperature Decay vs Acceptance Rate)")
    probs = meta.get('acceptance_probs', [])
    if probs and len(probs) == len(iters):
        # Kayan ortalama kabul olasılığı (rolling 50-step window)
        window = min(50, max(1, len(probs) // 10))
        prob_series = pd.Series(probs).rolling(window=window, min_periods=1).mean() * 100
        
        df_decay = pd.DataFrame({
            "İterasyon": iters,
            "Sıcaklık (T)": temp_hist,
            "Kabul Olasılığı P(%)": prob_series
        })
        
        fig_decay = go.Figure()
        fig_decay.add_trace(go.Scatter(
            x=iters,
            y=temp_hist,
            mode="lines",
            name="Sıcaklık T(k) (°C)",
            line=dict(color="#ea580c", width=2.5)
        ))
        fig_decay.add_trace(go.Scatter(
            x=iters,
            y=prob_series,
            mode="lines",
            name=f"Metropolis Kabul Oranı P(ΔZ, T) % ({window} Adım Kayan Ort.)",
            line=dict(color="#7c3aed", width=2, dash="dot"),
            yaxis="y2"
        ))
        fig_decay.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=350,
            xaxis=dict(title=dict(text="Arama İterasyonu (K)")),
            yaxis=dict(title=dict(text="Sıcaklık T (°C)", font=dict(color="#ea580c")), tickfont=dict(color="#ea580c")),
            yaxis2=dict(title=dict(text="Kabul Olasılığı P (%)", font=dict(color="#7c3aed")), tickfont=dict(color="#7c3aed"), overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_decay, width="stretch", key="t8_fig_decay")

    # 3-7 STANDART ÇİZELGE ANALİTİKLERİ VE MATRİS TABLOSU
    render_standard_schedule_analytics(
        results, num_days, key_prefix="t8_sa",
        solver_name="Simulated Annealing", start_chart_num=3
    )
