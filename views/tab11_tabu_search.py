"""
================================================================================
  VIEWS/TAB11_TABU_SEARCH.PY - SEKME 11: TABU SEARCH (TABU ARAMASI OPTİMİZASYONU)
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.tabu_search_solver import run_tabu_search
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

def render_tab11(params):
    """Sekme 11 içeriğini çizer: Tabu Search (Tabu Araması) Metasezgisel Çözücüsü ve Analitik Grafikler."""
    st.markdown("## 🤫 Sekme 11: Tabu Search (Tabu Araması & Hafıza Tabanlı Metasezgisel Optimizasyon)")

    # --- TEORİK BİLGİ KARTLARI ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #0284c7;">
<div class="card-title" style="color: #0369a1;">🧠 Kısa & Uzun Vadeli Hafıza Mekanizması (Tabu Listesi)</div>
<ul style="line-height: 1.6;">
<li><b>Tabu Listesi (Kısa Vadeli Hafıza):</b> Son ziyaret edilen veya değiştirilen çözümleri belirli bir süre (örneğin 15 adım) yasaklayan metasezgisel yöntemdir.</li>
<li><b>Tabu Tenure (L):</b> Yapılan bir vardiya takasının kaç iterasyon boyunca yasaklı kalacağını belirler. Arama uzayında aynı durumlara geri dönüp kilitlenmeyi (<b>Cycling / Döngü</b>) kesin olarak engeller.</li>
<li><b>Yönlendirilmiş Vadi Aşımı (Non-Monotonic Search):</b> Sadece iyileştiren hamleleri değil, yerel tuzaktan (Local Optimum) kurtulmak için geçici olarak kötüleşen hamleleri de kabul eder.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #7c3aed;">
<div class="card-title" style="color: #6d28d9;">🎯 Aspirasyon Kriteri (Aspiration Criterion) & Vadi Aşımı</div>
<ul style="line-height: 1.6;">
<li><b>Aspirasyon Kuralı:</b> Eğer tabu listesinde yasaklanmış bir hamle, şu ana kadar keşfedilmiş <b>en iyi global skordan (<i>Z</i><sub>best</sub>) daha iyi</b> bir sonuç üretiyorsa tabu kuralı delinir ve hamle derhal kabul edilir.</li>
<li><b>Küresel Optimum Güvencesi:</b> Bu kural sayesinde şampiyon bir çözüm asla yapay yasaklar yüzünden kaçırılmaz.</li>
<li><b>Deterministik Kararlılık:</b> Simulated Annealing gibi rastgele sıcaklık olasılıklarına güvenmek yerine, hafıza yönlendirmesiyle bilinçli ve sistematik arama yürütür.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- METASEZGİSEL (TABU SEARCH) ÜSTÜNLÜKLERİ PANOLARI ---
    st.markdown("### 🚀 Tabu Search Yönteminin Diğer Metasezgisellere Göre 4 Temel Üstünlüğü")

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        st.markdown("""<div style="background-color: #f0fdf4; border: 2px solid #059669; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #047857; margin-top: 0;">1. 🔄 Kesin Çevrim Engelleme (Cycle Avoidance)</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
Hill Climbing lokal tuzaklarda donar; Simulated Annealing ise aynı çukura tekrar düşebilir. <b>Tabu Search ise geçmiş hamleleri tabu listesinde hafızada tutarak aynı rotayı tekrar ziyaret etmeyi deterministik olarak engeller.</b>
</p>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background-color: #eff6ff; border: 2px solid #2563eb; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #1e40af; margin-top: 0;">2. ⚡ Akıllı ve Yönlendirilmiş Vadi Tırmanışı</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
Rastgele mutasyonlar yapmak yerine, her adımda geçerli en iyi komşuyu seçer. <b>Gerektiğinde ceza puanının artmasına izin vererek yerel minimum çukurundan hızla dışarı tırmanır.</b>
</p>
</div>""", unsafe_allow_html=True)

    with adv_col2:
        st.markdown("""<div style="background-color: #fef3c7; border: 2px solid #d97706; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #b45309; margin-top: 0;">3. 🎯 Aspirasyon Esnekliği (Hiçbir Fırsatı Kaçırmaz)</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
Klasik kısıtlayıcı algoritmaların aksine, <b>tarihi rekor kıran bir çözüm adayı ortaya çıktığında tabu yasağını anında geçersiz kılarak (Aspiration Override)</b> küresel en iyiye ulaşır.
</p>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background-color: #faf5ff; border: 2px solid #7c3aed; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #6d28d9; margin-top: 0;">4. ⏱️ Sanayide Anlık Vardiya Tamiri (Fast Repair)</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
Hastalık veya acil durumlarda çalışan izinleri bozulduğunda, <b>sadece problemli hücreleri tabu mekanizmasıyla yönlendirerek saniyeler içinde yeni dengeli çizelgeyi oluşturur.</b>
</p>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- PARAMETRE KONTROL PANELİ ---
    st.markdown("### 🎛️ Tabu Search Hafıza & Komşuluk Parametre Kontrol Paneli")

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        max_iter = st.slider("Maksimum İterasyon Sayısı (K)", 200, 3000, 750, 50, key="ts_iter")
    with p_col2:
        tabu_tenure = st.slider("Tabu Süresi / Yasak Adımı (L)", 3, 40, 15, 1, key="ts_tenure", help="Bir vardiya takasının tabu listesinde yasaklı kalacağı adım sayısı (Varsayılan: 15 adım)")
    with p_col3:
        neighborhood_size = st.slider("Komşuluk Örneklem Boyutu", 10, 40, 20, 5, key="ts_neighbor", help="Her iterasyonda değerlendirilecek aday takas sayısı")
    with p_col4:
        use_aspiration = st.checkbox("🎯 Aspirasyon Kriteri Aktif", value=True, key="ts_aspiration", help="En iyi skoru aşan tabu hamlelerin yasağını delerek kabul et.")
        use_diversification = st.checkbox("🌐 Uzun Vadeli Frekans Çeşitlendirmesi", value=False, key="ts_divers", help="Aşırı ziyaret edilen durumları cezalandırarak keşfi artır.")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # --- HESAPLAMA VE SÜRE TAHMİNİ PANELİ ---
    total_evaluations = max_iter * neighborhood_size
    est_ts_cpu_ms = round(total_evaluations * 0.28, 0)

    cost_ms = round(params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3)) * 0.6, 2) # NumPy C-hızı
    ts_total_ms = total_evaluations * cost_ms
    ts_time_str = f"{ts_total_ms/1000:.2f} sn" if ts_total_ms >= 1000 else f"{ts_total_ms:.0f} ms"

    st.markdown("#### 🧮 Tabu Arama & Komşuluk Değerlendirme Tahmin Paneli (Ön-Analiz)")

    pred_c1, pred_c2 = st.columns(2)
    with pred_c1:
        render_evaluator_cost_badge(total_evaluations, cost_ms, label="Tahmini Ceza Değerlendirme (K × N_size)")
    with pred_c2:
        st.metric(
            label="Tabu Listesi Yasaklama Süresi (L)",
            value=f"{tabu_tenure} İterasyon",
            help="Değiştirilen her işçi hücresinin kilitli kalacağı adım süresi."
        )

    st.divider()

    run_btn = st.button("🚀 Tabu Search Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t11")

    if not run_btn and "res_t11" not in st.session_state:
        st.warning("👈 Optimizasyonu başlatmak için yukarıdaki **'🚀 Tabu Search Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = params.get('live_stream_enabled', True)
        stream_interval = max(10, int(params.get('live_stream_interval', 50) / 2))
        
        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="Tabu Search Hafıza Tabanlı Arama",
            max_steps=max_iter,
            unit_name="İterasyon",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t11_tracker"
        )
        
        results = run_tabu_search(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            weights=weights,
            max_iterations=max_iter,
            tabu_tenure=tabu_tenure,
            neighborhood_size=neighborhood_size,
            use_aspiration=use_aspiration,
            use_diversification=use_diversification,
            seed=42,
            custom_workers=custom_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=stream_interval
        )
        tracker.finish(results.get('total_iterations', max_iter), results.get('final_score'))
        results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t11"] = results
    else:
        results = st.session_state["res_t11"]
        if params.get('live_stream_enabled', True) and 'stream_data' in results:
            render_live_stream_summary(results['stream_data'])

    # --- BİTİRME NEDENİ VE SERT KISIT UYGUNLUK BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")
    render_hard_constraints_status_card(
        results.get('is_feasible', True),
        results.get('hard_violations_count', 0),
        results.get('hard_violation_logs', []),
        solver_name="Tabu Search (Tabu Araması)",
        meta=results.get('meta')
    )

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Başlangıç Skoru</div>
        <div class="metric-value" style="color: #dc2626;">{results['initial_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Tabu En İyi Skor</div>
        <div class="metric-value" style="color: #059669;">{results['final_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İyileşme Oranı</div>
        <div class="metric-value" style="color: #1d4ed8;">%{results['improvement_rate']}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        is_feas = results.get('is_feasible', True)
        h_cnt = results.get('hard_violations_count', 0)
        feas_label = "✅ %100 GEÇERLİ" if is_feas else f"🚨 {h_cnt} İHLAL"
        feas_color = "#059669" if is_feas else "#dc2626"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Sert Kısıt Uygunluğu</div>
        <div class="metric-value" style="color: {feas_color}; font-size: 1.05rem;">{feas_label}</div>
        </div>""", unsafe_allow_html=True)

    meta = results.get('meta', {})

    with m5:
        tot_acc = meta.get('accepted_moves', results.get('accepted_moves', 0))
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Kabul Edilen</div>
        <div class="metric-value" style="color: #8b5cf6;">{tot_acc}</div>
        </div>""", unsafe_allow_html=True)

    with m6:
        asp_c = meta.get('aspiration_count', results.get('aspiration_count', 0))
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Aspirasyon</div>
        <div class="metric-value" style="color: #d97706;">{asp_c} Kez</div>
        </div>""", unsafe_allow_html=True)

    with m7:
        eval_c = results.get('eval_count', total_evaluations)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Ceza Çağrısı</div>
        <div class="metric-value" style="color: #6366f1;">{eval_c:,} Adet</div>
        </div>""", unsafe_allow_html=True)

    with m8:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Arama Süresi</div>
        <div class="metric-value" style="color: #059669;">{results['exec_time_ms']} ms</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Tabu Search Yakınsama & Analitik Grafikler")

    score_hist = results.get('score_history', results.get('best_score_history', []))
    iters = list(range(1, len(score_hist) + 1))
    curr_scores = meta.get('curr_score_history', results.get('curr_score_history', score_hist))
    tabu_sizes = meta.get('tabu_size_history', results.get('tabu_size_history', [0] * len(iters)))
    
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ İterasyon Bazlı Tabu Arama & Vadi Aşımı Grafiği (Best vs Current Score)")
        fig_ts_conv = go.Figure()
        fig_ts_conv.add_trace(go.Scatter(
            x=iters, y=score_hist,
            name="Global En İyi Skor Z_best",
            line=dict(color="#059669", width=2.5)
        ))
        fig_ts_conv.add_trace(go.Scatter(
            x=iters, y=curr_scores,
            name="Mevcut Çözüm Skoru Z_current (Vadi Çıkışları)",
            line=dict(color="#2563eb", width=1.5, dash="dot"),
            opacity=0.75
        ))
        fig_ts_conv.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            xaxis=dict(title="İterasyon (Adım)"),
            yaxis=dict(title="Toplam Ceza Skoru Z"),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_ts_conv, width="stretch", key="t11_fig_conv")

    with g_col2:
        st.markdown("##### 2️⃣ Aktif Tabu Listesi Boyutu & Aspirasyon Tetiklenme Noktaları")
        fig_tabu_size = go.Figure()
        fig_tabu_size.add_trace(go.Scatter(
            x=iters, y=tabu_sizes,
            name="Aktif Tabu Hücre Sayısı",
            line=dict(color="#d97706", width=2)
        ))
        asp_events = meta.get('aspiration_events', results.get('aspiration_events', []))
        if len(asp_events) > 0:
            asp_iters = []
            for ev in asp_events:
                it_num = ev['iteration'] if isinstance(ev, dict) else (ev + 1 if isinstance(ev, int) else None)
                if it_num is not None and 1 <= it_num <= len(tabu_sizes):
                    asp_iters.append(it_num)
            if len(asp_iters) > 0:
                asp_scores = [tabu_sizes[it - 1] for it in asp_iters]
                fig_tabu_size.add_trace(go.Scatter(
                    x=asp_iters, y=asp_scores,
                    mode="markers",
                    name="🎯 Aspirasyon Kriteriyle Tabu Delindi",
                    marker=dict(color="#dc2626", size=9, symbol="star")
                ))
        fig_tabu_size.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            xaxis=dict(title="İterasyon (Adım)"),
            yaxis=dict(title="Yasaklı Hücre Sayısı (Tabu Boyutu)"),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_tabu_size, width="stretch", key="t11_fig_tabu_size")

    # 3-7 STANDART ÇİZELGE ANALİTİKLERİ VE MATRİS TABLOSU
    render_standard_schedule_analytics(
        results, num_days, key_prefix="t11_ts",
        solver_name="Tabu Search", start_chart_num=3
    )
