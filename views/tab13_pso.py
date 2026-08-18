"""
================================================================================
  VIEWS/TAB13_PSO.PY - SEKME 13: PARTICLE SWARM OPTIMIZATION (DISCRETE PSO)
================================================================================
  Bu modül, Parçacık Sürü Optimizasyonu (Particle Swarm Optimization - PSO)
  ve Kesikli Sürü Zekası (Discrete PSO) metasezgisel optimizasyon algoritmasının
  arayüzünü, matematiksel ilkelerini, bilişsel/sosyal dinamiklerini ve
  canlı analitik görselleştirmelerini sunar.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.pso_solver import run_particle_swarm_optimization
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


def render_tab13(params):
    """Sekme 13 içeriğini çizer: Particle Swarm Optimization (Discrete PSO) Sürü Zekası Çözücüsü ve Analiz Panelleri."""
    st.markdown("## 🐝 Sekme 13: Particle Swarm Optimization (Discrete PSO - Parçacık Sürü Zekası)")
    st.markdown("""
    <div style="background-color: #fefce8; border-left: 5px solid #eab308; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; color: #854d0e;">
        📌 <b>Parçacık Sürü Zekası Paradigması (Kennedy & Eberhart, 1995; Clerc, 2002):</b> Kuş ve balık sürülerinin kolektif hareketinden ilham alan sürü zekası algoritmasıdır. 
        Her parçacık kendi bireysel en iyi konumu (<b><i>p</i><sub>best</sub></b> - Bilişsel Hafıza) ve tüm sürünün ortak küresel en iyi konumu (<b><i>g</i><sub>best</sub></b> - Sosyal Hafıza) 
        doğrultusunda hız ve rotasını güncelleyerek arama uzayını tarar.
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 1. TEORİK BİLGİ VE MATEMATİKSEL MODEL KARTLARI
    # ==============================================================================
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #eab308;">
            <div class="card-title" style="color: #a16207;">🐝 3 Temel Sürü Bileşeni ve Dinamiği</div>
            <ul style="line-height: 1.65; font-size: 0.92rem; color: #1e293b;">
                <li><b>1. Atalet / Momentum (Inertia - <i>w</i>):</b> Parçacığın önceki hareket yönünü ve hızını koruma eğilimidir. Sürünün yeni bölgeleri keşfetmesini (Exploration) sağlar.</li>
                <li><b>2. Bilişsel Çekim (Cognitive Component - <i>c</i><sub>1</sub>):</b> Parçacığın kendi geçmişindeki en başarılı çizelgeye (<i>p</i><sub>best</sub>) olan sadakati ve özgüvenidir.</li>
                <li><b>3. Sosyal Çekim (Social Component - <i>c</i><sub>2</sub>):</b> Sürünün lideri olan en başarılı çözüme (<i>g</i><sub>best</sub>) doğru çekilme ve sürüyle hizalanma eğilimidir (Exploitation).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #2563eb;">
            <div class="card-title" style="color: #1d4ed8;">🏗️ Vardiya Problemi İçin Kesikli (Discrete) PSO Dönüşümü</div>
            <ul style="line-height: 1.65; font-size: 0.92rem; color: #1e293b;">
                <li><b>Fark Takas Dizisi (Difference Swap Sequence):</b> Sürekli uzaydaki çıkarma işlemi yerine, iki çizelge arasındaki fark (<i>p</i><sub>best</sub> &ominus; <i>X<sub>i</sub></i>) ve (<i>g</i><sub>best</sub> &ominus; <i>X<sub>i</sub></i>) takas operatörleri listesi olarak çıkarılır.</li>
                <li><b>Hız Vektörü (Velocity <i>V<sub>i</sub></i>):</b> Uygulanması hedeflenen sert kısıt korumalı vardiya takaslarının öncelik havuzudur.</li>
                <li><b>Konum Güncellemesi (<i>X<sub>i</sub></i> &oplus; <i>V<sub>i</sub></i>):</b> Hız vektöründeki takaslar <i>check_swap_feasibility</i> kontrolüyle uygulanarak kısıtların %100 geçerli kalması sağlanır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Matematiksel Formül Gösterimi (LaTeX)
    st.markdown("##### 🧮 Kesikli Parçacık Sürü Zekası Matematiksel Güncelleme Denklemleri:")
    st.latex(r"V_i(t+1) = w(t) \cdot V_i(t) \;\oplus\; c_1 r_1 \cdot \left( p_{\text{best}, i} \ominus X_i(t) \right) \;\oplus\; c_2 r_2 \cdot \left( g_{\text{best}} \ominus X_i(t) \right)")
    st.latex(r"X_i(t+1) = X_i(t) \;\oplus\; V_i(t+1)")

    st.caption("Burada $w(t)$ dinamik atalet katsayısı, $r_1, r_2 \sim U(0,1)$ stokastik rastgele sayılar, $\ominus$ fark takas çıkarma operatörü ve $\oplus$ kısıt korumalı takas uygulama operatörüdür.")

    st.divider()

    # ==============================================================================
    # 2. GÜÇLÜ YANLAR, GÜÇSÜZ YANLAR VE KARŞILAŞTIRMA
    # ==============================================================================
    st.markdown("### ⚖️ Discrete PSO'nun Güçlü Yanları, Sınırları ve Diğer Algoritmalarla Kıyaslaması")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div style="background-color: #fefce8; border: 2px solid #ca8a04; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
            <h4 style="color: #a16207; margin-top: 0;">🌟 PSO'nun En Güçlü Yanları (Strengths)</h4>
            <ul style="font-size: 0.9rem; color: #1e293b; line-height: 1.6; padding-left: 18px; margin-bottom: 0;">
                <li><b>Kolektif Sürü Hafızası:</b> Tek çözümlü yöntemler (HC, SA, TS) tek bir patikada yürürken; PSO, 30 parçacığın ortak hafızası ile tüm uzayı aynı anda kuşatır.</li>
                <li><b>Dikiş Hatası Olmayan Çaprazlama:</b> Genetik Algoritmada (GA) gün bazlı kesim noktalarında sirkadiyen ritim kırılabilirken; Discrete PSO hedef yönelimli takas dizileri ile sert kısıtları bozmadan yönlenir.</li>
                <li><b>Hızlı ve Pürüzsüz Yakınsama:</b> <i>g</i><sub>best</sub> çekimi sayesinde sürü, arama uzayındaki en umut verici vadiye çok hızlı toplanır.</li>
                <li><b>Dinamik Keşif/Sömürü Dengesi:</b> Doğrusal azalan atalet ağırlığı <i>w(t)</i> ile arama başında küresel keşif, arama sonunda ise ince ayar sömürü yapılır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
            <h4 style="color: #b91c1c; margin-top: 0;">⚠️ PSO'nun Güçsüz / Dikkat Edilmesi Gereken Yanları (Limitations)</h4>
            <ul style="font-size: 0.9rem; color: #1e293b; line-height: 1.6; padding-left: 18px; margin-bottom: 0;">
                <li><b>Erken Yakınsama (Premature Convergence) Riski:</b> Sürüdeki parçacıklar güçlü bir yerel minimuma çok erken kapılırsa, çeşitlilik kaybolup arama kilitlenebilir (Türbülans mekanizması ile aşılır).</li>
                <li><b>Parametre Duyarlılığı:</b> <i>w</i>, <i>c</i><sub>1</sub> ve <i>c</i><sub>2</sub> katsayıları arasındaki denge bozulursa sürü ya dağılır ya da tek noktada donar.</li>
                <li><b>Kombinatoryal Dönüşüm İhtiyacı:</b> Klasik sürekli PSO doğrudan NSP'ye uygulanamaz; takas operatörleri ve kısıt onarım mimarisi şarttır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Karşılaştırma Özeti Tablosu
    st.markdown("##### 🔍 PSO'nun Diğer Metasezgisel Algoritmalar Karşısındaki Konumu:")
    df_pso_comp = pd.DataFrame({
        "Algoritma": [
            "Genetic Algorithm (Sekme 9)",
            "Memetic Algorithm (Sekme 10)",
            "Tabu Search (Sekme 11)",
            "Variable Neighborhood Search (Sekme 12)",
            "Particle Swarm Optimization (Sekme 13)"
        ],
        "Arama Paradigması": [
            "Popülasyon Bazlı Evrimsel",
            "Hibrit: Global GA + Lokal HC",
            "Hafıza Tabanlı Yörünge Arama",
            "Hiyerarşik Çoklu Komşuluk Değişimi",
            "Kolektif Sürü Zekası (Bilişsel + Sosyal)"
        ],
        "Hafıza Mekanizması": [
            "Popülasyon Gen Havuzu",
            "Popülasyon Gen Havuzu",
            "Kısa Vadeli Tabu Yasak Listesi",
            "Komşuluk İndeksi Kademesi",
            "🏆 Çift Hafıza: Bireysel (p_best) + Kolektif (g_best)"
        ],
        "Bilgi Aktarım Yolu": [
            "Genetik Çaprazlama (Crossover)",
            "Çaprazlama + Bireysel İyileştirme",
            "Aday komşu değerlendirmesi",
            "Shaking & Komşuluk Büyütme",
            "🏆 Hız Vektörü ve En İyilere Çekim (Velocity)"
        ],
        "Yakınsama Karakteristiği": [
            "Kademeli & Çeşitlilik Odaklı",
            "Hızlı ve Derinlemesine",
            "Sistematik ve Deterministik",
            "Çok Hızlı ve Esnek",
            "🚀 Çok Hızlı & Kolektif Odaklanma"
        ]
    })
    st.dataframe(df_pso_comp, width="stretch", hide_index=True)

    st.divider()

    # ==============================================================================
    # 3. PARAMETRE KONTROL PANELİ VE ÖN-ANALİZ
    # ==============================================================================
    st.markdown("### 🎛️ PSO Sürü Büyüklüğü & Davranış Katsayıları Kontrol Paneli")

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        swarm_size = st.slider("Sürü Boyutu (Parçacık Sayısı - S)", 10, 80, 30, 5, key="pso_swarm_size", help="Aynı anda arama yapan bağımsız çizelge parçacığı sayısı.")
    with p_col2:
        max_iter = st.slider("Maksimum İterasyon (T_max)", 20, 250, 100, 10, key="pso_max_iter", help="Sürünün kaç adım boyunca uçuş yapacağını belirler.")
    with p_col3:
        w_inertia = st.slider("Atalet Ağırlığı (w)", 0.40, 1.00, 0.72, 0.02, key="pso_w_inertia", help="Parçacığın önceki hızını koruma oranı (Varsayılan: 0.72 - Clerc Konstants).")
    with p_col4:
        c1_cog = st.slider("Bilişsel Katsayı (c1 - p_best)", 0.50, 3.00, 1.49, 0.05, key="pso_c1", help="Parçacığın kendi kişisel en iyi çözümüne yönelme gücü.")
    with p_col5:
        c2_soc = st.slider("Sosyal Katsayı (c2 - g_best)", 0.50, 3.00, 1.49, 0.05, key="pso_c2", help="Parçacığın sürünün küresel liderine yönelme gücü.")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # Ön Analiz Tahmin Rozeti
    total_evals = (swarm_size * max_iter) + swarm_size + 1
    cost_ms = round(params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3)), 3)

    st.markdown("#### 🧮 PSO Arama & Değerlendirme Tahmin Paneli (Ön-Analiz)")
    pred_c1, pred_c2 = st.columns(2)
    with pred_c1:
        render_evaluator_cost_badge(total_evals, cost_ms, label="Tahmini Ceza Değerlendirme (Sürü Boyutu × İterasyon)")
    with pred_c2:
        st.metric(
            label="Sürü Kolektif Kapasitesi",
            value=f"{swarm_size} Parçacık",
            delta=f"{max_iter} İterasyon × {swarm_size} = {swarm_size * max_iter:,} Konum Güncellemesi",
            help="Sürünün toplam arama uzayını tarama gücü."
        )

    st.divider()

    # ==============================================================================
    # 4. ÇALIŞTIRMA VE CANLI OPTİMİZASYON AKIŞI
    # ==============================================================================
    run_btn = st.button("🚀 Particle Swarm Optimization (Discrete PSO) Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t13")

    if not run_btn and "res_t13" not in st.session_state:
        st.warning("👈 Optimizasyonu başlatmak için yukarıdaki **'🚀 Particle Swarm Optimization (Discrete PSO) Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = params.get('live_stream_enabled', True)
        stream_interval = max(2, int(params.get('live_stream_interval', 50) / 10))

        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="Particle Swarm Optimization (Discrete PSO)",
            max_steps=max_iter,
            unit_name="Sürü İterasyonu",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t13_tracker"
        )

        results = run_particle_swarm_optimization(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            weights=weights,
            swarm_size=swarm_size,
            max_iterations=max_iter,
            w_inertia=w_inertia,
            c1_cognitive=c1_cog,
            c2_social=c2_soc,
            seed=42,
            custom_workers=custom_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=stream_interval
        )
        tracker.finish(results.get('total_iterations', max_iter), results.get('final_score'))
        results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t13"] = results
    else:
        results = st.session_state["res_t13"]
        if params.get('live_stream_enabled', True) and 'stream_data' in results:
            render_live_stream_summary(results['stream_data'])

    # --- BİTİRME NEDENİ VE SERT KISIT UYGUNLUK BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")
    render_hard_constraints_status_card(
        results.get('is_feasible', True),
        results.get('hard_violations_count', 0),
        results.get('hard_violation_logs', []),
        solver_name="Particle Swarm Optimization (Discrete PSO)",
        meta=results.get('meta')
    )

    # --- METRİK KARTLARI (8'Lİ PANEL) ---
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Başlangıç Skoru</div>
        <div class="metric-value" style="color: #dc2626;">{results['initial_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">PSO En İyi Skor</div>
        <div class="metric-value" style="color: #ca8a04;">{results['final_score']}</div>
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
        s_size = meta.get('swarm_size', swarm_size)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Sürü Büyüklüğü</div>
        <div class="metric-value" style="color: #8b5cf6;">{s_size} Parçacık</div>
        </div>""", unsafe_allow_html=True)

    with m6:
        v_swaps = meta.get('velocity_swaps_count', 0)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Hız Takasları</div>
        <div class="metric-value" style="color: #0284c7;">{v_swaps:,}</div>
        </div>""", unsafe_allow_html=True)

    with m7:
        eval_c = results.get('eval_count', total_evals)
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

    # ==============================================================================
    # 5. GÖRSEL VE ANALİTİK GRAFİKLER
    # ==============================================================================
    st.markdown("### 🎨 Particle Swarm Optimization (PSO) Yakınsama & Analitik Grafikler")

    score_hist = meta.get('gbest_score_history', results.get('score_history', []))
    avg_hist = meta.get('avg_swarm_score_history', score_hist)
    div_hist = meta.get('diversity_history', [])
    iters = list(range(len(score_hist)))

    # 1. SATIR: İTERASYON BAZLI YAKINSAMA VE PARÇACIK EN İYİ (p_best) DAĞILIMI
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ Sürü Yakınsama & Kolektif Öğrenme Grafiği (Küresel En İyi vs Sürü Ortalaması)")
        fig_pso_conv = go.Figure()
        fig_pso_conv.add_trace(go.Scatter(
            x=iters, y=score_hist,
            name="Küresel En İyi Skor (g_best)",
            line=dict(color="#ca8a04", width=2.8)
        ))
        if len(avg_hist) == len(iters):
            fig_pso_conv.add_trace(go.Scatter(
                x=iters, y=avg_hist,
                name="Sürü Ortalama Skoru (Average Swarm Score)",
                line=dict(color="#64748b", width=1.5, dash="dash")
            ))
        fig_pso_conv.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            xaxis=dict(title="Sürü İterasyonu (Adım)"),
            yaxis=dict(title="Toplam Ceza Skoru Z"),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_pso_conv, width="stretch", key="t13_fig_conv")

    with g_col2:
        st.markdown("##### 2️⃣ Sürü Parçacıklarının Bireysel En İyi (p_best) Ceza Puanı Dağılımı")
        pbest_vals = meta.get('pbest_scores', [results['final_score']] * swarm_size)
        df_pbest = pd.DataFrame({
            "Parçacık": [f"Parçacık #{i+1}" for i in range(len(pbest_vals))],
            "Kişisel En İyi Skor (p_best)": pbest_vals
        })
        fig_pbest = px.bar(
            df_pbest,
            x="Parçacık",
            y="Kişisel En İyi Skor (p_best)",
            color="Kişisel En İyi Skor (p_best)",
            color_continuous_scale="YlOrBr",
            text="Kişisel En İyi Skor (p_best)"
        )
        fig_pbest.add_hline(
            y=results['final_score'],
            line_dash="dot",
            line_color="#059669",
            annotation_text=f"g_best = {results['final_score']}",
            annotation_position="top right"
        )
        fig_pbest.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_pbest, width="stretch", key="t13_fig_pbest")

    # 3-7 STANDART ÇİZELGE ANALİTİKLERİ VE MATRİS TABLOSU
    render_standard_schedule_analytics(
        results, num_days, key_prefix="t13_pso",
        solver_name="Discrete PSO", start_chart_num=3
    )
