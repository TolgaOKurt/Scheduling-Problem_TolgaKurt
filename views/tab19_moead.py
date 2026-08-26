"""
================================================================================
  VIEWS/TAB19_MOEAD.PY - SEKME 19: MOEA/D AYRIŞTIRMA TABANLI ÇOK AMAÇLI OPTİMİZASYON
================================================================================
  Bu modül; Qingfu Zhang & Hui Li (IEEE TEVC 2007) tarafından geliştirilen
  "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition"
  algoritmasını interaktif kontrol paneli, 2D Pareto grafiği, Karar Destek
  Sistemi (DSS) ve yakınsama analitiği ile sunar.
================================================================================
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.worker_manager import generate_worker_profiles
from algorithms.moead_solver import solve_moead
from views.common_components import (
    render_schedule_heatmap,
    render_workload_chart,
    render_penalties_chart,
    render_posta_load_chart,
    render_shift_posta_stacked_chart,
    render_schedule_matrix_table,
    render_request_details_expander,
    render_worker_profiles_table,
    render_hard_constraints_status_card,
    render_evaluator_cost_badge,
    render_tradeoff_insight_banner,
    LiveStreamTracker,
    render_live_stream_summary
)


def render_tab19(global_params):
    """Sekme 19 içeriğini çizer: MOEA/D Ayrıştırma Tabanlı Çok Amaçlı Evrimsel Algoritma."""
    st.markdown("## 🌐 Sekme 19: MOEA/D (Ayrıştırma Tabanlı Çok Amaçlı Evrimsel Algoritma)")

    st.markdown("""
    <div style="background-color: #f0fdfa; border-left: 5px solid #0d9488; border-radius: 8px; padding: 16px 20px; margin-bottom: 22px; color: #115e59; line-height: 1.65;">
        <h4 style="margin-top: 0; color: #0f766e; font-size: 1.15rem;">🧩 Ayrıştırma (Decomposition) Paradigması: Zhang & Li (2007)</h4>
        NSGA-II popülasyonu kalabalıklaşma mesafesiyle bütünsel olarak yönetirken, 
        <b>MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)</b> çok amaçlı problemi 
        <b><i>N</i> adet eşzamanlı çözülen tek amaçlı alt probleme</b> ayrıştırır. 
        Her alt problem belirli bir <b>ağırlık yön vektörünü (&lambda;<sup><i>i</i></sup>)</b> optimize eder ve 
        sadece en yakın <b><i>T</i> komşusu (<i>B</i>(<i>i</i>))</b> ile bilgi paylaşımı yapar. 
        Bu sayede <b>Tchebycheff skalerleştirmesi</b> ile içbükey (non-convex) bölgeler dahil 
        tüm Pareto ön cephesi muazzam bir hızla ve homojen biçimde örülür.
    </div>
    """, unsafe_allow_html=True)

    # --- TEORİK KARTLAR ---
    c_col1, c_col2, c_col3 = st.columns(3)

    with c_col1:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #0d9488; height: 100%;">
            <div class="card-title" style="color: #0f766e;">🌐 1. Ayrıştırma (Decomposition)</div>
            <p style="font-size: 0.85rem; color: #334155; line-height: 1.55;">
                Çok amaçlı problem <i>N</i> adet yön vektörü (&lambda;<sup><i>i</i></sup>) ile <i>N</i> adet alt probleme bölünür. Her birey bir alt problemin en iyi çözümünü temsil eder.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_col2:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #3b82f6; height: 100%;">
            <div class="card-title" style="color: #1d4ed8;">🤝 2. Komşuluk Bilgi Paylaşımı</div>
            <p style="font-size: 0.85rem; color: #334155; line-height: 1.55;">
                Her alt problem <i>i</i>, yalnızca Öklid mesafesi en yakın <i>T</i> adet komşusu (<i>B</i>(<i>i</i>)) ile ebeveyn seçip ürer; genetik bilgi yerel komşulukta hızla yayılır.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_col3:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #8b5cf6; height: 100%;">
            <div class="card-title" style="color: #6d28d9;">📐 3. Tchebycheff & Dış Arşiv (EP)</div>
            <p style="font-size: 0.85rem; color: #334155; line-height: 1.55;">
                Tchebycheff skalerleştirmesi içbükey yüzeylerdeki çözümleri yakalar. Keşfedilen tüm baskın olmayan çözümler Dış Pareto Arşivinde (EP) saklanır.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 1. BÖLÜM: SEKME 19 BAĞIMSIZ MODEL GİRDİ KONTROL PANELİ
    # ==============================================================================
    st.markdown("### 🎛️ 1. Sekme 19 Bağımsız Model & Algoritma Girdi Paneli")
    st.caption("📌 Bu sekmedeki tüm problem boyutlarını, vardiya kotalarını, 2 amaç ceza ağırlıklarını ve MOEA/D parametrelerini doğrudan aşağıdan elle yapılandırabilirsiniz.")

    # 1.1 PROBLEM BOYUTU & VARDİYA TALEPLERİ
    with st.expander("🏢 1.1 Problem Boyutları & Vardiya Asgari Kadro Talepleri", expanded=True):
        dim_col1, dim_col2, dim_col3, dim_col4, dim_col5 = st.columns(5)
        with dim_col1:
            n_workers = st.number_input("Toplam Personel (N)", min_value=16, max_value=60, value=int(global_params.get('n_workers', 28)), step=1, key="t19_n_workers")
        with dim_col2:
            n_days = st.number_input("Planlama Periyodu (Gün)", min_value=7, max_value=28, value=int(global_params.get('n_days', 7)), step=7, key="t19_n_days")
        with dim_col3:
            r_day = st.number_input("Gündüz Min Kadro (R_day)", min_value=1, max_value=25, value=int(global_params.get('r_day', 9)), step=1, key="t19_r_day")
        with dim_col4:
            r_eve = st.number_input("Akşam Min Kadro (R_eve)", min_value=1, max_value=20, value=int(global_params.get('r_eve', 7)), step=1, key="t19_r_eve")
        with dim_col5:
            r_night = st.number_input("Gece Min Kadro (R_night)", min_value=1, max_value=15, value=int(global_params.get('r_night', 5)), step=1, key="t19_r_night")

        st.markdown("---")
        k_c1, k_c2, k_c3 = st.columns([2, 1, 1])
        with k_c1:
            worker_mode = st.selectbox("Personel Kadro Modu", ["Düzenli Kadro (Standart Dengeli)", "Rastgele Kadro (Rastgele İzin & Sertifika)", "🌟 Mükemmel Kadro"], index=0, key="t19_w_mode")
        with k_c2:
            worker_seed = st.number_input("Kadro Rastgelelik Tohumu (Worker Seed)", min_value=1, max_value=9999, value=42, step=1, key="t19_w_seed", help="Rastgele kadro üretiminde usta, sertifika ve izin dağılımını belirleyen tohum.")
        with k_c3:
            show_workers_t19 = st.checkbox("📋 Kadro Listesini İncele", value=False, key="t19_show_workers")

        is_rand_worker = "Rastgele" in worker_mode
        active_workers = generate_worker_profiles(n_workers, n_days, randomize=is_rand_worker, seed=int(worker_seed), mode=worker_mode)
        if show_workers_t19:
            render_worker_profiles_table(active_workers, title="👥 Sekme 19 İçin Oluşturulan Personel Kadrosu")

    # 1.2 ÇATIŞAN 2 AMACIN CEZA AĞIRLIKLARI (OBJECTIVE WEIGHTS)
    with st.expander("⚖️ 1.2 Çatışan 2 Temel Amaç Fonksiyonunun Matematiksel Formülleri & Ceza Katsayıları", expanded=True):
        st.markdown("##### 🏭 1. Amaç: İşletme & Üretim Verimliliği Ağırlıkları (f₁)")
        st.latex(r"f_1(X) = w_{\text{posta}} \cdot \text{Posta\_Dev}(X) + w_{\text{usta}} \cdot \text{No\_Usta}(X) + \text{FourPosta\_Pen}(X)")
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            w_posta = st.slider("Posta Takım Bütünlüğü Ceza Katsayısı (w_posta)", 0, 100, int(global_params.get('weights', {}).get('posta', 15)), 5, key="t19_w_posta")
        with w_col2:
            w_exp = st.slider("Vardiyada Kıdemli Usta Eksikliği Katsayısı (w_usta)", 0, 150, int(global_params.get('weights', {}).get('exp_mix', 60)), 5, key="t19_w_exp")

        st.markdown("##### 👨‍🏭 2. Amaç: Çalışan Memnuniyeti, Adalet & Ergonomi Ağırlıkları (f₂)")
        st.latex(r"f_2(X) = w_{\text{izin}} \cdot \text{Pref\_Viol}(X) + w_{\text{gece}} \cdot \text{Night\_Imb}(X) + w_{\text{work}} \cdot \text{Work\_Imb}(X) + w_{\text{sirk}} \cdot \text{Circadian}(X)")
        w2_col1, w2_col2, w2_col3, w2_col4 = st.columns(4)
        with w2_col1:
            w_pref = st.slider("Kişisel İzin İhlali Katsayısı (w_izin)", 0, 150, int(global_params.get('weights', {}).get('pref_off', 40)), 5, key="t19_w_pref")
        with w2_col2:
            w_night = st.slider("Gece Nöbeti Dengesizliği (w_gece)", 0, 100, int(global_params.get('weights', {}).get('night_imb', 25)), 5, key="t19_w_night")
        with w2_col3:
            w_work = st.slider("Toplam İş Yükü Dengesizliği (w_is)", 0, 100, int(global_params.get('weights', {}).get('workload_imb', 20)), 5, key="t19_w_work")
        with w2_col4:
            w_circ = st.slider("Sirkadiyen Ritim İhlali (w_sirk)", 0, 150, int(global_params.get('weights', {}).get('circadian', 50)), 5, key="t19_w_circ")

    weights = {
        'posta': w_posta,
        'exp_mix': w_exp,
        'pref_off': w_pref,
        'night_imb': w_night,
        'workload_imb': w_work,
        'circadian': w_circ
    }

    # 1.3 MOEA/D ALGORİTMA HİPERPARAMETRELERİ
    with st.expander("🧬 1.3 MOEA/D Ayrıştırma ve Evrim Parametreleri", expanded=True):
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            pop_size = st.slider("Ağırlık Vektörü / Popülasyon Boyutu (N)", min_value=10, max_value=80, value=30, step=5, key="t19_pop_size", help="Çözüm uzayını bölecek yön vektörü sayısı.")
            decomp_method = st.selectbox(
                "Ayrıştırma / Skalerleştirme Yöntemi",
                ["Tchebycheff (Önerilen)", "Weighted Sum (Ağırlıklı Toplam)", "PBI (Boundary Intersection)"],
                index=0,
                key="t19_decomp_method",
                help="Alt problemleri tekil skaler fonksiyona dönüştürme yaklaşımı."
            )
        with m_c2:
            max_generations = st.slider("Maksimum Jenerasyon (G)", min_value=10, max_value=200, value=50, step=10, key="t19_max_gens")
            neighborhood_size = st.slider("Komşuluk Boyutu (T)", min_value=2, max_value=15, value=5, step=1, key="t19_neigh_size", help="Her alt problemin bilgi paylaşacağı en yakın komşu ağırlık sayısı.")
        with m_c3:
            crossover_rate = st.slider("Çaprazlama Olasılığı (p_c)", 0.50, 1.00, 0.85, 0.05, key="t19_pc")
            mutation_rate = st.slider("Takas Mutasyonu Olasılığı (p_m)", 0.01, 0.30, 0.08, 0.01, key="t19_pm")
            use_seed = st.checkbox("Arama Tohumunu Sabitle (Seed=42)", value=True, key="t19_use_seed")
            seed_val = 42 if use_seed else None

    # 1.4 ÖN HESAPLAMA VE ÇAĞRI TAHMİN ROZETİ
    cost_ms = global_params.get('call_cost_ms', round(0.001492 * (n_workers * n_days) + 0.1670, 3))
    total_evals = pop_size + (max_generations * pop_size)
    worst_case_ms = total_evals * cost_ms * 1.5
    worst_case_str = f"{worst_case_ms/1000:.1f} sn" if worst_case_ms >= 1000 else f"{worst_case_ms:.0f} ms"

    st.markdown("#### 🧮 Çok Amaçlı Hesaplama Karmaşıklığı & Değerlendirici Tahmin Paneli")
    pred_c1, pred_c2, pred_c3 = st.columns(3)
    with pred_c1:
        render_evaluator_cost_badge(total_evals, cost_ms, label="Amaç Fonksiyonu Değerlendirmesi (N + G×N)")
    with pred_c2:
        st.metric(
            label="Maksimum Süre (Worst-Case)",
            value=worst_case_str,
            help="Tüm nesiller ve komşuluk bilgi paylaşımları için en kötü durum (üst sınır) hesaplama süresi."
        )
    with pred_c3:
        st.metric(
            label="Algoritmik Ayrıştırma Karmaşıklığı",
            value="O(G · N · T)",
            help="Zhang & Li (2007) Komşuluk Bilgi Paylaşımı asimptotik karmaşıklığı."
        )

    st.divider()

    # ==============================================================================
    # 2. BÖLÜM: ÇALIŞTIRMA VE CANLI OPTİMİZASYON AKIŞI
    # ==============================================================================
    run_btn = st.button("🚀 MOEA/D Çok Amaçlı Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t19")

    if not run_btn and "res_t19" not in st.session_state:
        st.warning("👈 MOEA/D optimizasyonunu koşturmak için yukarıdaki **'🚀 MOEA/D Çok Amaçlı Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = global_params.get('live_stream_enabled', True)
        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="MOEA/D Çok Amaçlı Optimizasyon Akışı",
            max_steps=max_generations,
            unit_name="Nesil",
            enabled=stream_enabled,
            stream_interval=1,
            key_prefix="t19_tracker"
        )

        moead_results = solve_moead(
            n_workers=n_workers,
            n_days=n_days,
            r_day=r_day,
            r_eve=r_eve,
            r_night=r_night,
            weights=weights,
            pop_size=pop_size,
            max_generations=max_generations,
            neighborhood_size=neighborhood_size,
            decomposition_method=decomp_method,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            seed=seed_val,
            custom_workers=active_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=1
        )

        final_best_score = moead_results['knee_solution'].get('total_score', 0)
        tracker.finish(max_generations, final_best_score)
        moead_results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t19"] = moead_results
    else:
        moead_results = st.session_state["res_t19"]
        if global_params.get('live_stream_enabled', True) and 'stream_data' in moead_results:
            render_live_stream_summary(moead_results['stream_data'])

    # Çözüm verilerini ayıkla
    pareto_solutions = moead_results['pareto_solutions']
    knee_sol = moead_results['knee_solution']
    ext_f1_sol = moead_results['extreme_f1_solution']
    ext_f2_sol = moead_results['extreme_f2_solution']
    metrics = moead_results['metrics']
    pop_objs = moead_results['population_objectives']

    st.success(f"📌 **MOEA/D Optimizasyonu Tamamlandı:** {len(pareto_solutions)} adet baskın olmayan Pareto optimal çözüm Dış Arşive (EP) alındı! (Hesaplama Süresi: {metrics['exec_time_ms']} ms)")

    st.divider()

    # ==============================================================================
    # 3. BÖLÜM: İNTERAKTİF 2D PARETO ÖN CEPHESİ (PLOTLY SCATTER)
    # ==============================================================================
    st.markdown("### 📈 3. İnteraktif 2D Pareto Ön Cephesi (Pareto Frontier)")
    st.caption("Aşağıdaki 2 boyutlu grafikte, MOEA/D popülasyon bireyleri (gri elmaslar) ve evrim boyunca Dış Arşivde (EP) toplanan kesin Pareto cephesi (turkuaz çizgi) gösterilmektedir.")

    fig_pareto = go.Figure()

    # 1. Katman: Son Nesil Popülasyon Bireyleri
    pop_f1 = [ov[0] for ov in pop_objs]
    pop_f2 = [ov[1] for ov in pop_objs]
    fig_pareto.add_trace(go.Scatter(
        x=pop_f1,
        y=pop_f2,
        mode='markers',
        name=f'Mevcut Popülasyon ({len(pop_f1)} Alt Problem)',
        marker=dict(size=7, color='#94a3b8', symbol='diamond-open'),
        hoverinfo='skip'
    ))

    # 2. Katman: Dış Pareto Arşivi (EP) Cephesi
    pareto_f1 = [s['f1'] for s in pareto_solutions]
    pareto_f2 = [s['f2'] for s in pareto_solutions]
    pareto_hover_text = [
        f"<b>Dış Arşiv (EP) Çözümü #{i+1}</b><br>🏭 f₁ (İşletme): {s['f1']:.0f}<br>👨‍🏭 f₂ (Çalışan): {s['f2']:.0f}<br>🎯 Toplam Skor (Z): {s['total_score']}<br>Posta Cezası: {s['penalties'].get('Posta Takım Bütünlüğü İhlali', 0)}<br>İzin Cezası: {s['penalties'].get('Kişisel İzin İhlali', 0)}"
        for i, s in enumerate(pareto_solutions)
    ]

    fig_pareto.add_trace(go.Scatter(
        x=pareto_f1,
        y=pareto_f2,
        mode='lines+markers',
        name='Dış Pareto Arşivi (EP Frontier)',
        line=dict(color='#0d9488', width=3, shape='linear'),
        marker=dict(size=11, color='#14b8a6', symbol='circle'),
        text=pareto_hover_text,
        hoverinfo='text'
    ))

    # 3. Özel Vurgular: Diz Noktası, İşletme Uç ve Çalışan Uç
    if knee_sol:
        fig_pareto.add_trace(go.Scatter(
            x=[knee_sol['f1']],
            y=[knee_sol['f2']],
            mode='markers+text',
            name='⭐ Diz Noktası (Knee Point / Altın Denge)',
            marker=dict(size=18, color='#f59e0b', symbol='star'),
            text=["⭐ Diz Noktası"],
            textposition="top right",
            textfont=dict(color="#b45309", size=12, family="Arial Black"),
            hovertext=[f"⭐ <b>Diz Noktası (Altın Denge)</b><br>f₁: {knee_sol['f1']:.0f} | f₂: {knee_sol['f2']:.0f}<br>Toplam Z: {knee_sol['total_score']}"],
            hoverinfo='text'
        ))

    if ext_f1_sol:
        fig_pareto.add_trace(go.Scatter(
            x=[ext_f1_sol['f1']],
            y=[ext_f1_sol['f2']],
            mode='markers+text',
            name='🏭 İşletme Odaklı Uç Çözüm (Min f₁)',
            marker=dict(size=14, color='#2563eb', symbol='diamond'),
            text=["🏭 Min f₁"],
            textposition="bottom left",
            textfont=dict(color="#1d4ed8", size=11),
            hovertext=[f"🏭 <b>İşletme Odaklı Uç Çözüm</b><br>f₁: {ext_f1_sol['f1']:.0f} | f₂: {ext_f1_sol['f2']:.0f}<br>Toplam Z: {ext_f1_sol['total_score']}"],
            hoverinfo='text'
        ))

    if ext_f2_sol:
        fig_pareto.add_trace(go.Scatter(
            x=[ext_f2_sol['f1']],
            y=[ext_f2_sol['f2']],
            mode='markers+text',
            name='👨‍🏭 Çalışan Odaklı Uç Çözüm (Min f₂)',
            marker=dict(size=14, color='#e11d48', symbol='square'),
            text=["👨‍🏭 Min f₂"],
            textposition="top right",
            textfont=dict(color="#be185d", size=11),
            hovertext=[f"👨‍🏭 <b>Çalışan Odaklı Uç Çözüm</b><br>f₁: {ext_f2_sol['f1']:.0f} | f₂: {ext_f2_sol['f2']:.0f}<br>Toplam Z: {ext_f2_sol['total_score']}"],
            hoverinfo='text'
        ))

    fig_pareto.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=480,
        xaxis=dict(title="🏭 f₁: İşletme & Üretim Verimliliği Cezası (Posta & Usta)", gridcolor="#e2e8f0"),
        yaxis=dict(title="👨‍🏭 f₂: Çalışan Memnuniyeti Cezası (İzin, Gece, Sirkadiyen)", gridcolor="#e2e8f0"),
        legend=dict(orientation="h", y=1.12, x=0.0),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    st.plotly_chart(fig_pareto, width="stretch", key="t19_pareto_scatter")

    # Metrik Rozetleri
    pm_c1, pm_c2, pm_c3, pm_c4 = st.columns(4)
    with pm_c1:
        st.metric(label="Dış Arşiv (EP) Boyutu", value=f"{len(pareto_solutions)} Çözüm", help="Baskın olmayan kesin optimal çizelgeler.")
    with pm_c2:
        st.metric(label="Nihai Hiper-Hacim (HV)", value=f"{metrics['final_hypervolume']:,.0f}", help="Pareto cephesinin kapsadığı çözüm uzayı hacmi.")
    with pm_c3:
        st.metric(label="Deb's Yayılım Metriği (Spread Δ)", value=f"{metrics['final_spread_delta']:.3f}", help="Pareto noktalarının homojen dağılım katsayısı.")
    with pm_c4:
        st.metric(label="Diz Noktası Toplam Cezası (Z)", value=f"{knee_sol.get('total_score', 0)}", help="Altın denge noktasındaki toplam ceza puanı.")

    # 💡 Yönetimsel Ödünleşim & Altın Denge Bilgi Kartı
    render_tradeoff_insight_banner(ext_f1_sol, ext_f2_sol, knee_sol)

    st.divider()

    # ==============================================================================
    # 4. BÖLÜM: İNTERAKTİF ÇİZELGE DENETÇİSİ & KARAR DESTEK SİSTEMİ (DSS)
    # ==============================================================================
    st.markdown("### 🔍 4. İnteraktif Çizelge Denetçisi & Karar Destek Sistemi (DSS)")
    st.caption("Dış Pareto Arşivinden (EP) bir yönetimsel strateji veya özel nokta seçerek ilgili çizelgenin ısı haritasını ve personel dağılımını inceleyiniz.")

    strategy_options = [
        "⭐ 1. Strateji: Diz Noktası (Knee Point / Altın Denge - Önerilen)",
        "🏭 2. Strateji: İşletme Odaklı Uç Çözüm (Min f₁ - Üretim Öncelikli)",
        "👨‍🏭 3. Strateji: Çalışan Odaklı Uç Çözüm (Min f₂ - Sendika / Ergonomi Öncelikli)",
        "📋 4. Seçenek: Pareto Cephesindeki Özel Bir Arşiv Noktasını İncele"
    ]

    sel_strategy = st.radio("🎯 İncelenecek Çizelge Stratejisini Seçin:", strategy_options, index=0, horizontal=True, key="t19_strategy_radio")

    if "1. Strateji" in sel_strategy:
        selected_schedule_dict = knee_sol
        strategy_badge = "⭐ Diz Noktası (Knee Point / Altın Denge)"
        strategy_desc = "Minimum karşılıklı tavizle işletme ve personel arasında optimum uzlaşma noktası."
    elif "2. Strateji" in sel_strategy:
        selected_schedule_dict = ext_f1_sol
        strategy_badge = "🏭 İşletme Odaklı Uç Çözüm (Min f₁)"
        strategy_desc = "Posta takım bütünlüğü ve usta mevcudiyeti tam koruma altındadır."
    elif "3. Strateji" in sel_strategy:
        selected_schedule_dict = ext_f2_sol
        strategy_badge = "👨‍🏭 Çalışan Odaklı Uç Çözüm (Min f₂)"
        strategy_desc = "Tüm çalışanların kişisel izin talepleri ve ergonomi dengesi maksimize edilmiştir."
    else:
        pareto_labels = [f"Arşiv Çözümü #{i+1} | f₁={s['f1']:.0f}, f₂={s['f2']:.0f} (Z={s['total_score']})" for i, s in enumerate(pareto_solutions)]
        sel_idx = st.selectbox("Pareto Cephesinden İncelemek İstediğiniz Çözümü Seçin:", range(len(pareto_solutions)), format_func=lambda i: pareto_labels[i], key="t19_custom_sol_select")
        selected_schedule_dict = pareto_solutions[sel_idx]
        strategy_badge = f"📋 Dış Arşiv Çözümü #{sel_idx+1}"
        strategy_desc = f"Pareto ön cephesinde f₁={selected_schedule_dict['f1']:.0f} ve f₂={selected_schedule_dict['f2']:.0f} değerine sahip arşiv çözümü."

    st.markdown(f"#### 📌 Seçili Senaryo: **{strategy_badge}**")
    st.info(f"💡 **Senaryo Karakteristiği:** {strategy_desc}")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #2563eb;">
            <div class="metric-label">🏭 İşletme Cezası (f₁)</div>
            <div class="metric-value" style="color: #1d4ed8;">{selected_schedule_dict['f1']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #ec4899;">
            <div class="metric-label">👨‍🏭 Çalışan Cezası (f₂)</div>
            <div class="metric-value" style="color: #be185d;">{selected_schedule_dict['f2']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #0d9488;">
            <div class="metric-label">🎯 Toplam Ceza Skoru (Z)</div>
            <div class="metric-value" style="color: #0f766e;">{selected_schedule_dict['total_score']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        feas_text = "✅ %100 GEÇERLİ" if selected_schedule_dict.get('is_feasible', True) else f"❌ {selected_schedule_dict.get('hard_violations_count', 0)} İhlal"
        feas_color = "#059669" if selected_schedule_dict.get('is_feasible', True) else "#dc2626"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {feas_color};">
            <div class="metric-label">🛡️ Sert Kısıt Uygunluğu</div>
            <div class="metric-value" style="color: {feas_color};">{feas_text}</div>
        </div>
        """, unsafe_allow_html=True)

    render_hard_constraints_status_card(
        selected_schedule_dict.get('is_feasible', True),
        selected_schedule_dict.get('hard_violations_count', 0),
        selected_schedule_dict.get('hard_violation_logs', []),
        solver_name="MOEA/D Çok Amaçlı Çözücü"
    )

    # Görsel Grafikler
    render_schedule_heatmap(
        selected_schedule_dict['schedule'],
        active_workers,
        n_days,
        key_prefix="t19_sel",
        title=f"Vardiya Çizelgesi Isı Haritası ({strategy_badge})"
    )

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        render_penalties_chart(
            selected_schedule_dict['penalties'],
            key_prefix="t19_sel",
            title=f"Yumuşak Kısıt Ceza Dağılımı ({strategy_badge})"
        )
    with g_col2:
        render_workload_chart(
            selected_schedule_dict['schedule'],
            active_workers,
            key_prefix="t19_sel",
            title=f"Çalışan İş Yükü ve Gece Nöbeti Dağılımı ({strategy_badge})"
        )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        render_posta_load_chart(
            selected_schedule_dict['schedule'],
            active_workers,
            key_prefix="t19_sel",
            title=f"Posta Yük ve Gece Dağılımı ({strategy_badge})"
        )
    with p_col2:
        render_shift_posta_stacked_chart(
            selected_schedule_dict['schedule'],
            active_workers,
            n_days,
            key_prefix="t19_sel",
            title=f"Gün/Vardiya Bazında Posta Dağılımı ({strategy_badge})"
        )

    # Tablo ve İzin Dökümleri
    render_schedule_matrix_table(
        selected_schedule_dict['schedule'],
        active_workers,
        n_days,
        title=f"🗓️ Tam Vardiya Çizelge Matrisi Tablosu ({strategy_badge})"
    )

    render_request_details_expander(
        selected_schedule_dict['request_details'],
        title=f"📋 Kişisel İzin Talepleri ve Karşılanma Durumu Raporu ({strategy_badge})"
    )

    st.divider()

    # ==============================================================================
    # 5. BÖLÜM: ÇOK AMAÇLI EVRİMSEL YAKINSAMA VE PERFORMANS ANALİTİĞİ
    # ==============================================================================
    st.markdown("### 📊 5. Çok Amaçlı Evrimsel Yakınsama Analitiği")
    st.caption("MOEA/D algoritmasının jenerasyonlar boyunca Hiper-Hacim (HV), Dış Arşiv büyüklüğü ve amaç fonksiyonlarının gelişim grafikleri:")

    gens_list = list(range(1, max_generations + 1))

    c_g1, c_g2 = st.columns(2)

    with c_g1:
        st.markdown("##### 1️⃣ Hiper-Hacim (Hypervolume - HV) Yakınsama Eğrisi")
        fig_hv = go.Figure()
        fig_hv.add_trace(go.Scatter(
            x=gens_list,
            y=metrics['hv_history'],
            mode='lines+markers',
            name='Hiper-Hacim (HV)',
            line=dict(color='#0d9488', width=2.5),
            marker=dict(size=4)
        ))
        fig_hv.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=320,
            xaxis=dict(title="Jenerasyon (Nesil)"),
            yaxis=dict(title="HV Skoru (Kapsanan Hacim)"),
            margin=dict(l=30, r=30, t=20, b=30)
        )
        st.plotly_chart(fig_hv, width="stretch", key="t19_fig_hv")

    with c_g2:
        st.markdown("##### 2️⃣ Dış Pareto Arşivi (EP Size) Büyüme Eğrisi")
        fig_ep = go.Figure()
        fig_ep.add_trace(go.Scatter(
            x=gens_list,
            y=metrics['ep_size_history'],
            mode='lines+markers',
            name='Dış Arşivdeki Çözüm Sayısı',
            line=dict(color='#8b5cf6', width=2.5),
            marker=dict(size=4)
        ))
        fig_ep.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=320,
            xaxis=dict(title="Jenerasyon (Nesil)"),
            yaxis=dict(title="Arşivdeki Çözüm Sayısı"),
            margin=dict(l=30, r=30, t=20, b=30)
        )
        st.plotly_chart(fig_ep, width="stretch", key="t19_fig_ep")

    # 3. Grafik: Amaç Bazlı İlerleme
    st.markdown("##### 3️⃣ Amaç Fonksiyonlarının Nesiller Boyunca Evrimi (Min f₁ vs. Min f₂ vs. Ortalama)")
    fig_objs = go.Figure()
    fig_objs.add_trace(go.Scatter(x=gens_list, y=metrics['min_f1_history'], name="Min f₁ (İşletme Cezası)", line=dict(color="#2563eb", width=2)))
    fig_objs.add_trace(go.Scatter(x=gens_list, y=metrics['min_f2_history'], name="Min f₂ (Çalışan Cezası)", line=dict(color="#ec4899", width=2)))
    fig_objs.add_trace(go.Scatter(x=gens_list, y=metrics['avg_f1_history'], name="Ortalama f₁", line=dict(color="#93c5fd", width=1.5, dash="dot")))
    fig_objs.add_trace(go.Scatter(x=gens_list, y=metrics['avg_f2_history'], name="Ortalama f₂", line=dict(color="#f472b6", width=1.5, dash="dot")))
    fig_objs.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=340,
        xaxis=dict(title="Jenerasyon (Nesil)"),
        yaxis=dict(title="Ceza Puanı"),
        legend=dict(orientation="h", y=1.15)
    )
    st.plotly_chart(fig_objs, width="stretch", key="t19_fig_objs")

    st.divider()

    # ==============================================================================
    # 6. BÖLÜM: 3 TEMEL YÖNETİMSEL STRATEJİ KARŞILAŞTIRMA MATRİSİ
    # ==============================================================================
    st.markdown("### ⚖️ 6. Yönetimsel Strateji Ödünleşim Matrisi (Trade-off Matrix)")

    def _calc_stats(sol_dict):
        reqs = sol_dict.get('request_details', [])
        total_reqs = len(reqs)
        fulfilled = sum(1 for r in reqs if "Karşılandı" in r['durum'])
        pct_req = round((fulfilled / max(1, total_reqs)) * 100, 1)

        pen = sol_dict.get('penalties', {})
        posta_p = pen.get('Posta Takım Bütünlüğü İhlali', 0)
        usta_p = pen.get('Kıdem / Usta Eksikliği', 0)
        circ_p = pen.get('Sirkadiyen Ritim İhlali (Akşam->Gündüz)', 0)

        sched = sol_dict['schedule']
        night_counts = (sched == 3).sum(axis=1)
        night_std = round(float(np.std(night_counts)), 2)

        return {
            'f1': sol_dict['f1'],
            'f2': sol_dict['f2'],
            'total': sol_dict['total_score'],
            'pct_req': f"%{pct_req}",
            'posta_p': posta_p,
            'usta_p': usta_p,
            'circ_p': circ_p,
            'night_std': night_std
        }

    s_mng = _calc_stats(ext_f1_sol)
    s_knee = _calc_stats(knee_sol)
    s_emp = _calc_stats(ext_f2_sol)

    tradeoff_df = pd.DataFrame({
        "Karşılaştırma Boyutu": [
            "Strateji Tanımı & Önceliği",
            "İşletme Cezası f₁ (Posta & Usta)",
            "Çalışan Cezası f₂ (İzin & Gece & Ergonomi)",
            "Toplam Ceza Skoru (Z = f₁ + f₂)",
            "Karşılanan Kişisel İzin Oranı (%)",
            "Posta Bütünlüğü Ceza Puanı",
            "Kıdemli Usta Eksikliği Cezası",
            "Sirkadiyen Ritim İhlali Cezası",
            "Gece Nöbeti Adaleti (Standart Sapma)",
            "En Uygun Olduğu Dönem"
        ],
        "🏭 1. Strateji: İşletme Odaklı (Min f₁)": [
            "Üretim sürekliliği & Posta birliği",
            f"{s_mng['f1']:.0f} (En İyi)",
            f"{s_mng['f2']:.0f} (Tavizli)",
            f"{s_mng['total']}",
            f"{s_mng['pct_req']}",
            f"{s_mng['posta_p']}",
            f"{s_mng['usta_p']}",
            f"{s_mng['circ_p']}",
            f"{s_mng['night_std']}",
            "Acil sipariş ve yüksek üretim kotaları"
        ],
        "⭐ 2. Strateji: Diz Noktası (Knee Point)": [
            "Altın Denge / Minimum karşılıklı taviz",
            f"{s_knee['f1']:.0f} (Dengeli)",
            f"{s_knee['f2']:.0f} (Dengeli)",
            f"{s_knee['total']}",
            f"{s_knee['pct_req']}",
            f"{s_knee['posta_p']}",
            f"{s_knee['usta_p']}",
            f"{s_knee['circ_p']}",
            f"{s_knee['night_std']}",
            "Standart ve sürdürülebilir fabrika işletimi"
        ],
        "👨‍🏭 3. Strateji: Çalışan Odaklı (Min f₂)": [
            "Maksimum çalışan memnuniyeti & İzinler",
            f"{s_emp['f1']:.0f} (Tavizli)",
            f"{s_emp['f2']:.0f} (En İyi)",
            f"{s_emp['total']}",
            f"{s_emp['pct_req']}",
            f"{s_emp['posta_p']}",
            f"{s_emp['usta_p']}",
            f"{s_emp['circ_p']}",
            f"{s_emp['night_std']}",
            "Bayram, bakım periyotları & sendikal mutabakat"
        ]
    })

    st.dataframe(tradeoff_df, hide_index=True, use_container_width=True)
