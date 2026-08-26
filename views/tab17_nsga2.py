"""
================================================================================
  VIEWS/TAB17_NSGA2.PY - SEKME 17: NSGA-II İLE ÇOK AMAÇLI OPTİMİZASYON (Deb et al., 2002)
================================================================================
  Bu modül; Yöneylem Araştırması ve Evrimsel Hesaplama literatürünün altın standardı
  olan NSGA-II (Deb et al., 2002) çok amaçlı algoritmasını, interaktif Pareto
  ön cephesi (Pareto Frontier) görselleştirmesini ve Karar Destek Sistemini (DSS)
  bağımsız ve modüler olarak sunar.
================================================================================
"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.worker_manager import generate_worker_profiles
from algorithms.nsga2_solver import run_nsga2_optimization
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


def render_tab17(global_params):
    """Sekme 17 içeriğini çizer: NSGA-II Çok Amaçlı Optimizasyon ve Pareto Analitiği."""
    st.markdown("## 🎯 Sekme 17: NSGA-II (Deb et al., 2002) ile Çok Amaçlı Optimizasyon & Pareto Analizi")

    st.markdown("""
    <div style="background-color: #eff6ff; border-left: 5px solid #2563eb; border-radius: 8px; padding: 16px 20px; margin-bottom: 22px; color: #1e40af; line-height: 1.65;">
        <h4 style="margin-top: 0; color: #1d4ed8; font-size: 1.15rem;">🧬 Kalyanmoy Deb et al. (2002) NSGA-II ile Vektörel Optimizasyon</h4>
        Klasik tek amaçlı çözücüler tüm ceza kalemlerini tek bir yapay ağırlıkta birleştirirken, 
        <b>NSGA-II (Non-dominated Sorting Genetic Algorithm II)</b> algoritması çatışan amaçları 
        (<b>İşletme Verimliliği <i>f</i><sub>1</sub></b> vs. <b>Çalışan Memnuniyeti <i>f</i><sub>2</sub></b>) bağımsız vektörel eksenler olarak eşzamanlı optimize eder.
        Sonuçta karar vericiye tek bir zorunlu çizelge yerine <b>Pareto Optimal Seçenekler Yelpazesi (Pareto Frontier)</b> sunulur.
    </div>
    """, unsafe_allow_html=True)

    # --- TEORİK KARTLAR ---
    c_col1, c_col2, c_col3 = st.columns(3)

    with c_col1:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #3b82f6; height: 100%;">
            <div class="card-title" style="color: #1d4ed8;">⚡ 1. Hızlı Baskınlık Sıralaması</div>
            <p style="font-size: 0.85rem; color: #334155; line-height: 1.55;">
                Popülasyonu katman katman (Front 1, Front 2, ...) Pareto rütbelerine ayırır. 
                <i>O</i>(<i>M</i> &times; <i>N</i><sup>2</sup>) karmaşıklığı ile en iyi ödünleşim sunan çözümleri öne çıkarır.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_col2:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #10b981; height: 100%;">
            <div class="card-title" style="color: #047857;">📏 2. Kalabalıklaşma Mesafesi</div>
            <p style="font-size: 0.85rem; color: #334155; line-height: 1.55;">
                Pareto cephesindeki çözümlerin yoğunluğunu (Crowding Distance) ölçerek çözümlerin tek bir noktaya yığılmasını önler ve homojen dağılımı garanti eder.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_col3:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #8b5cf6; height: 100%;">
            <div class="card-title" style="color: #6d28d9;">🛡️ 3. (N + N) → N Elitizm</div>
            <p style="font-size: 0.85rem; color: #334155; line-height: 1.55;">
                Ebeveyn ve yavru popülasyonları birleştirip (2<i>N</i>) içinden en yüksek rütbeli ve en yüksek çeşitliliğe sahip seçkin <i>N</i> bireyi sonraki nesle taşır.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 1. BÖLÜM: SEKME 17 BAĞIMSIZ MODEL GİRDİ KONTROL PANELİ
    # ==============================================================================
    st.markdown("### 🎛️ 1. Sekme 17 Bağımsız Model & Algoritma Girdi Paneli")
    st.caption("📌 Bu sekmedeki tüm problem boyutlarını, vardiya kotalarını, 2 amaç ceza ağırlıklarını ve NSGA-II parametrelerini doğrudan aşağıdan elle yapılandırabilirsiniz.")

    # 1.1 PROBLEM BOYUTU & VARDİYA TALEPLERİ
    with st.expander("🏢 1.1 Problem Boyutları & Vardiya Asgari Kadro Talepleri", expanded=True):
        dim_col1, dim_col2, dim_col3, dim_col4, dim_col5 = st.columns(5)
        with dim_col1:
            n_workers = st.number_input("Toplam Personel (N)", min_value=16, max_value=60, value=int(global_params.get('n_workers', 28)), step=1, key="t17_n_workers")
        with dim_col2:
            n_days = st.number_input("Planlama Periyodu (Gün)", min_value=7, max_value=28, value=int(global_params.get('n_days', 7)), step=7, key="t17_n_days")
        with dim_col3:
            r_day = st.number_input("Gündüz Min Kadro (R_day)", min_value=1, max_value=25, value=int(global_params.get('r_day', 9)), step=1, key="t17_r_day")
        with dim_col4:
            r_eve = st.number_input("Akşam Min Kadro (R_eve)", min_value=1, max_value=20, value=int(global_params.get('r_eve', 7)), step=1, key="t17_r_eve")
        with dim_col5:
            r_night = st.number_input("Gece Min Kadro (R_night)", min_value=1, max_value=15, value=int(global_params.get('r_night', 5)), step=1, key="t17_r_night")

        st.markdown("---")
        k_c1, k_c2, k_c3 = st.columns([2, 1, 1])
        with k_c1:
            worker_mode = st.selectbox("Personel Kadro Modu", ["Düzenli Kadro (Standart Dengeli)", "Rastgele Kadro (Rastgele İzin & Sertifika)", "🌟 Mükemmel Kadro"], index=0, key="t17_w_mode")
        with k_c2:
            worker_seed = st.number_input("Kadro Rastgelelik Tohumu (Worker Seed)", min_value=1, max_value=9999, value=42, step=1, key="t17_w_seed", help="Rastgele kadro üretiminde usta, sertifika ve izin dağılımını belirleyen tohum.")
        with k_c3:
            show_workers_t17 = st.checkbox("📋 Kadro Listesini İncele", value=False, key="t17_show_workers")

        is_rand_worker = "Rastgele" in worker_mode
        active_workers = generate_worker_profiles(n_workers, n_days, randomize=is_rand_worker, seed=int(worker_seed), mode=worker_mode)
        if show_workers_t17:
            render_worker_profiles_table(active_workers, title="👥 Sekme 17 İçin Oluşturulan Personel Kadrosu")

    # 1.2 ÇATIŞAN 2 AMACIN CEZA AĞIRLIKLARI (OBJECTIVE WEIGHTS)
    with st.expander("⚖️ 1.2 Çatışan 2 Temel Amaç Fonksiyonunun Matematiksel Formülleri & Ceza Katsayıları", expanded=True):
        st.markdown("##### 🏭 1. Amaç: İşletme & Üretim Verimliliği Ağırlıkları (f₁)")
        st.latex(r"f_1(X) = w_{\text{posta}} \cdot \text{Posta\_Dev}(X) + w_{\text{usta}} \cdot \text{No\_Usta}(X) + \text{FourPosta\_Pen}(X)")
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            w_posta = st.slider("Posta Takım Bütünlüğü Ceza Katsayısı (w_posta)", 0, 100, int(global_params.get('weights', {}).get('posta', 15)), 5, key="t17_w_posta")
        with w_col2:
            w_exp = st.slider("Vardiyada Kıdemli Usta Eksikliği Katsayısı (w_usta)", 0, 150, int(global_params.get('weights', {}).get('exp_mix', 60)), 5, key="t17_w_exp")

        st.markdown("##### 👨‍🏭 2. Amaç: Çalışan Memnuniyeti, Adalet & Ergonomi Ağırlıkları (f₂)")
        st.latex(r"f_2(X) = w_{\text{izin}} \cdot \text{Pref\_Viol}(X) + w_{\text{gece}} \cdot \text{Night\_Imb}(X) + w_{\text{work}} \cdot \text{Work\_Imb}(X) + w_{\text{sirk}} \cdot \text{Circadian}(X)")
        w2_col1, w2_col2, w2_col3, w2_col4 = st.columns(4)
        with w2_col1:
            w_pref = st.slider("Kişisel İzin İhlali Katsayısı (w_izin)", 0, 150, int(global_params.get('weights', {}).get('pref_off', 40)), 5, key="t17_w_pref")
        with w2_col2:
            w_night = st.slider("Gece Nöbeti Dengesizliği (w_gece)", 0, 100, int(global_params.get('weights', {}).get('night_imb', 25)), 5, key="t17_w_night")
        with w2_col3:
            w_work = st.slider("Toplam İş Yükü Dengesizliği (w_is)", 0, 100, int(global_params.get('weights', {}).get('workload_imb', 20)), 5, key="t17_w_work")
        with w2_col4:
            w_circ = st.slider("Sirkadiyen Ritim İhlali (w_sirk)", 0, 150, int(global_params.get('weights', {}).get('circadian', 50)), 5, key="t17_w_circ")

    weights = {
        'posta': w_posta,
        'exp_mix': w_exp,
        'pref_off': w_pref,
        'night_imb': w_night,
        'workload_imb': w_work,
        'circadian': w_circ
    }

    # 1.3 NSGA-II HİPERPARAMETRELERİ
    with st.expander("🧬 1.3 NSGA-II Evrimsel Arama Hiperparametreleri", expanded=True):
        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
        with p_col1:
            pop_size = st.slider("Popülasyon Büyüklüğü (N)", 30, 200, 60, 10, key="t17_pop")
        with p_col2:
            generations = st.slider("Jenerasyon Sayısı (G)", 10, 400, 80, 10, key="t17_gen")
        with p_col3:
            crossover_rate = st.slider("Çaprazlama Oranı (p_c)", 0.50, 1.00, 0.85, 0.05, format="%.2f", key="t17_pc")
        with p_col4:
            mutation_rate = st.slider("Mutasyon Oranı (p_m)", 0.01, 0.30, 0.08, 0.01, format="%.2f", key="t17_pm")
        with p_col5:
            seed = st.number_input("Arama Algoritması Tohumu (Seed)", min_value=1, max_value=9999, value=42, step=1, key="t17_seed", help="Evrimsel çaprazlama ve mutasyon operatörlerinin rassal tohumu.")

    # 1.4 ÖN HESAPLAMA VE ÇAĞRI TAHMİN ROZETİ
    cost_ms = global_params.get('call_cost_ms', round(0.001492 * (n_workers * n_days) + 0.1670, 3))
    total_evals = pop_size + (generations * pop_size)
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
            help="Tüm nesil ve popülasyon sıralamaları için en kötü durum (üst sınır) hesaplama süresi."
        )
    with pred_c3:
        st.metric(
            label="Algoritmik Sıralama Karmaşıklığı",
            value="O(G · M · N²)",
            help="Deb et al. (2002) Hızlı Baskınlık Sıralaması teorik asimptotik karmaşıklığı."
        )

    st.divider()

    # ==============================================================================
    # 2. BÖLÜM: ÇALIŞTIRMA VE CANLI OPTİMİZASYON AKIŞI
    # ==============================================================================
    run_btn = st.button("🚀 NSGA-II Çok Amaçlı Evrimsel Optimizasyonu Başlat", type="primary", width="stretch", key="btn_run_t17")

    if not run_btn and "res_t17" not in st.session_state:
        st.warning("👈 Çok amaçlı Pareto optimizasyonunu başlatmak için yukarıdaki **'🚀 NSGA-II Çok Amaçlı Evrimsel Optimizasyonu Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = global_params.get('live_stream_enabled', True)
        stream_interval = max(1, int(global_params.get('live_stream_interval', 50) / 25))

        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="NSGA-II Çok Amaçlı Evrimsel Arama (Deb et al., 2002)",
            max_steps=generations,
            unit_name="Nesil",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t17_tracker"
        )

        nsga2_results = run_nsga2_optimization(
            n_workers=n_workers,
            n_days=n_days,
            r_day=r_day,
            r_eve=r_eve,
            r_night=r_night,
            weights=weights,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            seed=seed,
            custom_workers=active_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=stream_interval
        )
        
        final_best_score = nsga2_results['knee_solution']['total_score']
        tracker.finish(generations, final_best_score)
        nsga2_results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t17"] = nsga2_results
    else:
        nsga2_results = st.session_state["res_t17"]
        if global_params.get('live_stream_enabled', True) and 'stream_data' in nsga2_results:
            render_live_stream_summary(nsga2_results['stream_data'])

    # Çözüm verilerini ayıkla
    pareto_solutions = nsga2_results['pareto_solutions']
    knee_sol = nsga2_results['knee_solution']
    ext_f1_sol = nsga2_results['extreme_f1_solution']
    ext_f2_sol = nsga2_results['extreme_f2_solution']
    all_evals = nsga2_results['all_evaluated_solutions']
    metrics = nsga2_results['metrics']

    st.success(f"📌 **Optimizasyon Başarıyla Tamamlandı:** {len(pareto_solutions)} adet baskın olmayan Pareto optimal çözüm keşfedildi! (Hesaplama Süresi: {metrics['exec_time_ms']} ms | Çağrı: {metrics['total_evaluations']:,})")

    st.divider()

    # ==============================================================================
    # 3. BÖLÜM: İNTERAKTİF PARETO ÖN CEPHESİ (PARETO FRONTIER 2D SCATTER)
    # ==============================================================================
    st.markdown("### 📈 3. İnteraktif Pareto Ön Cephesi & Amaç Uzayı Dağılımı")
    st.caption("Aşağıdaki 2 boyutlu grafikte, f₁ (İşletme Cezası) ve f₂ (Çalışan Cezası) eksenlerinde baskın olmayan (Rank 1) Pareto çözümleri birbirine yeşil sınır çizgisiyle bağlanmıştır.")

    # 2D Plotly Scatter Plot
    fig_pareto = go.Figure()

    # 1. Katman: Tüm Popülasyon Bireyleri (Sönük Noktalar)
    if all_evals:
        df_all = pd.DataFrame(all_evals)
        fig_pareto.add_trace(go.Scatter(
            x=df_all['f1'],
            y=df_all['f2'],
            mode='markers',
            name='Popülasyon Bireyleri (Front 2+)',
            marker=dict(size=6, color='#cbd5e1', opacity=0.6),
            hoverinfo='skip'
        ))

    # 2. Katman: Pareto Ön Cephesi (Rank 1 Bağlantı Çizgisi)
    pareto_f1 = [s['f1'] for s in pareto_solutions]
    pareto_f2 = [s['f2'] for s in pareto_solutions]
    pareto_hover_text = [
        f"<b>Çözüm #{i+1}</b><br>🏭 f₁ (İşletme): {s['f1']:.0f}<br>👨‍🏭 f₂ (Çalışan): {s['f2']:.0f}<br>🎯 Toplam Skor (Z): {s['total_score']}<br>Posta Cezası: {s['penalties'].get('Posta Takım Bütünlüğü İhlali', 0)}<br>İzin Cezası: {s['penalties'].get('Kişisel İzin İhlali', 0)}<br>Usta Eksikliği: {s['penalties'].get('Kıdem / Usta Eksikliği', 0)}"
        for i, s in enumerate(pareto_solutions)
    ]

    fig_pareto.add_trace(go.Scatter(
        x=pareto_f1,
        y=pareto_f2,
        mode='lines+markers',
        name='Pareto Ön Cephesi (Rank 1 Frontier)',
        line=dict(color='#059669', width=2.5, shape='linear'),
        marker=dict(size=10, color='#10b981', symbol='circle'),
        text=pareto_hover_text,
        hoverinfo='text'
    ))

    # 3. Özel Vurgular: Diz Noktası, İşletme Uç ve Çalışan Uç
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

    st.plotly_chart(fig_pareto, width="stretch", key="t17_pareto_scatter")

    # Pareto Metrik Rozetleri
    pm_c1, pm_c2, pm_c3, pm_c4 = st.columns(4)
    with pm_c1:
        st.metric(label="Pareto Çözüm Sayısı (|F1|)", value=f"{len(pareto_solutions)} Adet", help="Baskın olmayan (Non-dominated) optimal çizelgelerin sayısı.")
    with pm_c2:
        st.metric(label="Nihai Hiper-Hacim (Hypervolume)", value=f"{metrics['final_hypervolume']:,.0f}", help="Pareto cephesinin kapsadığı toplam çözüm uzayı hacmi (Yüksek olması iyidir).")
    with pm_c3:
        st.metric(label="Deb's Yayılım Metriği (Spread Δ)", value=f"{metrics['final_spread_delta']:.3f}", help="Pareto çözümlerinin homojen dağılım katsayısı (0'a yakın olması homojendir).")
    with pm_c4:
        st.metric(label="Diz Noktası Toplam Cezası (Z)", value=f"{knee_sol['total_score']}", help="Altın denge noktasındaki toplam yumuşak kısıt ceza puanı.")

    # 💡 Yönetimsel Ödünleşim & Altın Denge Bilgi Kartı
    render_tradeoff_insight_banner(ext_f1_sol, ext_f2_sol, knee_sol)

    st.divider()

    # ==============================================================================
    # 4. BÖLÜM: İNTERAKTİF ÇİZELGE DENETÇİSİ & KARAR DESTEK SİSTEMİ (DSS)
    # ==============================================================================
    st.markdown("### 🔍 4. İnteraktif Çizelge Denetçisi & Karar Destek Sistemi (DSS)")
    st.caption("Pareto ön cephesinden bir yönetimsel strateji veya özel çözüm seçerek ilgili çizelgenin tüm vardiya ısı haritasını, ceza dökümünü ve personel iş yükü dağılımını inceleyiniz.")

    # Strateji / Çözüm Seçici
    strategy_options = [
        "⭐ 1. Strateji: Diz Noktası (Knee Point / Altın Denge - Önerilen)",
        "🏭 2. Strateji: İşletme Odaklı Uç Çözüm (Min f₁ - Üretim Öncelikli)",
        "👨‍🏭 3. Strateji: Çalışan Odaklı Uç Çözüm (Min f₂ - Sendika / Ergonomi Öncelikli)",
        "📋 4. Seçenek: Pareto Cephesindeki Özel Bir Çözümü İncele"
    ]

    sel_strategy = st.radio("🎯 İncelenecek Çizelge Stratejisini Seçin:", strategy_options, index=0, horizontal=True, key="t17_strategy_radio")

    if "1. Strateji" in sel_strategy:
        selected_schedule_dict = knee_sol
        strategy_badge = "⭐ Diz Noktası (Knee Point / Altın Denge)"
        strategy_desc = "Her iki taraftan da minimum tavizle maksimum doyum sağlayan dengeli uzlaşma noktası."
    elif "2. Strateji" in sel_strategy:
        selected_schedule_dict = ext_f1_sol
        strategy_badge = "🏭 İşletme Odaklı Uç Çözüm (Min f₁)"
        strategy_desc = "Posta takım bütünlüğü kusursuzdur, usta eksikliği yoktur; çalışan izin talepleri esnetilmiştir."
    elif "3. Strateji" in sel_strategy:
        selected_schedule_dict = ext_f2_sol
        strategy_badge = "👨‍🏭 Çalışan Odaklı Uç Çözüm (Min f₂)"
        strategy_desc = "Kişisel izinlerin tamamına yakını verilmiştir, gece nöbetleri eşittir; postalar bölünmüştür."
    else:
        pareto_labels = [f"Çözüm #{i+1} | f₁={s['f1']:.0f}, f₂={s['f2']:.0f} (Z={s['total_score']})" for i, s in enumerate(pareto_solutions)]
        sel_idx = st.selectbox("Pareto Cephesinden İncelemek İstediğiniz Çözümü Seçin:", range(len(pareto_solutions)), format_func=lambda i: pareto_labels[i], key="t17_custom_sol_select")
        selected_schedule_dict = pareto_solutions[sel_idx]
        strategy_badge = f"📋 Pareto Çözümü #{sel_idx+1}"
        strategy_desc = f"Pareto ön cephesinde f₁={selected_schedule_dict['f1']:.0f} ve f₂={selected_schedule_dict['f2']:.0f} dengesine sahip özel çizelge."

    # Seçili Çözümün KPI Kartları
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
        <div class="metric-card" style="border-left: 4px solid #10b981;">
            <div class="metric-label">🎯 Toplam Ceza Skoru (Z)</div>
            <div class="metric-value" style="color: #047857;">{selected_schedule_dict['total_score']}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        feas_text = "✅ %100 GEÇERLİ" if selected_schedule_dict['is_feasible'] else f"❌ {selected_schedule_dict['hard_violations_count']} İhlal"
        feas_color = "#059669" if selected_schedule_dict['is_feasible'] else "#dc2626"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid {feas_color};">
            <div class="metric-label">🛡️ Sert Kısıt Uygunluğu</div>
            <div class="metric-value" style="color: {feas_color};">{feas_text}</div>
        </div>
        """, unsafe_allow_html=True)

    # Sert Kısıt Bildirim Kartı
    render_hard_constraints_status_card(
        selected_schedule_dict['is_feasible'],
        selected_schedule_dict['hard_violations_count'],
        selected_schedule_dict['hard_violation_logs'],
        solver_name="NSGA-II (Deb et al., 2002)"
    )

    # Seçili Çözümün Görsel Grafikleri
    render_schedule_heatmap(
        selected_schedule_dict['schedule'],
        active_workers,
        n_days,
        key_prefix="t17_sel",
        title=f"Vardiya Çizelgesi Isı Haritası ({strategy_badge})"
    )

    g_col1, g_col2 = st.columns(2)
    with g_col1:
        render_penalties_chart(
            selected_schedule_dict['penalties'],
            key_prefix="t17_sel",
            title=f"Yumuşak Kısıt Ceza Dağılımı ({strategy_badge})"
        )
    with g_col2:
        render_workload_chart(
            selected_schedule_dict['schedule'],
            active_workers,
            key_prefix="t17_sel",
            title=f"Çalışan İş Yükü ve Gece Nöbeti Dağılımı ({strategy_badge})"
        )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        render_posta_load_chart(
            selected_schedule_dict['schedule'],
            active_workers,
            key_prefix="t17_sel",
            title=f"Posta Yük ve Gece Dağılımı ({strategy_badge})"
        )
    with p_col2:
        render_shift_posta_stacked_chart(
            selected_schedule_dict['schedule'],
            active_workers,
            n_days,
            key_prefix="t17_sel",
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
    # 5. BÖLÜM: ÇOK AMAÇLI EVRİMSEL YAKINSAMA ANALİTİKLERİ
    # ==============================================================================
    st.markdown("### 📊 5. Çok Amaçlı Evrimsel Yakınsama & Performans Analitikleri")

    gens_list = list(range(1, len(metrics['hv_history']) + 1))
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 1️⃣ Hiper-Hacim (Hypervolume - HV) Genleşme Eğrisi")
        fig_hv = px.line(
            x=gens_list,
            y=metrics['hv_history'],
            labels={"x": "Jenerasyon (Nesil)", "y": "Hiper-Hacim (HV)"}
        )
        fig_hv.update_traces(line_color="#059669", line_width=2.5)
        fig_hv.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=320)
        st.plotly_chart(fig_hv, width="stretch", key="t17_fig_hv")

    with c2:
        st.markdown("##### 2️⃣ Pareto Cephesi Büyüklüğü (Frontier Size) Eğrisi")
        fig_fs = px.line(
            x=gens_list,
            y=metrics['front_size_history'],
            labels={"x": "Jenerasyon (Nesil)", "y": "Pareto Çözüm Sayısı (|F1|)"}
        )
        fig_fs.update_traces(line_color="#8b5cf6", line_width=2.5)
        fig_fs.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=320)
        st.plotly_chart(fig_fs, width="stretch", key="t17_fig_fs")

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
    st.plotly_chart(fig_objs, width="stretch", key="t17_fig_objs")

    st.divider()

    # ==============================================================================
    # 6. BÖLÜM: 3 TEMEL YÖNETİMSEL STRATEJİ KARŞILAŞTIRMA MATRİSİ
    # ==============================================================================
    st.markdown("### 📋 6. Yönetimsel Strateji Ödünleşim Matrisi (Trade-off Matrix)")

    def _calc_stats(sol_dict):
        reqs = sol_dict['request_details']
        total_reqs = len(reqs)
        fulfilled = sum(1 for r in reqs if "Karşılandı" in r['durum'])
        pct_req = round((fulfilled / max(1, total_reqs)) * 100, 1)
        
        pen = sol_dict['penalties']
        posta_p = pen.get('Posta Takım Bütünlüğü İhlali', 0)
        usta_p = pen.get('Kıdem / Usta Eksikliği', 0)
        circ_p = pen.get('Sirkadiyen Ritim İhlali (Akşam->Gündüz)', 0)

        # Gece nöbeti standart sapması
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
