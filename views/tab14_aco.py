"""
================================================================================
  VIEWS/TAB14_ACO.PY - SEKME 14: ANT COLONY OPTIMIZATION (ACO)
================================================================================
  Bu modül, Karınca Kolonisi Optimizasyonu (Ant Colony Optimization - ACO)
  ve Max-Min Ant System (MMAS) metasezgisel optimizasyon algoritmasının
  arayüzünü, matematiksel ilkelerini, feromon dinamiklerini ve
  canlı analitik görselleştirmelerini sunar.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.aco_solver import run_ant_colony_optimization
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


def render_tab14(params):
    """Sekme 14 içeriğini çizer: Ant Colony Optimization (ACO) Karınca Kolonisi Çözücüsü ve Analiz Panelleri."""
    st.markdown("## 🐜 Sekme 14: Ant Colony Optimization (ACO - Karınca Kolonisi Optimizasyonu)")
    st.markdown("""
    <div style="background-color: #fefce8; border-left: 5px solid #ca8a04; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; color: #854d0e;">
        📌 <b>Karınca Kolonisi Paradigması (Dorigo, 1992; Stützle & Hoos, 2000 - MMAS):</b> Gerçek karıncaların yiyecek ararken patikalara bıraktıkları 
        <b>feromon (kimyasal koku izi)</b> yoğunluğuna ve sezgisel yol cazibesine göre en kısa patikayı keşfetme mantığına dayanır. 
        Vardiya probleminde; sirkadiyen ritme uyan, posta bütünlüğünü koruyan ve dengeli nöbet sağlayan başarılı vardiya atamalarında feromon yoğunluğu pekişir.
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 1. TEORİK BİLGİ VE MATEMATİKSEL MODEL KARTLARI
    # ==============================================================================
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #ca8a04;">
            <div class="card-title" style="color: #a16207;">🐜 3 Temel Karınca Kolonisi Mekanizması</div>
            <ul style="line-height: 1.65; font-size: 0.92rem; color: #1e293b;">
                <li><b>1. Feromon Hafızası (Pheromone Trails - &tau;):</b> Karıncaların önceki turlarda bulduğu kaliteli vardiya atamalarının kolektif ortak hafızasıdır (&alpha; ağırlığı ile yönetilir).</li>
                <li><b>2. Sezgisel Görünürlük (Heuristic Visibility - &eta;):</b> Bir vardiya atamasının yerel kurallara (kişisel izin talebi, kıdemli usta ihtiyacı, dinlenme) anlık cazibesidir (&beta; ağırlığı ile yönetilir).</li>
                <li><b>3. Buharlaşma & Unutma (Evaporation - &rho;):</b> Zamanla feromonların buharlaşmasıdır; koloninin erken kilitlenmesini (stagnation) önler ve yeni alternatif patikaları keşfetmeyi sağlar.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #2563eb;">
            <div class="card-title" style="color: #1d4ed8;">🏗️ Vardiya Problemi İçin Çizge & Matris ACO Modeli</div>
            <ul style="line-height: 1.65; font-size: 0.92rem; color: #1e293b;">
                <li><b>Çizge Karar Düğümleri:</b> <i>N</i> işçi &times; <i>D</i> gün matrisinde her hücre (<i>i, t</i>) için <i>k</i> &isin; {0, 1, 2, 3} vardiya tercihi bir çizge adımıdır.</li>
                <li><b>Max-Min Ant System (MMAS) Sınırları:</b> Feromonların aşırı birikip tek bir çözüme erken kilitlenmesini veya tamamen sönmesini önlemek için [&tau;<sub>min</sub>, &tau;<sub>max</sub>] sınırları uygulanır.</li>
                <li><b>Daemon Actions (Lokal İyileştirme):</b> Her turda en başarılı karıncanın çizelgesine yerel mikro-takaslar uygulanarak ceza skoru derinlemesine minimize edilir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Matematiksel Formül Gösterimi (LaTeX)
    st.markdown("##### 🧮 Karınca Kolonisi Optimizasyonu Matematiksel Güncelleme Denklemleri:")
    st.latex(r"P_{i, t, k} = \frac{\left[ \tau_{i, t, k} \right]^\alpha \cdot \left[ \eta_{i, t, k} \right]^\beta}{\sum_{k' \in \text{Geçerli}} \left[ \tau_{i, t, k'} \right]^\alpha \cdot \left[ \eta_{i, t, k'} \right]^\beta}")
    st.latex(r"\tau_{i, t, k}(t+1) = (1 - \rho) \cdot \tau_{i, t, k}(t) \;+\; \Delta \tau_{i, t, k}^{\text{best}}, \quad \Delta \tau = \frac{Q}{1 + Z(S^{\text{best}})}")
    st.latex(r"\tau_{\min} \le \tau_{i, t, k} \le \tau_{\max}")

    st.caption(r"Burada $P_{i,t,k}$ karıncanın $i$. işçiye $t$. günde $k$ vardiyasını atama olasılığı, $\tau$ feromon yoğunluğu, $\eta$ sezgisel cazibe, $\rho$ buharlaşma oranı ve $Z(S)$ ceza fonksiyonudur.")

    st.divider()

    # ==============================================================================
    # 2. GÜÇLÜ YANLAR, SINIRLAR VE ÖDÜNLEŞİMLER (TRADE-OFFS)
    # ==============================================================================
    st.markdown("### ⚖️ Ant Colony Optimization (ACO) Güçlü Yanları, Sınırları ve Mühendislik Dengesi")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div style="background-color: #fefce8; border: 2px solid #ca8a04; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
            <h4 style="color: #a16207; margin-top: 0;">🌟 ACO'nun En Güçlü Yanları (Strengths)</h4>
            <ul style="font-size: 0.9rem; color: #1e293b; line-height: 1.6; padding-left: 18px; margin-bottom: 0;">
                <li><b>Pozitif Geri Besleme ile Yönelim:</b> Kaliteli vardiya dizilimleri feromonla ödüllendirildikçe koloni hızla en uygun çizelge deseninde hizalanır.</li>
                <li><b>Buharlaşma ile Esneklik:</b> Feromon buharlaşması sayesinde algoritma kötüleşen eski tercihlere takılmaz; yerel çukurlardan dinamik olarak kurtulur.</li>
                <li><b>Sezgisel Bilgi Entegrasyonu:</b> Kişisel izin talepleri ve usta kısıtları <i>&eta;</i> (eta) matrisiyle doğrudan karınca kararlarına rehberlik eder.</li>
                <li><b>Paralel Çoklu Keşif:</b> Kolonideki her karınca arama uzayının farklı bir bölgesini aynı anda test eder.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background-color: #f8fafc; border: 2px solid #94a3b8; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
            <h4 style="color: #475569; margin-top: 0;">⚠️ Sınırları ve Dikkat Edilmesi Gerekenler (Trade-offs)</h4>
            <ul style="font-size: 0.9rem; color: #1e293b; line-height: 1.6; padding-left: 18px; margin-bottom: 0;">
                <li><b>Erken Durgunluk (Stagnation) Riski:</b> Buharlaşma oranı (<i>&rho;</i>) çok düşük seçilirse feromonlar aşırı birikir ve koloni erken kilitlenebilir.</li>
                <li><b>Hiperparametre Dengesi:</b> <i>&alpha;</i> (geçmiş hafıza) ve <i>&beta;</i> (anlık sezgi) katsayıları problem ölçeğine göre dengeli ayarlanmalıdır.</li>
                <li><b>Hesaplama Maliyeti:</b> Her iterasyonda <i>m</i> adet karıncanın çizelge inşa etmesi ve feromon matrisini güncellemesi ek CPU maliyeti getirir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 3. HİPERPARAMETRE KONTROL PANELİ
    # ==============================================================================
    st.markdown("### 🎛️ Karınca Kolonisi Hiperparametreleri")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    hp1, hp2, hp3, hp4, hp5 = st.columns(5)

    with hp1:
        n_ants = st.slider("🐜 Karınca Sayısı (m)", min_value=5, max_value=60, value=20, step=5, key="aco_n_ants", help="Her iterasyonda arama yapan karınca sayısı.")

    with hp2:
        max_iter = st.slider("🔄 Maksimum İterasyon", min_value=10, max_value=200, value=60, step=10, key="aco_max_iter", help="Toplam karınca turu sayısı.")

    with hp3:
        evap_rate = st.slider("💨 Buharlaşma Oranı (ρ)", min_value=0.02, max_value=0.50, value=0.15, step=0.01, key="aco_evap", help="Her turda eski feromonların silinme yüzdesi.")

    with hp4:
        alpha_val = st.slider("🧠 Feromon Ağırlığı (α)", min_value=0.2, max_value=4.0, value=1.0, step=0.2, key="aco_alpha", help="Geçmiş kolektif feromon hafızasının etki kuvveti.")

    with hp5:
        beta_val = st.slider("👁️ Sezgisel Ağırlık (β)", min_value=0.2, max_value=5.0, value=2.0, step=0.2, key="aco_beta", help="Kişisel izin ve kural cazibesinin etki kuvveti.")

    # ==============================================================================
    # 4. HESAPLAMA YÜKÜ VE ÖN-ANALİZ PANELİ
    # ==============================================================================
    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    # Her iterasyonda: m karınca çizelge inşası + en iyi karıncaya 12 mikro-onarım takası
    aco_est_calls = int((n_ants * max_iter) + (12 * max_iter) + 1)
    aco_total_ms = aco_est_calls * cost_ms
    aco_time_str = f"{aco_total_ms/1000:.2f} sn" if aco_total_ms >= 1000 else f"{aco_total_ms:.0f} ms"

    st.markdown("#### 🧮 Karınca Kolonisi Değerlendirme & Süre Tahmin Paneli (Ön-Analiz)")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        render_evaluator_cost_badge(aco_est_calls, cost_ms, label="Tahmini Ceza Değerlendirme (m Karınca × İterasyon + Lokal Onarım)")
    with e_col2:
        st.metric(
            label="Koloni Arama Kapasitesi",
            value=f"{n_ants} Karınca",
            delta=f"{max_iter} Tur × {n_ants} = {n_ants * max_iter:,} Karınca Gezintisi (~{aco_time_str})",
            help="Koloninin her turda paralel olarak keşfettiği bağımsız çizelge patikası sayısı."
        )

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 Ant Colony Optimization (ACO) Çözücüsünü Çalıştır", type="primary", width="stretch", key="btn_run_t14")

    if not run_btn and "res_t14" not in st.session_state:
        st.warning("👈 Karınca Kolonisi Optimizasyonunu başlatmak için yukarıdaki **'🚀 ACO Çözücüsünü Çalıştır'** butonuna basınız.")
        return

    # ==============================================================================
    # 5. ALGORİTMA ÇALIŞTIRMA & CANLI STREAMLIT GÖRSELLEŞTİRME
    # ==============================================================================
    if run_btn:
        stream_enabled = params.get('live_stream_enabled', True)
        stream_interval = max(2, int(params.get('live_stream_interval', 50) / 10))

        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="Ant Colony Optimization (ACO)",
            max_steps=max_iter,
            unit_name="Karınca Turu",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t14_tracker"
        )

        results = run_ant_colony_optimization(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            weights=weights,
            n_ants=n_ants,
            max_iterations=max_iter,
            evaporation_rate=evap_rate,
            alpha=alpha_val,
            beta=beta_val,
            seed=42,
            custom_workers=custom_workers,
            callback=tracker.update if stream_enabled else None
        )
        tracker.finish(max_iter, results.get('final_score'))
        results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t14"] = results
    else:
        results = st.session_state["res_t14"]
        if 'stream_data' in results and results['stream_data']:
            render_live_stream_summary(results['stream_data'], key_prefix="t14_summary")

    # ==============================================================================
    # 6. SERT KISITLAR VE METRİK KARTLARI
    # ==============================================================================
    render_hard_constraints_status_card(
        is_feasible=results.get('is_feasible', True),
        hard_violations_count=results.get('hard_violations_count', 0),
        hard_violation_logs=results.get('hard_violation_logs', []),
        solver_name="Ant Colony Optimization (ACO)",
        meta=results.get('meta')
    )

    render_metaheuristic_metric_cards(results)

    st.divider()

    # ==============================================================================
    # 7. GRAFİKLER & ÖZEL FEROMON YOĞUNLUK HARİTASI
    # ==============================================================================
    st.markdown("### 🎨 Karınca Kolonisi Analitik Grafikleri & Yakınsama Analizi")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ İterasyona Göre Ceza Skoru İyileşme Eğrisi (Convergence)")
        hist = results['score_history']
        df_hist = pd.DataFrame({
            "İterasyon": list(range(len(hist))),
            "Ceza Skoru (Z)": hist
        })
        fig_conv = px.line(
            df_hist,
            x="İterasyon",
            y="Ceza Skoru (Z)",
            markers=True,
            color_discrete_sequence=["#ca8a04"]
        )
        fig_conv.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc"
        )
        st.plotly_chart(fig_conv, width="stretch", key="t14_fig_conv")

    with g_col2:
        st.markdown("##### 2️⃣ 🐜 Koloni Feromon Yoğunluk Matrisi (Pheromone Trail Heatmap)")
        st.caption(r"Karıncaların atanan vardiya patikalarında biriktirdiği nihai feromon kuvveti ($[\tau_{\min}, \tau_{\max}]$). Yüksek değerler koloninin üzerinde uzlaştığı kilit atamaları gösterir.")
        
        active_phero = results.get('meta', {}).get('active_pheromone_matrix')
        if active_phero is not None:
            phero_arr = np.array(active_phero)
            fig_phero = px.imshow(
                phero_arr,
                labels=dict(x="Gün", y="İşçi", color="Feromon (τ)"),
                x=[f"G{d+1}" for d in range(num_days)],
                y=[w['name'] for w in results['workers']],
                color_continuous_scale="YlOrBr"
            )
            fig_phero.update_layout(
                height=360,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_phero, width="stretch", key="t14_fig_phero")

    # Standart Analitik Grafikler & Çizelge Tablosu
    render_standard_schedule_analytics(
        results=results,
        num_days=num_days,
        key_prefix="t14_aco",
        solver_name="Ant Colony Optimization (ACO)",
        start_chart_num=3,
        show_request_details=True
    )
