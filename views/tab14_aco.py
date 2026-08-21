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
    aco_est_calls = int((n_ants * max_iter) + (12 * max_iter) + 1)

    st.markdown("#### 🧮 Karınca Kolonisi Değerlendirme Tahmin Paneli (Ön-Analiz)")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        render_evaluator_cost_badge(aco_est_calls, cost_ms, label="Tahmini Ceza Değerlendirme (m Karınca × İterasyon + Lokal Onarım)")
    with e_col2:
        st.metric(
            label="Koloni Arama Kapasitesi",
            value=f"{n_ants} Karınca",
            delta=f"{max_iter} Tur × {n_ants} = {n_ants * max_iter:,} Karınca Gezintisi",
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

    render_metaheuristic_metric_cards(results, move_label="Kabul / Tur")

    st.divider()

    # ==============================================================================
    # 7. GELİŞMİŞ KARINCA KOLONİSİ ANALİTİK VE AÇIKLAYICI GRAFİKLERİ
    # ==============================================================================
    st.markdown("### 🎨 Karınca Kolonisi Analitik Grafikleri & İç Dinamik Analizi")
    st.caption("Algoritmanın çalışma mantığını, karıncaların sürü halindeki arama davranışını ve feromon hafızasının evrimini gösteren analitik paneller:")

    meta = results.get('meta', {})

    # 1. SATIR: ÇEŞİTLİLİK BANDI & FEROMON DİNAMİKLERİ
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ 🐜 Koloni Çeşitlilik Bandı (Best vs. Mean vs. Worst Ant)")
        st.caption("Her turdaki 20 karıncanın en iyi, ortalama ve en kötü skorları. Bandın daralması koloninin tek bir optimal çizelge üzerinde uzlaştığını (yakınsadığını) kanıtlar.")
        
        iter_best = meta.get('iter_best_history', results['score_history'])
        iter_mean = meta.get('iter_mean_history', results['score_history'])
        iter_worst = meta.get('iter_worst_history', results['score_history'])
        iters = list(range(len(iter_best)))

        fig_band = go.Figure()
        # Üst sınır (Worst)
        fig_band.add_trace(go.Scatter(
            x=iters, y=iter_worst,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Alt sınır (Best) ve Aradaki Bant Dolgusu
        fig_band.add_trace(go.Scatter(
            x=iters, y=iter_best,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(202, 138, 4, 0.18)',
            name='Koloni Çeşitlilik Bandı (Min - Max)',
            hoverinfo='skip'
        ))
        # En kötü karınca çizgisi
        fig_band.add_trace(go.Scatter(
            x=iters, y=iter_worst,
            mode='lines',
            line=dict(color='#ef4444', width=1.5, dash='dot'),
            name='En Kötü Karınca (Max Z)'
        ))
        # Ortalama karınca çizgisi
        fig_band.add_trace(go.Scatter(
            x=iters, y=iter_mean,
            mode='lines',
            line=dict(color='#2563eb', width=2),
            name='Ortalama Karınca (Ortalama Z)'
        ))
        # Küresel en iyi çizgisi
        fig_band.add_trace(go.Scatter(
            x=iters, y=results['score_history'],
            mode='lines+markers',
            marker=dict(size=4),
            line=dict(color='#ca8a04', width=3),
            name='Küresel En İyi (Global Best Z)'
        ))
        fig_band.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            xaxis_title="İterasyon (Karınca Turu)",
            yaxis_title="Ceza Skoru (Z)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_band, width="stretch", key="t14_fig_band")

    with g_col2:
        st.markdown("##### 2️⃣ 📈 Feromon Dinamikleri: Aktif vs. Terk Edilen Patikalar")
        st.caption(r"Buharlaşma ($\rho$) ve Takviyenin ($\Delta \tau$) koku ayrışması: Karıncaların seçtiği **Aktif Patikaların güçlenmesi** ile seçilmeyen yolların $\tau_{\min}=0.1$'e buharlaşması.")

        active_phero_hist = meta.get('active_pheromone_history', [])
        inactive_phero_hist = meta.get('inactive_pheromone_history', [])
        mean_phero = meta.get('mean_pheromone_history', [])
        iters_phero = list(range(len(mean_phero)))

        fig_phero_dyn = go.Figure()
        
        # 1. Aktif Çözüm Patikasındaki Feromon
        if active_phero_hist:
            fig_phero_dyn.add_trace(go.Scatter(
                x=iters_phero, y=active_phero_hist,
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(color='#ea580c', width=3),
                name='🟠 Aktif Çözüm Patikası (Seçilen Yollar)'
            ))
        
        # 2. Genel Matris Ortalaması
        fig_phero_dyn.add_trace(go.Scatter(
            x=iters_phero, y=mean_phero,
            mode='lines',
            line=dict(color='#ca8a04', width=2),
            name='🟡 Tüm Matris Ortalaması (Genel τ)'
        ))

        # 3. Terk Edilen / Seçilmeyen Yollar
        if inactive_phero_hist:
            fig_phero_dyn.add_trace(go.Scatter(
                x=iters_phero, y=inactive_phero_hist,
                mode='lines',
                line=dict(color='#94a3b8', width=1.8, dash='dot'),
                name='⚪ Terk Edilen Yollar (Buharlaşan %75)'
            ))

        fig_phero_dyn.add_hline(y=0.1, line_dash="dash", line_color="#64748b", annotation_text="τ_min (0.1 Alt Taban)")
        
        fig_phero_dyn.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            xaxis_title="İterasyon (Karınca Turu)",
            yaxis_title="Feromon Şiddeti (τ)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_phero_dyn, width="stretch", key="t14_fig_phero_dyn")

    # 2. SATIR: GÜÇ DENGESİ & TAKAS KABUL ORANI
    g_col3, g_col4 = st.columns(2)

    with g_col3:
        st.markdown("##### 3️⃣ ⚖️ Feromon Hafızası (τᵅ) vs. Sezgisel Cazibe (ηᵝ) Güç Dengesi")
        st.caption("Karıncaların her vardiya türünde karar verirken geçmiş tecrübeye (feromon) mi yoksa kural cazibesine (sezgi) mi daha çok güvendiğini gösterir.")

        shift_labels = meta.get('shift_labels', ["0: İzin (OFF)", "1: Gündüz", "2: Akşam", "3: Gece"])
        phero_pow = meta.get('shift_pheromone_power', [1.0, 1.0, 1.0, 1.0])
        heur_pow = meta.get('shift_heuristic_power', [1.0, 1.0, 1.0, 1.0])

        fig_balance = go.Figure()
        fig_balance.add_trace(go.Bar(
            x=shift_labels,
            y=phero_pow,
            name='Feromon Hafızası (τ^α)',
            marker_color='#ca8a04'
        ))
        fig_balance.add_trace(go.Bar(
            x=shift_labels,
            y=heur_pow,
            name='Sezgisel Cazibe (η^β)',
            marker_color='#2563eb'
        ))
        fig_balance.update_layout(
            barmode='group',
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            xaxis_title="Vardiya Türü",
            yaxis_title="Ortalama Çekicilik Skoru",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_balance, width="stretch", key="t14_fig_balance")

    with g_col4:
        st.markdown("##### 4️⃣ 🎯 Karınca Takas Karar Dağılımı & Sert Kısıt Kabul Oranı")
        st.caption("Karıncaların denediği takas hamlelerinin kabul/ret istatistiği. Sert kısıtların (4 MYK / 11s dinlenme) arama uzayını ne kadar daralttığını kanıtlar.")

        swap_stats = meta.get('swap_stats', {'prob_rejected': 0, 'hard_rejected': 0, 'accepted': 0})
        labels_donut = [
            "🟢 Kabul Edilen Geçerli Takaslar",
            "🔴 Sert Kısıt (MYK/Dinlenme) Engeli",
            "🟡 Düşük Feromon Olasılığı Reddi"
        ]
        values_donut = [
            swap_stats.get('accepted', 0),
            swap_stats.get('hard_rejected', 0),
            swap_stats.get('prob_rejected', 0)
        ]
        colors_donut = ["#16a34a", "#dc2626", "#eab308"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels_donut,
            values=values_donut,
            hole=0.45,
            marker_colors=colors_donut,
            textinfo="percent+label",
            insidetextorientation="radial"
        )])
        fig_donut.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_donut, width="stretch", key="t14_fig_donut")

    # 3. SATIR: KOLONİ FEROMON YOĞUNLUK MATRİSİ (ISI HARİTASI)
    st.markdown("##### 5️⃣ 🐜 Koloni Feromon Yoğunluk Matrisi (Pheromone Trail Heatmap)")
    st.caption(r"Karıncaların atanan vardiya patikalarında biriktirdiği nihai feromon kuvveti ($[\tau_{\min}, \tau_{\max}]$). Yüksek değerler koloninin üzerinde uzlaştığı kilit atamaları gösterir.")
    
    active_phero = meta.get('active_pheromone_matrix')
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
            height=380,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_phero, width="stretch", key="t14_fig_phero")

    # Standart Analitik Grafikler & Çizelge Tablosu
    render_standard_schedule_analytics(
        results=results,
        num_days=num_days,
        key_prefix="t14_aco",
        solver_name="Ant Colony Optimization (ACO)",
        start_chart_num=6,
        show_request_details=True
    )
