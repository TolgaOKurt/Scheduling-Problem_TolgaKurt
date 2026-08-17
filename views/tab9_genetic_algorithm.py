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
    render_request_details_expander
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
        generations = st.slider("Jenerasyon Sayısı (G)", 20, 300, 100, 10, key="ga_gen")
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
    total_evaluations = pop_size + generations * (pop_size - elitism_count)
    # Deneysel Benchmark Ölçümü: 1 kromozom değerlendirmesi N işçi ve D gün bazında ~0.00157 ms
    est_ga_cpu_ms = round(total_evaluations * num_workers * num_days * 0.00157, 1)

    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    ga_total_ms = total_evaluations * cost_ms
    ga_time_str = f"{ga_total_ms/1000:.2f} sn" if ga_total_ms >= 1000 else f"{ga_total_ms:.0f} ms"

    st.markdown("#### 🧮 Evrimsel Hesaplama & Ceza Değerlendirici Tahmin Paneli (Ön-Analiz)")

    pred_c1, pred_c2 = st.columns(2)
    with pred_c1:
        st.metric(
            label="Tahmini Ceza Değerlendirme (P × (1+G))",
            value=f"{total_evaluations:,} Çağrı",
            delta=f"{cost_ms:.2f} ms * {total_evaluations:,} = {ga_time_str}",
            delta_color="off",
            help="Tüm jenerasyonlar boyunca popülasyondaki kromozomların toplam ceza fonksiyonu değerlendirme sayısı."
        )
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
        with st.spinner("⏳ Genetik Algoritma Popülasyonu Evrimleştiriliyor... Lütfen Bekleyiniz..."):
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
                custom_workers=custom_workers
            )
            st.session_state["res_t9"] = results
    else:
        results = st.session_state["res_t9"]

    # --- ÇALIŞMAYI BİTİRME NEDENİ BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Başlangıç Skoru</div>
        <div class="metric-value" style="color: #dc2626;">{results['initial_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">En İyi Skor</div>
        <div class="metric-value" style="color: #059669;">{results['final_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İyileşme Oranı</div>
        <div class="metric-value" style="color: #1d4ed8;">%{results['improvement_rate']}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Jenerasyon</div>
        <div class="metric-value" style="color: #8b5cf6;">{results['generations_run']} Nesil</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        eval_c = results.get('eval_count', total_evaluations)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Ceza Çağrısı (Evaluator)</div>
        <div class="metric-value" style="color: #6366f1;">{eval_c:,} Adet</div>
        </div>""", unsafe_allow_html=True)

    with m6:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Evrim Süresi</div>
        <div class="metric-value" style="color: #059669;">{results['exec_time_ms']} ms</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Genetik Algoritma Evrimsel Yakınsama & Analitik Grafikler")

    gens = list(range(1, len(results['best_score_history']) + 1))
    
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ Jenerasyon Bazlı Evrimsel Yakınsama Grafiği (Best vs Avg Score)")
        fig_ga_conv = go.Figure()
        fig_ga_conv.add_trace(go.Scatter(
            x=gens, y=results['best_score_history'],
            name="En İyi Birey Skoru Z_best",
            line=dict(color="#059669", width=2.5)
        ))
        fig_ga_conv.add_trace(go.Scatter(
            x=gens, y=results['avg_score_history'],
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
            x=gens, y=results['diversity_history'],
            labels={"x": "Jenerasyon", "y": "Popülasyon Çeşitliliği (Std Dev)"},
            line_shape="linear"
        )
        fig_div.update_traces(line_color="#ec4899", line_width=2.5)
        fig_div.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_div, width="stretch", key="t9_fig_div")

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        render_schedule_heatmap(results['schedule'], results['workers'], num_days, key_prefix="t9_ga", title="3️⃣ GA En İyi Vardiya Dağılım Isı Haritası (Heatmap)")

    with g_col4:
        render_workload_chart(results['schedule'], results['workers'], key_prefix="t9_ga", title="4️⃣ GA Çalışan İş Yükü ve Gece Nöbeti Dağılımı", color_seq=["#059669", "#dc2626"])

    g_col5, g_col6 = st.columns(2)

    with g_col5:
        render_penalties_chart(results['penalties'], key_prefix="t9_ga", title="5️⃣ GA Yumuşak Kısıt Ceza Puanı Dağılımı")

    with g_col6:
        render_posta_load_chart(results['schedule'], results['workers'], key_prefix="t9_ga", title="6️⃣ GA Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")

    # 7. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI
    render_shift_posta_stacked_chart(results['schedule'], results['workers'], num_days, key_prefix="t9_ga", title="7️⃣ Gün ve Vardiya Bazında Posta Dağılımı (Gündüz, Akşam ve Gece Vardiyalarında Hangi Postadan Kaç Kişi Var?)")

    st.divider()

    # --- FULL SCHEDULE MATRIX TABLE ---
    render_schedule_matrix_table(results['schedule'], results['workers'], num_days, title="🗓️ Genetik Algoritma Tarafından Üretilen Tam Vardiya Çizelgesi")
    
    # --- KİŞİSEL İZİN TALEPLERİ DETAY RAPORU ---
    if 'request_details' in results:
        render_request_details_expander(results['request_details'])
