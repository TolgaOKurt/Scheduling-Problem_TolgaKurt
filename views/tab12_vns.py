"""
================================================================================
  VIEWS/TAB12_VNS.PY - SEKME 12: VARIABLE NEIGHBORHOOD SEARCH (VNS)
================================================================================
  Bu modül, Değişken Komşuluk Araması (Variable Neighborhood Search - VNS)
  metasezgisel optimizasyon algoritmasının arayüzünü, teorik ilkelerini,
  hiyerarşik komşuluk yapılarını (N_1, N_2, N_3), güçlü/güçsüz yönlerini ve
  canlı analitik görselleştirmelerini sunar.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.vns_solver import run_variable_neighborhood_search
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


def render_tab12(params):
    """Sekme 12 içeriğini çizer: Variable Neighborhood Search (VNS) Metasezgisel Çözücüsü ve Analiz Panelleri."""
    st.markdown("## 🔄 Sekme 12: Variable Neighborhood Search (VNS - Değişken Komşuluk Araması)")
    st.markdown("""
    <div style="background-color: #f0fdfa; border-left: 5px solid #0d9488; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; color: #115e59;">
        📌 <b>VNS Paradigması (Mladenović & Hansen, 1997):</b> Tek bir sabit komşuluk yapısında (örneğin sadece 2 işçi takasında) arama yapmak yerine, 
        <b>hiyerarşik çoklu komşuluk uzayları (<i>N</i><sub>1</sub>, <i>N</i><sub>2</sub>, <i>N</i><sub>3</sub>...)</b> tanımlayarak yerel minimum çukurlarına takıldıkça komşuluk ölçeğini sistematik olarak değiştiren modern bir metasezgiseldir.
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 1. TEORİK BİLGİ VE HİYERARŞİK KOMŞULUK KARTLARI
    # ==============================================================================
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #0d9488;">
            <div class="card-title" style="color: #0f766e;">🎯 3 Temel VNS İlkesi ve Çalışma Döngüsü</div>
            <ul style="line-height: 1.65; font-size: 0.92rem; color: #1e293b;">
                <li><b>İlke 1 (Relatif Yerel Optimum):</b> Bir komşuluk yapısına (<i>N</i><sub>1</sub>) göre yerel minimum olan bir çözüm, başka bir komşuluk yapısına (<i>N</i><sub>2</sub> veya <i>N</i><sub>3</sub>) göre yerel minimum olmak zorunda değildir.</li>
                <li><b>İlke 2 (Küresel Optimum Tanımı):</b> Gerçek küresel optimum, <b>tüm olası komşuluk yapılarına göre</b> aynı anda yerel minimum olan çözümdür.</li>
                <li><b>İlke 3 (Çalkalama & İniş - Shaking & Local Search):</b> <i>N<sub>k</sub></i> komşuluğundan rastgele bir aday üretilir (Shaking) ve yerel arama (VND) uygulanır. Eğer iyileşme varsa <i>N</i><sub>1</sub>'e dönülür; yoksa komşuluk <i>k</i> &larr; <i>k</i> + 1 olarak genişletilir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #2563eb;">
            <div class="card-title" style="color: #1d4ed8;">🏗️ Çelik Tesis Modeli İçin 3 Hiyerarşik Komşuluk Yapısı</div>
            <ul style="line-height: 1.65; font-size: 0.92rem; color: #1e293b;">
                <li><b><i>N</i><sub>1</sub> (Mikro Düzey - 2 İşçi Takası):</b> Aynı gün içerisinde çalışan 2 işçinin vardiyaları takas edilir (İnce ayar / Fine tuning).</li>
                <li><b><i>N</i><sub>2</sub> (Orta/Mezo Düzey - Çok Günlü Blok Kaydırma):</b> 1 işçinin veya aynı postadaki 2 işçinin 2 günlük vardiya blokları kaydırılarak sirkadiyen ritim ve izin dengesi geniş pencerede çözülür.</li>
                <li><b><i>N</i><sub>3</sub> (Makro Düzey - 4-Posta Ekip Takası):</b> Bir günde Posta A ile Posta B'nin tamamı takas edilir. Postalar otonom MYK setine sahip olduğundan posta bölünmesi cezası tek hamlede sıfırlanır!</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 2. GÜÇLÜ YANLAR, GÜÇSÜZ YANLAR VE DİĞER ALGORİTMALARDAN FARKLAR
    # ==============================================================================
    st.markdown("### ⚖️ VNS'in Güçlü Yanları, Sınırları ve Diğer Algoritmalarla Karşılaştırması")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div style="background-color: #f0fdf4; border: 2px solid #16a34a; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
            <h4 style="color: #15803d; margin-top: 0;">🌟 VNS'in En Güçlü Yanları (Strengths)</h4>
            <ul style="font-size: 0.9rem; color: #1e293b; line-height: 1.6; padding-left: 18px; margin-bottom: 0;">
                <li><b>Yerel Tuzakları Hiyerarşik Aşma:</b> Hill Climbing tek operatörde tıkanırken, VNS <i>N</i><sub>1</sub> &rarr; <i>N</i><sub>2</sub> &rarr; <i>N</i><sub>3</sub> geçişiyle yerel vadilerden kolayca sıyrılır.</li>
                <li><b>Posta Takım Bütünlüğünde Üstün Başarı:</b> Makro <i>N</i><sub>3</sub> operatörü, 4-Posta takımını bölmeden blok halinde döndürerek sanayide ekip ruhunu korur.</li>
                <li><b>Hafif ve Hızlı Bellek Kullanımı:</b> Tabu Search gibi karmaşık hafıza tablolarına veya GA gibi 50-100 bireylik ağır popülasyon matrislerine ihtiyaç duymaz; tek çözümle yıldırım hızında çalışır.</li>
                <li><b>Parametre Sadeliği:</b> Ayarlanması gereken onlarca karmaşık hiperparametre (sıcaklık soğuma oranı, mutasyon olasılığı vb.) yerine sadece komşuluk tanımlarına odaklanır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div style="background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
            <h4 style="color: #b91c1c; margin-top: 0;">⚠️ VNS'in Güçsüz / Dikkat Edilmesi Gereken Yanları (Limitations)</h4>
            <ul style="font-size: 0.9rem; color: #1e293b; line-height: 1.6; padding-left: 18px; margin-bottom: 0;">
                <li><b>Probleme Özel Komşuluk Tasarımı Şartı:</b> Standart bir VNS operatörü yoktur; <i>N</i><sub>1</sub>, <i>N</i><sub>2</sub>, <i>N</i><sub>3</sub> yapılarının çizelgeleme problemine ve MYK kısıtlarına özel tasarlanması gerekir.</li>
                <li><b>Üst Düzey Komşuluklarda Feasibility Riski:</b> <i>N</i><sub>3</sub> gibi makro takaslar çok agresif uygulanırsa sert kısıtların (11 saat dinlenme, 4 MYK zorunlu ehliyeti) bozulma riski artar.</li>
                <li><b>Matematiksel Optimum Garantisi Yoktur:</b> ILP/MILP gibi küresel optimumu %100 matematiksel olarak ispatlayamaz; ancak sezgisel olarak %98-99 kalitede çözümler üretir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Karşılaştırma Özeti Tablosu
    st.markdown("##### 🔍 VNS'in Diğer 7 Algoritma Karşısındaki Konumu:")
    df_vns_comp = pd.DataFrame({
        "Algoritma": [
            "Hill Climbing (Sekme 7)",
            "Simulated Annealing (Sekme 8)",
            "Genetic Algorithm (Sekme 9)",
            "Tabu Search (Sekme 11)",
            "Variable Neighborhood Search (Sekme 12)"
        ],
        "Arama Uzayı Mekanizması": [
            "Tekil komşuluk (Sadece N_1)",
            "Stokastik sıcaklık kabulü (Metropolis)",
            "Popülasyon evrimi & Çaprazlama",
            "Kısa vadeli hafıza & Yasak listesi",
            "Hiyerarşik Çoklu Komşuluk (N_1 → N_2 → N_3)"
        ],
        "Yerel Tuzaktan Kaçış Stratejisi": [
            "❌ Kaçamaz (İlk tepede durur)",
            "🌡️ Olasılıksal kötüleşen hamle kabulü",
            "🧬 Genetik çeşitlilik ve mutasyon",
            "🤫 Yasaklı hamleleri hafızada kilitleme",
            "🔄 Komşuluk ölçeğini büyüterek (Shaking) vadi dışına atlama"
        ],
        "Posta Bütünlüğü Yaklaşımı": [
            "Yavaş (Tek tek takas)",
            "Orta (Rastgele kabul)",
            "Bozulabilir (Çaprazlama dikiş hataları)",
            "Hızlı (Aspirasyon yönlendirmeli)",
            "🏆 Mükemmel (N_3 doğrudan tüm postayı taşır)"
        ],
        "Hesaplama Yükü & Hızı": [
            "⚡ Çok Hızlı (~100 ms)",
            "🚀 Hızlı (~500 ms)",
            "⏱️ Ağır (~3-8 sn)",
            "🚀 Çok Hızlı (~400 ms)",
            "⚡ Çok Hızlı & Hafif (~250-800 ms)"
        ]
    })
    st.dataframe(df_vns_comp, width="stretch", hide_index=True)

    st.divider()

    # ==============================================================================
    # 3. PARAMETRE KONTROL PANELİ VE ÖN-ANALİZ
    # ==============================================================================
    st.markdown("### 🎛️ VNS İterasyon & Komşuluk Parametre Kontrol Paneli")

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        max_iter = st.slider("Maksimum VNS İterasyon Sayısı (K_max)", 200, 3000, 1000, 100, key="vns_iter", help="Dış VNS döngüsünün toplam adım sayısı.")
    with p_col2:
        max_neighborhoods = st.slider("Kullanılacak Komşuluk Sayısı (K_neigh)", 1, 3, 3, 1, key="vns_k_neigh", help="1: Sadece N_1 (Mikro), 2: N_1 + N_2 (Orta), 3: N_1 + N_2 + N_3 (Tüm Hiyerarşi)")
    with p_col3:
        local_search_depth = st.slider("Lokal Arama Derinliği (VND Depth)", 5, 30, 15, 5, key="vns_ls_depth", help="Her çalkalama sonrasında uygulanacak mikro iyileştirme adımı sayısı.")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # Ön Analiz Tahmin Rozeti
    total_evals = max_iter * (local_search_depth + 1)
    cost_ms = round(params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3)), 3)

    st.markdown("#### 🧮 VNS Arama & Değerlendirme Tahmin Paneli (Ön-Analiz)")
    pred_c1, pred_c2 = st.columns(2)
    with pred_c1:
        render_evaluator_cost_badge(total_evals, cost_ms, label="Tahmini Ceza Değerlendirme (K_max × [L_depth + 1])")
    with pred_c2:
        st.metric(
            label="Hiyerarşik Komşuluk Sayısı",
            value=f"{max_neighborhoods} Düzey",
            delta="N_1 (Mikro) + N_2 (Mezo) + N_3 (Makro)" if max_neighborhoods == 3 else ("N_1 + N_2" if max_neighborhoods == 2 else "N_1"),
            help="Sistematik olarak taranacak komşuluk seviyeleri."
        )

    st.divider()

    # ==============================================================================
    # 4. ÇALIŞTIRMA VE CANLI OPTİMİZASYON AKIŞI
    # ==============================================================================
    run_btn = st.button("🚀 Variable Neighborhood Search (VNS) Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t12")

    if not run_btn and "res_t12" not in st.session_state:
        st.warning("👈 Optimizasyonu başlatmak için yukarıdaki **'🚀 Variable Neighborhood Search (VNS) Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        stream_enabled = params.get('live_stream_enabled', True)
        stream_interval = max(10, int(params.get('live_stream_interval', 50) / 2))

        live_placeholder = st.empty()
        tracker = LiveStreamTracker(
            placeholder=live_placeholder,
            title="Variable Neighborhood Search (VNS)",
            max_steps=max_iter,
            unit_name="İterasyon",
            enabled=stream_enabled,
            stream_interval=stream_interval,
            key_prefix="t12_tracker"
        )

        results = run_variable_neighborhood_search(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            weights=weights,
            max_iterations=max_iter,
            max_neighborhoods=max_neighborhoods,
            local_search_depth=local_search_depth,
            seed=42,
            custom_workers=custom_workers,
            callback=tracker.update if stream_enabled else None,
            stream_interval=stream_interval
        )
        tracker.finish(results.get('total_iterations', max_iter), results.get('final_score'))
        results['stream_data'] = tracker.get_stream_data()
        st.session_state["res_t12"] = results
    else:
        results = st.session_state["res_t12"]
        if params.get('live_stream_enabled', True) and 'stream_data' in results:
            render_live_stream_summary(results['stream_data'])

    # --- BİTİRME NEDENİ VE SERT KISIT UYGUNLUK BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")
    render_hard_constraints_status_card(
        results.get('is_feasible', True),
        results.get('hard_violations_count', 0),
        results.get('hard_violation_logs', []),
        solver_name="Variable Neighborhood Search (VNS)",
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
        <div class="metric-label">VNS En İyi Skor</div>
        <div class="metric-value" style="color: #0d9488;">{results['final_score']}</div>
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
        n_escapes = meta.get('successful_escapes', {})
        n3_count = n_escapes.get(3, 0)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">N3 Posta Takası</div>
        <div class="metric-value" style="color: #0284c7;">{n3_count} Kez</div>
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
    st.markdown("### 🎨 Variable Neighborhood Search (VNS) Yakınsama & Analitik Grafikler")

    score_hist = results.get('score_history', results.get('best_score_history', []))
    curr_scores = meta.get('curr_score_history', results.get('curr_score_history', score_hist))
    iters = list(range(1, len(score_hist) + 1))

    # 1. SATIR: İTERASYON BAZLI YAKINSAMA VE VNS KOMŞULUK BAŞARI DAĞILIMI
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ İterasyon Bazlı VNS Yakınsama & Vadi Aşımı Grafiği (Best vs Current Score)")
        fig_vns_conv = go.Figure()
        fig_vns_conv.add_trace(go.Scatter(
            x=iters, y=score_hist,
            name="Global En İyi Skor Z_best",
            line=dict(color="#0d9488", width=2.5)
        ))
        if len(curr_scores) == len(iters):
            fig_vns_conv.add_trace(go.Scatter(
                x=iters, y=curr_scores,
                name="Mevcut Çözüm Skoru Z_current (Çalkalama & Vadi Çıkışları)",
                line=dict(color="#64748b", width=1.2, dash="dot"),
                opacity=0.75
            ))
        fig_vns_conv.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            xaxis=dict(title="VNS İterasyonu (Adım)"),
            yaxis=dict(title="Toplam Ceza Skoru Z"),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_vns_conv, width="stretch", key="t12_fig_conv")

    with g_col2:
        st.markdown("##### 2️⃣ Hiyerarşik Komşulukların Başarı ve İyileştirme Dağılımı")
        n_usage = meta.get('neighborhood_usage', {1: 0, 2: 0, 3: 0})
        n_succ = meta.get('successful_escapes', {1: 0, 2: 0, 3: 0})

        df_neigh_stats = pd.DataFrame([
            {"Komşuluk Yapısı": "N₁: Mikro (2-İşçi Takası)", "Toplam Deneme": n_usage.get(1, 0), "Başarılı İyileştirme": n_succ.get(1, 0)},
            {"Komşuluk Yapısı": "N₂: Mezo (Blok Kaydırma)", "Toplam Deneme": n_usage.get(2, 0), "Başarılı İyileştirme": n_succ.get(2, 0)},
            {"Komşuluk Yapısı": "N₃: Makro (Posta Takası)", "Toplam Deneme": n_usage.get(3, 0), "Başarılı İyileştirme": n_succ.get(3, 0)},
        ])
        fig_nstats = px.bar(
            df_neigh_stats,
            x="Komşuluk Yapısı",
            y=["Toplam Deneme", "Başarılı İyileştirme"],
            barmode="group",
            color_discrete_sequence=["#0d9488", "#2563eb"],
            text_auto=True
        )
        fig_nstats.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_nstats, width="stretch", key="t12_neigh_stats")

    # 3-7 STANDART ÇİZELGE ANALİTİKLERİ VE MATRİS TABLOSU
    render_standard_schedule_analytics(
        results, num_days, key_prefix="t12_vns",
        solver_name="VNS", start_chart_num=3
    )
