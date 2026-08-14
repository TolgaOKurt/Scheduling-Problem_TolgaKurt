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
        pop_size = st.slider("Popülasyon Büyüklüğü (P)", 20, 200, 50, 10, key="ga_pop")
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

    st.markdown("#### 🧮 Evrimsel Hesaplama & Süre Tahmin Paneli (Ön-Analiz)")

    pred_c1, pred_c2, pred_c3 = st.columns(3)
    with pred_c1:
        st.metric(
            label="Toplam Değerlendirilecek Birey Sayısı",
            value=f"{total_evaluations:,} Kromozom",
            help="Tüm jenerasyonlar boyunca değerlendirilecek toplam birey sayısı."
        )
    with pred_c2:
        st.metric(
            label="Tahmini Evrim Süresi",
            value=f"~{est_ga_cpu_ms:,} ms",
            help="Evrim tamamlanana kadar harcanacak tahmini CPU süresi."
        )
    with pred_c3:
        st.metric(
            label="Popülasyon Koruma Oranı (Elitizm)",
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
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">En İyi Başlangıç Skoru</div>
        <div class="metric-value" style="color: #dc2626;">{results['initial_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Evrimleşmiş En İyi Skor</div>
        <div class="metric-value" style="color: #059669;">{results['final_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İyileşme Oranı</div>
        <div class="metric-value" style="color: #1d4ed8;">%{results['improvement_rate']}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Tamamlanan Jenerasyon</div>
        <div class="metric-value" style="color: #8b5cf6;">{results['generations_run']} Nesil</div>
        </div>""", unsafe_allow_html=True)

    with m5:
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
        st.markdown("##### 3️⃣ GA En İyi Vardiya Dağılım Isı Haritası (Heatmap)")
        fig_ga_map = px.imshow(
            results['schedule'],
            labels=dict(x="Günler", y="Çalışanlar", color="Vardiya (0:OFF, 1:G, 2:A, 3:N)"),
            x=[f"G{d+1}" for d in range(num_days)],
            y=[w['name'] for w in results['workers']],
            color_continuous_scale=[[0, '#cbd5e1'], [0.33, '#fde047'], [0.66, '#f97316'], [1.0, '#1e3a8a']]
        )
        fig_ga_map.update_layout(height=400)
        st.plotly_chart(fig_ga_map, width="stretch", key="t9_fig_map")

    with g_col4:
        st.markdown("##### 4️⃣ GA Çalışan İş Yükü ve Gece Nöbeti Dağılımı")
        work_days = [np.sum(results['schedule'][i, :] > 0) for i in range(num_workers)]
        night_days = [np.sum(results['schedule'][i, :] == 3) for i in range(num_workers)]
        
        df_ga_workload = pd.DataFrame({
            "İşçi": [w['name'] for w in results['workers']],
            "Toplam Çalışma": work_days,
            "Gece Nöbeti": night_days
        })
        
        fig_ga_wl = px.bar(
            df_ga_workload,
            x="İşçi",
            y=["Toplam Çalışma", "Gece Nöbeti"],
            barmode="group",
            color_discrete_sequence=["#059669", "#dc2626"]
        )
        fig_ga_wl.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=400, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_ga_wl, width="stretch", key="t9_fig_wl")

    g_col5, g_col6 = st.columns(2)

    with g_col5:
        st.markdown("##### 5️⃣ GA Yumuşak Kısıt Ceza Puanı Dağılımı")
        df_ga_penalties = pd.DataFrame({
            "Kısıt Tipi": list(results['penalties'].keys()),
            "Ceza Puanı": list(results['penalties'].values())
        })
        fig_ga_pen = px.bar(
            df_ga_penalties,
            x="Ceza Puanı",
            y="Kısıt Tipi",
            orientation="h",
            text="Ceza Puanı",
            color="Ceza Puanı",
            color_continuous_scale="Reds"
        )
        fig_ga_pen.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=340, showlegend=False)
        st.plotly_chart(fig_ga_pen, width="stretch", key="t9_fig_pen")

    with g_col6:
        st.markdown("##### 6️⃣ GA Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")
        posta_data = []
        for p in ['Posta A', 'Posta B', 'Posta C', 'Posta D']:
            p_wids = [w['id'] for w in results['workers'] if w['posta'] == p]
            if len(p_wids) > 0:
                tot_w = np.sum(results['schedule'][p_wids, :] > 0)
                tot_n = np.sum(results['schedule'][p_wids, :] == 3)
                posta_data.append({"Posta": p, "Toplam Vardiya": tot_w, "Gece Vardiyası": tot_n})
        
        df_posta = pd.DataFrame(posta_data)
        fig_posta = px.bar(
            df_posta,
            x="Posta",
            y=["Toplam Vardiya", "Gece Vardiyası"],
            barmode="group",
            color_discrete_sequence=["#2563eb", "#dc2626"]
        )
        fig_posta.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=340, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_posta, width="stretch", key="t9_fig_posta")

    # 7. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI
    st.markdown("### 🏢 7️⃣ Gün ve Vardiya Bazında Posta Dağılımı (Gündüz, Akşam ve Gece Vardiyalarında Hangi Postadan Kaç Kişi Var?)")
    shift_labels = {1: "Gündüz (08-16)", 2: "Akşam (16-24)", 3: "Gece (24-08)"}
    shift_posta_rows = []
    
    for d in range(num_days):
        for k in [1, 2, 3]:
            for p in ['Posta A', 'Posta B', 'Posta C', 'Posta D']:
                p_wids = [w['id'] for w in results['workers'] if w['posta'] == p]
                count_in_shift = sum(1 for wid in p_wids if results['schedule'][wid, d] == k)
                shift_posta_rows.append({
                    "Gün_Vardiya": f"G{d+1} - {shift_labels[k]}",
                    "Gün": f"Gün {d+1:02d}",
                    "Vardiya": shift_labels[k],
                    "Posta": p,
                    "Çalışan Sayısı": int(count_in_shift)
                })

    df_shift_posta = pd.DataFrame(shift_posta_rows)

    fig_sp_all = px.bar(
        df_shift_posta,
        x="Gün_Vardiya",
        y="Çalışan Sayısı",
        color="Posta",
        barmode="stack",
        text="Çalışan Sayısı",
        color_discrete_map={
            "Posta A": "#2563eb",
            "Posta B": "#059669",
            "Posta C": "#d97706",
            "Posta D": "#7c3aed"
        }
    )
    fig_sp_all.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=450, legend=dict(orientation="h", y=1.15), xaxis_tickangle=-45)
    st.plotly_chart(fig_sp_all, width="stretch", key="t9_sp_all")

    st.divider()

    # --- SEKME 7, 8 ve 9 METASEZGİSEL ALGORİTMA KARŞILAŞTIRMA TABLOSU ---
    st.markdown("### 📊 Sezgisel ve Metasezgisel Çözücülerin Bütüncül Karşılaştırma Matrisi (Sekme 7 - 8 - 9)")
    st.markdown("Aşağıdaki tablo, Vardiya Çizelgeleme Problemi (NSP) çözümünde kullanılan **Tepeden Tırmanma (Sekme 7)**, **Tavlama Benzetimi (Sekme 8)** ve **Genetik Algoritma (Sekme 9)** metasezgisel yöntemlerinin teorik ve yapısal özelliklerini karşılaştırmaktadır.")

    # 1. Canlı Çalıştırma Metrikleri (Varsa Göster)
    res7 = st.session_state.get('res_t7')
    res8 = st.session_state.get('res_t8')
    res9 = results

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        score_7_str = f"{res7['final_score']} Ceza Puanı" if res7 else "Henüz Çalıştırılmadı"
        time_7_str = f"{res7['exec_time_ms']} ms" if res7 else "-"
        st.markdown(f"""<div style="background-color:#fff7ed; border:1px solid #f97316; border-radius:8px; padding:12px; text-align:center;">
        <h5 style="color:#c2410c; margin:0;">🏔️ Sekme 7: Tepeden Tırmanma</h5>
        <div style="font-weight:bold; font-size:1.1rem; color:#1e293b; margin-top:5px;">{score_7_str}</div>
        <div style="font-size:0.85rem; color:#64748b;">Çalışma Süresi: {time_7_str}</div>
        </div>""", unsafe_allow_html=True)

    with m_col2:
        score_8_str = f"{res8['final_score']} Ceza Puanı" if res8 else "Henüz Çalıştırılmadı"
        time_8_str = f"{res8['exec_time_ms']} ms" if res8 else "-"
        st.markdown(f"""<div style="background-color:#f0fdf4; border:1px solid #16a34a; border-radius:8px; padding:12px; text-align:center;">
        <h5 style="color:#15803d; margin:0;">🔥 Sekme 8: Tavlama Benzetimi</h5>
        <div style="font-weight:bold; font-size:1.1rem; color:#1e293b; margin-top:5px;">{score_8_str}</div>
        <div style="font-size:0.85rem; color:#64748b;">Çalışma Süresi: {time_8_str}</div>
        </div>""", unsafe_allow_html=True)

    with m_col3:
        score_9_str = f"{res9['final_score']} Ceza Puanı" if res9 else "Henüz Çalıştırılmadı"
        time_9_str = f"{res9['exec_time_ms']} ms" if res9 else "-"
        st.markdown(f"""<div style="background-color:#eff6ff; border:1px solid #2563eb; border-radius:8px; padding:12px; text-align:center;">
        <h5 style="color:#1d4ed8; margin:0;">🧬 Sekme 9: Genetik Algoritma</h5>
        <div style="font-weight:bold; font-size:1.1rem; color:#1e293b; margin-top:5px;">{score_9_str}</div>
        <div style="font-size:0.85rem; color:#64748b;">Çalışma Süresi: {time_9_str}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Yapısal Karşılaştırma Matrisi (HTML Table)
    comp_html = """
    <table style="width:100%; border-collapse:collapse; margin-top:10px; font-size:0.92rem; background-color:#ffffff; border:1px solid #cbd5e1; border-radius:8px; overflow:hidden;">
        <thead>
            <tr style="background-color:#0f172a; color:#ffffff; text-align:left;">
                <th style="padding:12px 15px; width:22%;">Karşılaştırma Kriteri</th>
                <th style="padding:12px 15px; width:26%; color:#fdba74;">🏔️ Sekme 7: Tepeden Tırmanma (Hill Climbing)</th>
                <th style="padding:12px 15px; width:26%; color:#86efac;">🔥 Sekme 8: Tavlama Benzetimi (Simulated Annealing)</th>
                <th style="padding:12px 15px; width:26%; color:#93c5fd;">🧬 Sekme 9: Genetik Algoritma (Genetic Algorithm)</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Arama Paradigması</td>
                <td style="padding:10px 15px;">Tek Noktalı Yöresel Arama (Local Hill Search)</td>
                <td style="padding:10px 15px;">Stokastik Termodinamik Kabul (Metropolis Kriteri)</td>
                <td style="padding:10px 15px;">Popülasyon Bazlı Evrimsel Arama (Biyolojik Evrim)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Çözüm Havuzu / Uzay</td>
                <td style="padding:10px 15px;">Tekil Çözüm Matrisi (1 Çözüm)</td>
                <td style="padding:10px 15px;">Tekil Çözüm + Olasılıksal Sıçrama (1 Çözüm)</td>
                <td style="padding:10px 15px;">Kromozom Popülasyon Havuzu (P adet Çözüm)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Yerel Tuzaklardan (Local Optima) Kaçış</td>
                <td style="padding:10px 15px; color:#c2410c; font-weight:bold;">Zayıf (Tepede Takılı Kalma Riski Yüksek)</td>
                <td style="padding:10px 15px; color:#15803d; font-weight:bold;">İyi (Yüksek Sıcaklıkta Kötü Hamle Kabulü)</td>
                <td style="padding:10px 15px; color:#1d4ed8; font-weight:bold;">Çok Üstün (Mutasyon & Çaprazlama Çeşitliliği)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Ana Operatörler</td>
                <td style="padding:10px 15px;">Vardiya Takası (Swap Move)</td>
                <td style="padding:10px 15px;">Komşu Üretimi + Sıcaklık Düşürme (α)</td>
                <td style="padding:10px 15px;">Turnuva Seçimi, Çaprazlama (pc), Mutasyon (pm), Elitizm (e)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Hesaplama Hızı & Süre</td>
                <td style="padding:10px 15px; color:#15803d; font-weight:bold;">Yıldırım Hızında (~50-1,000 ms)</td>
                <td style="padding:10px 15px; color:#0284c7; font-weight:bold;">Çok Hızlı (~300-7,600 ms)</td>
                <td style="padding:10px 15px; color:#d97706; font-weight:bold;">Orta / Yoğun (~3,000-50,000 ms)</td>
            </tr>
            <tr style="background-color:#f8fafc;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Parametre Hassasiyeti</td>
                <td style="padding:10px 15px;">Düşük (Sadece İterasyon Sayısı)</td>
                <td style="padding:10px 15px;">Yüksek (T0, Tmin, α Soğuma Katsayısı)</td>
                <td style="padding:10px 15px;">Çok Yüksek (P, G, pc, pm, e Hassas Dengesi)</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(comp_html, unsafe_allow_html=True)

    # 3. Metasezgisel Performans & Ceza Puanı Farklılıklarının Nedeni Kartı
    st.markdown("""
    <div style="background-color: #f8fafc; border: 2px solid #3b82f6; border-radius: 10px; padding: 18px; margin-top: 15px; margin-bottom: 20px;">
    <h4 style="color: #1e40af; margin-top: 0;">🧠 Yöneylem Araştırması & Metasezgisel Performans Analizi: Neden Genetik Algoritma Daha Yüksek Ceza Skoru ve Süre Verir?</h4>
    <p style="color: #334155; font-size: 0.95rem; line-height: 1.6;">
    Vardiya Çizelgeleme Problemi (NSP) çözümlerinde <b>Genetik Algoritmanın (1138 Ceza / ~50 sn)</b>, <b>Hill Climbing (993 Ceza / ~1 sn)</b> ve <b>Simulated Annealing (693 Ceza / ~7.6 sn)</b> yöntemlerine göre daha yüksek ceza puanı ve süre vermesi <b>yapay zeka ve metasezgisel optimizasyon literatüründe bilinen ve beklenen bir durumdur:</b>
    </p>
    <ul style="color: #1e293b; font-size: 0.93rem; line-height: 1.6;">
        <li><b>1. Çaprazlama (Crossover) Operatörünün "Kırma/Tahrip Etme" Etkisi (Boundary Disruption):</b> Genetik Algoritma 2 başarılı ebeveyn çizelgeyi rastgele bir kesim gününde (örn: Gün 7) kesip birleştirir. İki iyi çizelge tam ortadan kesilip dikiş atıldığında; Gün 6'dan Gün 7'ye geçişte sirkadiyen ritim (Gece → Gündüz), kayan 7 günlük haftalık izin ve posta bütünlüğü kısıtları kırılır ve ceza puanı yükselir.</li>
        <li><b>2. Lokal Cerrahi Tamir (HC & SA) vs. Global Balyoz Harmanlama (GA):</b> 
        Hill Climbing ve Simulated Annealing, tek bir geçerli matris üzerinde <i>2 işçilik mikro takaslar (micro-swaps)</i> yaparak kısıtları bozmadan cezaları cerrah gibi tek tek tamir eder. GA ise 50 farklı matrisi global olarak harmanlar (balyoz etkisi); kısıtlı arama uzayında mikro ince ayar (fine-tuning) yapmakta zorlanır.</li>
        <li><b>3. Yüksek Hesaplama Yükü (Computational Overhead):</b> 
        Hill Climbing 1 matris üzerinde ~1 saniye çalışırken; Genetik Algoritma 50 popülasyon × 100 jenerasyon = <b>5.000 adet tam matrisin</b> kısıt değerlendirmesini ve turnuva seçimlerini yaptığı için süre ~50 saniyeye ulaşır.</li>
        <li><b>4. Erken Yakınsama (Premature Convergence):</b> 
        Popülasyon 20-30 jenerasyon sonra birbirine benzemeye başlar (genetik çeşitlilik düşer) ve kısıtlı arama uzayında tıkandığı için yerel minimumda takılır.</li>
        <li><b>💡 Literatürdeki Çözüm (Memetik Algoritma / Hybrid GA):</b> Vardiya çizelgeleme gibi sıkı kısıtlı problemlerde literatürde saf Genetik Algoritma yerine GA'nın üzerine Hill Climbing eklenerek <b>Memetik Algoritma (GA + Local Search)</b> tercih edilir.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- FULL SCHEDULE MATRIX TABLE ---
    st.markdown("### 🗓️ Genetik Algoritma Tarafından Üretilen Tam Vardiya Çizelgesi")
    shift_names = {0: "OFF", 1: "Gündüz", 2: "Akşam", 3: "Gece"}
    matrix_data = []
    
    for i, w in enumerate(results['workers']):
        row = {"İşçi": w['name'], "Posta": w['posta'], "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi"}
        for d in range(num_days):
            row[f"Gün {d+1}"] = shift_names[results['schedule'][i, d]]
        matrix_data.append(row)

    df_ga_view = pd.DataFrame(matrix_data)
    st.dataframe(df_ga_view, width="stretch", hide_index=True)
