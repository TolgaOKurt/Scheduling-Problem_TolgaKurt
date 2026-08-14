"""
================================================================================
  VIEWS/TAB10_MEMETIC_ALGORITHM.PY - SEKME 10: MEMETİK ALGORİTMA (HIBRİT OPTİMİZASYON)
================================================================================
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.memetic_algorithm_solver import run_memetic_algorithm

def render_tab10(params):
    """Sekme 10 içeriğini çizer: Memetik Algoritma (MA) Hibrit Evrimsel Çözücü ve Analitik Grafikler."""
    st.markdown("## 🧬 Sekme 10: Memetic Algorithm (Hibrit Genetik + Yerel Tırmanma Çözücüsü)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb;">
<div class="card-title" style="color: #1d4ed8;">🧠 Genetik Arama + Lokal Cerrahi Tamir (Memetik Mantığı)</div>
<ul style="line-height: 1.6;">
<li><b>Global Arama (Genetik Algoritma):</b> Popülasyon tabanlı evrimsel arama ile geniş çözüm uzayındaki küresel minimum potansiyeline sahip bölgeleri keşfeder.</li>
<li><b>Lokal Cerrahi Tamir (Hill Climbing):</b> Çaprazlama operatörünün dikiş günlerinde kırdığı sirkadiyen ritim ve posta bütünlüğü kurallarını <i>nokta atışı mikro takaslar</i> ile tamir eder.</li>
<li><b>Evrimsel Uygunluk (Fitness):</b> Her kromozomun ceza puanı (Z) hesaplanır. Lokal tamir gören bireyler daha yüksek evrimsel uygunluğa kavuşur.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #059669;">
<div class="card-title" style="color: #047857;">🏆 Neden Saf Genetik Algoritmadan Çok Daha Üstündür?</div>
<ul style="line-height: 1.6;">
<li><b>Çaprazlama Tahribatını Önleme:</b> Saf GA'nın kromozomları rastgele kesip birleştirmesinden doğan dikiş hataları Memetik Lokal Arama ile anında giderilir.</li>
<li><b>Erken Yakınsamayı Engelleme:</b> Popülasyon yerel tuzaklara takılmadan sürekli iyileşmeye devam eder.</li>
<li><b>En Düşük Ceza Skoru:</b> Hem Hill Climbing'in yerel tamir gücünü hem de GA'nın küresel arama genişliğini tek potada birleştirir.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- PARAMETRE KONTROL PANELİ ---
    st.markdown("### 🎛️ Memetik Algoritma Hibrit Parametre Kontrol Paneli")
    
    p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
    with p_col1:
        pop_size = st.slider("Popülasyon Büyüklüğü (P)", 20, 200, 50, 10, key="ma_pop")
    with p_col2:
        generations = st.slider("Jenerasyon Sayısı (G)", 20, 300, 100, 10, key="ma_gen")
    with p_col3:
        crossover_rate = st.slider("Çaprazlama Oranı (p_c)", 0.50, 1.00, 0.85, 0.05, format="%.2f", key="ma_pc")
    with p_col4:
        mutation_rate = st.slider("Mutasyon Oranı (p_m)", 0.01, 0.30, 0.05, 0.01, format="%.2f", key="ma_pm")
    with p_col5:
        local_depth = st.slider("Lokal Tamir Derinliği (k)", 1, 20, 5, 1, key="ma_depth", help="Çaprazlama sonrası çocuğa uygulanacak Hill Climbing mikro-takas adım sayısı.")
    with p_col6:
        elitism_count = st.slider("Elitizm Sayısı (e)", 1, 10, 2, 1, key="ma_elitism")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # --- HESAPLAMA VE SÜRE TAHMİNİ PANELİ ---
    total_evaluations = pop_size + generations * (pop_size - elitism_count) * (1 + local_depth * 0.2)
    # Ampirik Benchmark Ölçümü: 1 Memetik evrim + lokal takas adımı hücre başına ~0.00373 ms
    est_ma_cpu_ms = round(total_evaluations * num_workers * num_days * 0.00373, 1)

    st.markdown("#### 🧮 Evrimsel & Hibrit Değerlendirme Tahmin Paneli")

    pred_c1, pred_c2, pred_c3 = st.columns(3)
    with pred_c1:
        st.metric(
            label="Toplam Değerlendirilecek Birey & Takas Adımı",
            value=f"{int(total_evaluations):,} İterasyon",
            help="Popülasyon evrimi ve lokal tamir adımları dahil toplam değerlendirme."
        )
    with pred_c2:
        st.metric(
            label="Tahmini Hibrit Evrim Süresi",
            value=f"~{est_ma_cpu_ms:,} ms",
            help="Memetik iyileştirme tamamlanana kadar harcanacak tahmini CPU süresi."
        )
    with pred_c3:
        st.metric(
            label="Popülasyon Koruma Oranı (Elitizm)",
            value=f"%{round((elitism_count / pop_size) * 100, 1)}",
            help="Her nesilde korunup doğrudan aktarılan şampiyon kromozom oranı."
        )

    st.divider()

    run_btn = st.button("🚀 Memetik Algoritma Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t10")

    if not run_btn and "res_t10" not in st.session_state:
        st.warning("👈 Hibrit aramayı başlatmak için yukarıdaki **'🚀 Memetik Algoritma Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        with st.spinner("⏳ Memetik Algoritma (GA + Hill Climbing) Evrimleştiriliyor ve Yerel Olarak Tamir Ediliyor... Lütfen Bekleyiniz..."):
            results = run_memetic_algorithm(
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
                local_search_depth=local_depth,
                elitism_count=elitism_count,
                seed=42,
                custom_workers=custom_workers
            )
            st.session_state["res_t10"] = results
    else:
        results = st.session_state["res_t10"]

    # --- BİTİRME NEDENİ BİLDİRİMİ ---
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
        <div class="metric-label">Memetik En İyi Skor</div>
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
        <div class="metric-label">Hesaplama Süresi</div>
        <div class="metric-value" style="color: #059669;">{results['exec_time_ms']} ms</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Memetik Algoritma Evrimsel Yakınsama & Analitik Grafikler")

    gens = list(range(1, len(results['best_score_history']) + 1))
    
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ Jenerasyon Bazlı Memetik Yakınsama Grafiği (Best vs Avg Score)")
        fig_ma_conv = go.Figure()
        fig_ma_conv.add_trace(go.Scatter(
            x=gens, y=results['best_score_history'],
            name="Memetik En İyi Birey Skoru Z_best",
            line=dict(color="#059669", width=2.5)
        ))
        fig_ma_conv.add_trace(go.Scatter(
            x=gens, y=results['avg_score_history'],
            name="Popülasyon Ortalama Skoru Z_avg",
            line=dict(color="#2563eb", width=2, dash="dash")
        ))
        fig_ma_conv.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380,
            xaxis=dict(title="Jenerasyon (Nesil)"),
            yaxis=dict(title="Toplam Ceza Skoru Z"),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig_ma_conv, width="stretch", key="t10_fig_conv")

    with g_col2:
        st.markdown("##### 2️⃣ Popülasyon Genetik Çeşitlilik İndeksi (Standart Sapma)")
        fig_div = px.line(
            x=gens, y=results['diversity_history'],
            labels={"x": "Jenerasyon", "y": "Popülasyon Çeşitliliği (Std Dev)"},
            line_shape="linear"
        )
        fig_div.update_traces(line_color="#2563eb", line_width=2.5)
        fig_div.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_div, width="stretch", key="t10_fig_div")

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        st.markdown("##### 3️⃣ Memetik Optimal Vardiya Matrisi Isı Haritası (Heatmap)")
        fig_ma_map = px.imshow(
            results['schedule'],
            labels=dict(x="Günler", y="Çalışanlar", color="Vardiya (0:OFF, 1:G, 2:A, 3:N)"),
            x=[f"G{d+1}" for d in range(num_days)],
            y=[w['name'] for w in results['workers']],
            color_continuous_scale=[[0, '#cbd5e1'], [0.33, '#fde047'], [0.66, '#f97316'], [1.0, '#1e3a8a']]
        )
        fig_ma_map.update_layout(height=400)
        st.plotly_chart(fig_ma_map, width="stretch", key="t10_fig_map")

    with g_col4:
        st.markdown("##### 4️⃣ Memetik Çalışan İş Yükü ve Gece Nöbeti Dağılımı")
        work_days = [np.sum(results['schedule'][i, :] > 0) for i in range(num_workers)]
        night_days = [np.sum(results['schedule'][i, :] == 3) for i in range(num_workers)]
        
        df_ma_workload = pd.DataFrame({
            "İşçi": [w['name'] for w in results['workers']],
            "Toplam Çalışma": work_days,
            "Gece Nöbeti": night_days
        })
        
        fig_ma_wl = px.bar(
            df_ma_workload,
            x="İşçi",
            y=["Toplam Çalışma", "Gece Nöbeti"],
            barmode="group",
            color_discrete_sequence=["#059669", "#dc2626"]
        )
        fig_ma_wl.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=400, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_ma_wl, width="stretch", key="t10_fig_wl")

    g_col5, g_col6 = st.columns(2)

    with g_col5:
        st.markdown("##### 5️⃣ Memetik Minimize Edilmiş Yumuşak Kısıt Cezaları (&min; Z)")
        df_ma_penalties = pd.DataFrame({
            "Kısıt Tipi": list(results['penalties'].keys()),
            "Ceza Puanı": list(results['penalties'].values())
        })
        fig_ma_pen = px.bar(
            df_ma_penalties,
            x="Ceza Puanı",
            y="Kısıt Tipi",
            orientation="h",
            text="Ceza Puanı",
            color="Ceza Puanı",
            color_continuous_scale="Reds"
        )
        fig_ma_pen.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=340, showlegend=False)
        st.plotly_chart(fig_ma_pen, width="stretch", key="t10_fig_pen")

    with g_col6:
        st.markdown("##### 6️⃣ Memetik Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")
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
        st.plotly_chart(fig_posta, width="stretch", key="t10_fig_posta")

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
    st.plotly_chart(fig_sp_all, width="stretch", key="t10_sp_all")

    st.divider()

    # --- METASEZGİSEL ÇÖZÜCÜLERİN 4-YÖNLÜ BÜTÜNCÜL KARŞILAŞTIRMA MATRİSİ ---
    st.markdown("### 📊 Tüm Metasezgisel Çözücülerin Bütüncül Karşılaştırma Matrisi (Sekme 7 - 8 - 9 - 10)")
    st.markdown("Aşağıdaki tablo, Vardiya Çizelgeleme Problemi (NSP) çözümünde kullanılan **Tepeden Tırmanma (Sekme 7)**, **Tavlama Benzetimi (Sekme 8)**, **Genetik Algoritma (Sekme 9)** ve **Memetik Algoritma (Sekme 10)** metasezgisel yöntemlerinin metriklerini karşılaştırmaktadır.")

    res7 = st.session_state.get('res_t7')
    res8 = st.session_state.get('res_t8')
    res9 = st.session_state.get('res_t9')
    res10 = results

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        score_7_str = f"{res7['final_score']} Ceza Puanı" if res7 else "Henüz Çalıştırılmadı"
        time_7_str = f"{res7['exec_time_ms']} ms" if res7 else "-"
        st.markdown(f"""<div style="background-color:#fff7ed; border:1px solid #f97316; border-radius:8px; padding:12px; text-align:center;">
        <h5 style="color:#c2410c; margin:0;">🏔️ Sekme 7: Hill Climbing</h5>
        <div style="font-weight:bold; font-size:1.1rem; color:#1e293b; margin-top:5px;">{score_7_str}</div>
        <div style="font-size:0.85rem; color:#64748b;">Çalışma Süresi: {time_7_str}</div>
        </div>""", unsafe_allow_html=True)

    with m_col2:
        score_8_str = f"{res8['final_score']} Ceza Puanı" if res8 else "Henüz Çalıştırılmadı"
        time_8_str = f"{res8['exec_time_ms']} ms" if res8 else "-"
        st.markdown(f"""<div style="background-color:#f0fdf4; border:1px solid #16a34a; border-radius:8px; padding:12px; text-align:center;">
        <h5 style="color:#15803d; margin:0;">🔥 Sekme 8: Simulated Annealing</h5>
        <div style="font-weight:bold; font-size:1.1rem; color:#1e293b; margin-top:5px;">{score_8_str}</div>
        <div style="font-size:0.85rem; color:#64748b;">Çalışma Süresi: {time_8_str}</div>
        </div>""", unsafe_allow_html=True)

    with m_col3:
        score_9_str = f"{res9['final_score']} Ceza Puanı" if res9 else "Henüz Çalıştırılmadı"
        time_9_str = f"{res9['exec_time_ms']} ms" if res9 else "-"
        st.markdown(f"""<div style="background-color:#fef2f2; border:1px solid #ef4444; border-radius:8px; padding:12px; text-align:center;">
        <h5 style="color:#b91c1c; margin:0;">🧬 Sekme 9: Genetik Algoritma</h5>
        <div style="font-weight:bold; font-size:1.1rem; color:#1e293b; margin-top:5px;">{score_9_str}</div>
        <div style="font-size:0.85rem; color:#64748b;">Çalışma Süresi: {time_9_str}</div>
        </div>""", unsafe_allow_html=True)

    with m_col4:
        score_10_str = f"{res10['final_score']} Ceza Puanı" if res10 else "Henüz Çalıştırılmadı"
        time_10_str = f"{res10['exec_time_ms']} ms" if res10 else "-"
        st.markdown(f"""<div style="background-color:#eff6ff; border:2px solid #2563eb; border-radius:8px; padding:12px; text-align:center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <h5 style="color:#1d4ed8; margin:0;">🏆 Sekme 10: Memetik Algoritma</h5>
        <div style="font-weight:bold; font-size:1.15rem; color:#059669; margin-top:5px;">{score_10_str}</div>
        <div style="font-size:0.85rem; color:#64748b;">Çalışma Süresi: {time_10_str}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # HTML Table
    comp_html = """
    <table style="width:100%; border-collapse:collapse; margin-top:10px; font-size:0.92rem; background-color:#ffffff; border:1px solid #cbd5e1; border-radius:8px; overflow:hidden;">
        <thead>
            <tr style="background-color:#0f172a; color:#ffffff; text-align:left;">
                <th style="padding:12px 15px; width:20%;">Karşılaştırma Kriteri</th>
                <th style="padding:12px 15px; width:20%; color:#fdba74;">🏔️ Sekme 7: Hill Climbing</th>
                <th style="padding:12px 15px; width:20%; color:#86efac;">🔥 Sekme 8: Simulated Annealing</th>
                <th style="padding:12px 15px; width:20%; color:#fca5a5;">🧬 Sekme 9: Genetic Algorithm</th>
                <th style="padding:12px 15px; width:20%; color:#93c5fd;">🏆 Sekme 10: Memetic Algorithm</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Arama Paradigması</td>
                <td style="padding:10px 15px;">Tek Noktalı Yöresel Arama</td>
                <td style="padding:10px 15px;">Stokastik Termodinamik Kabul</td>
                <td style="padding:10px 15px;">Popülasyon Bazlı Evrimsel Arama</td>
                <td style="padding:10px 15px; font-weight:bold; color:#1d4ed8;">Hibrit: Global Evrim + Lokal HC Tamir</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Çaprazlama Hasarı Tamiri</td>
                <td style="padding:10px 15px;">Yok</td>
                <td style="padding:10px 15px;">Yok</td>
                <td style="padding:10px 15px; color:#dc2626;">❌ Yok (Dikiş hataları kalır)</td>
                <td style="padding:10px 15px; color:#059669; font-weight:bold;">✅ Var (Çaprazlama dikiş noktaları anında tamir edilir)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Yerel Tuzaktan Kaçış</td>
                <td style="padding:10px 15px; color:#c2410c;">Zayıf</td>
                <td style="padding:10px 15px; color:#15803d;">İyi (Sıcaklık Sıçraması)</td>
                <td style="padding:10px 15px; color:#1d4ed8;">Çok Üstün</td>
                <td style="padding:10px 15px; color:#059669; font-weight:bold;">🏆 Mükemmel (Popülasyon + Lokal Tamir)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:10px 15px; font-weight:bold; color:#334155;">Hesaplama Hızı & Süre</td>
                <td style="padding:10px 15px; color:#15803d; font-weight:bold;">Yıldırım Hızında (~50-1,000 ms)</td>
                <td style="padding:10px 15px; color:#0284c7; font-weight:bold;">Çok Hızlı (~300-7,600 ms)</td>
                <td style="padding:10px 15px; color:#d97706;">Orta (~3,000-50,000 ms)</td>
                <td style="padding:10px 15px; color:#2563eb; font-weight:bold;">Dengeli Hibrit (~5,000-40,000 ms)</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(comp_html, unsafe_allow_html=True)

    st.divider()

    # --- FULL SCHEDULE MATRIX TABLE ---
    st.markdown("### 🗓️ Memetik Algoritma Tarafından Üretilen Tam Vardiya Çizelgesi")
    shift_names = {0: "OFF", 1: "Gündüz", 2: "Akşam", 3: "Gece"}
    matrix_data = []
    
    for i, w in enumerate(results['workers']):
        row = {"İşçi": w['name'], "Posta": w['posta'], "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi"}
        for d in range(num_days):
            row[f"Gün {d+1}"] = shift_names[results['schedule'][i, d]]
        matrix_data.append(row)

    df_ma_view = pd.DataFrame(matrix_data)
    st.dataframe(df_ma_view, width="stretch", hide_index=True)
