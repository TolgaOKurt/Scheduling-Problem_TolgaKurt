"""
================================================================================
  VIEWS/TAB7_HILL_CLIMBING.PY - SEKME 7: HILL CLIMBING / LOCAL SEARCH (TIRMANMA)
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.hill_climbing_solver import run_hill_climbing

def render_tab7(params):
    """Sekme 7 içeriğini çizer: Hill Climbing / Local Search Metasezgisel Çözücüsü ve Grafikler."""
    st.markdown("## 🏔️ Hill Climbing / Local Search (Tepeden Tırmanma & Yöresel Arama)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #d97706;">
<div class="card-title" style="color: #b45309;">🔄 Komşuluk Operatörleri (Neighborhood Search)</div>
<ul>
<li><b>Vardiya Takası (Swap Move):</b> Aynı gün içerisinde çalışan iki işçinin vardiyaları takas edilir.</li>
<li><b>Kabul Kriteri (Iterative Descent):</b> Yapılan takas Sert Kısıtları bozmuyorsa VE toplam ceza puanını <b>düşürüyorsa</b> hamle kabul edilir (&Delta;Z < 0).</li>
<li><b>Yöresel Arama:</b> Arama uzayında adım adım ilerleyerek daha adil nöbet ve posta bütünlüğü arar.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #7c3aed;">
<div class="card-title" style="color: #6d28d9;">📈 Yakınsama & Yerel Optimum (Local Optimum)</div>
<ul>
<li><b>Aşamalı İyileşme:</b> İlk Greedy çizelgenin eksiklerini (sirkadiyen ihlaller, posta bölünmeleri) binlerce iterasyonda düzeltir.</li>
<li><b>Yakınsama Eğrisi:</b> Algoritmanın iterasyonlar boyunca ceza puanını nasıl adım adım düşürdüğü canlı grafikte izlenir.</li>
<li><b>Limit:</b> Çevresinde daha iyi bir komşu kalmadığında veya maksimum iterasyon bittiğinde durur.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- METASEZGİSEL (HILL CLIMBING) ÜSTÜNLÜKLERİ PANOLARI ---
    st.markdown("### 🚀 Metasezgisel (Hill Climbing) Yönteminin ILP / MILP'e Göre 4 Temel Üstünlüğü")

    adv_col1, adv_col2 = st.columns(2)

    with adv_col1:
        st.markdown("""<div style="background-color: #f0fdf4; border: 2px solid #059669; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #047857; margin-top: 0;">1. 🌐 Devasa Tesislerde Ölçeklenebilirlik (Scalability)</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
MILP problemleri NP-Hard'dır. 500+ işçi ve 365 günlük devasa tesislerde MILP denklem sayısı milyonlara ulaşır ve RAM/süre kilitlenmesi yaşanır. <b>Hill Climbing ise hafızayı şişirmeden saniyeler içinde çözüme ulaşır.</b>
</p>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background-color: #eff6ff; border: 2px solid #2563eb; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #1e40af; margin-top: 0;">2. 🧩 Doğrusal Olmayan (Non-Linear) Kısıt Esnekliği</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
ILP'de tüm kurallar doğrusal (linear) olmak zorundadır. Hill Climbing'de ise <b>if-else mantığı, yapay zeka sinir ağları, olasılıksal/stokastik kurallar veya karmaşık Python fonksiyonları serbestçe kullanılabilir.</b>
</p>
</div>""", unsafe_allow_html=True)

    with adv_col2:
        st.markdown("""<div style="background-color: #fef3c7; border: 2px solid #d97706; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #b45309; margin-top: 0;">3. ⏱️ Anlık Vardiya Tamiri (Real-Time Re-Scheduling)</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
Aniden 2 işçi hastalanıp gelmediğinde MILP tüm modeli sıfırdan kurup çözer. <b>Hill Climbing ise var olan planı başlangıç alarak 50 ms içinde lokal takaslarla anında tamir eder (Repair Heuristic).</b>
</p>
</div>""", unsafe_allow_html=True)

        st.markdown("""<div style="background-color: #faf5ff; border: 2px solid #7c3aed; border-radius: 10px; padding: 16px; margin-bottom: 15px;">
<h4 style="color: #6d28d9; margin-top: 0;">4. ⚙️ Düşük Donanım & Çözücü Bağımsızlığı</h4>
<p style="color: #1e293b; font-size: 0.95rem;">
Pahalı veya ticari MILP çözücülere (Gurobi, CPLEX) ihtiyaç duymaz. <b>Tamamen açık kaynaklı ve her türlü cihazda hafif çalışan bir kural motoruna sahiptir.</b>
</p>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🎛️ Hill Climbing İterasyon Kontrolü")
    max_iter = st.slider("Maksimum İterasyon Sayısı (K)", 2000, 10000, 3000, 1000, key="hc_iter")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    st.divider()

    run_btn = st.button("🚀 Hill Climbing Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t7")

    if not run_btn and "res_t7" not in st.session_state:
        st.warning("👈 Optimizasyonu başlatmak için yukarıdaki **'🚀 Hill Climbing Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        with st.spinner("⏳ Hill Climbing Yerel Araması Çalıştırılıyor... Lütfen Bekleyiniz..."):
            results = run_hill_climbing(
                n_workers=num_workers,
                n_days=num_days,
                r_day=req_day,
                r_eve=req_eve,
                r_night=req_night,
                weights=weights,
                max_iterations=max_iter,
                seed=42,
                custom_workers=custom_workers
            )
            st.session_state["res_t7"] = results
    else:
        results = st.session_state["res_t7"]

    # --- ÇALIŞMAYI BİTİRME NEDENİ BİLDİRİMİ ---
    st.info(f"📌 **Çözücünün Çalışmayı Bitirme Nedeni:** {results['termination_reason']}")

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Başlangıç Ceza Skoru</div>
        <div class="metric-value" style="color: #dc2626;">{results['initial_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İyileştirilmiş Son Skor</div>
        <div class="metric-value" style="color: #059669;">{results['final_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İyileşme Oranı</div>
        <div class="metric-value" style="color: #1d4ed8;">%{results['improvement_rate']}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Kabul Edilen Hamle</div>
        <div class="metric-value" style="color: #7c3aed;">{results['accepted_moves']} / {results['total_iterations']}</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Arama Süresi</div>
        <div class="metric-value" style="color: #059669;">{results['exec_time_ms']} ms</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- 3 GELİŞMİŞ YÖNTEMİN KARŞILAŞTIRMA MATRİSİ ---
    st.markdown("### 📊 Gelişmiş Yaklaşımların Bütüncül Karşılaştırma Matrisi (CSP vs. ILP vs. Hill Climbing)")

    df_3way_hc = pd.DataFrame({
        "Karşılaştırma Özelliği": [
            "Algoritma Sınıfı",
            "Sert Kısıt Garantisi (%100 Feasibility)",
            "Ceza İyileştirme Mekanizması",
            "Arama Yöntemi",
            "En Uygun (Optimal) Çözüm Garantisi",
            "Hesaplama Esnekliği"
        ],
        "Sekme 5: CSP Backtracking": ["Kısıt Tatmin (AI)", "✅ %100 Kusursuz Garanti", "❌ Yok (İlk çizelgede kalır)", "Arama Ağacı (DFS)", "❌ Yok (Yalnız Feasible)", "⏱️ Milisaniyeler"],
        "Sekme 6: ILP / MILP": ["Matematiksel Programlama", "✅ %100 Kusursuz Garanti", "🏆 Tam Optimizasyon (min Z*)", "Dal-Sınır (Branch & Bound)", "🏆 MATEMATİKSEL GARANTİ", "⏱️ Saniyeler"],
        "Sekme 7: Hill Climbing": ["Metasezgisel (Local Search)", "✅ %100 Kusursuz Garanti", "📈 İteratif Hamle İyileştirmesi", "Komşuluk Araması (Swap Move)", "⚠️ Yerel Optimum (Local Min)", "⚡ Çok Hızlı (İteratif)"]
    })

    st.dataframe(df_3way_hc, width="stretch", hide_index=True)

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Hill Climbing Local Search Grafikleri & Yakınsama Analizi")

    # 1. İTERASYON BAZLI CEZA PUANI DÜŞÜŞ YAKINSAMA GRAFİĞİ
    st.markdown("##### 1️⃣ İterasyon Bazlı Ceza Puanı Düşüş & Yakınsama Eğrisi (Convergence Curve)")
    df_history = pd.DataFrame({
        "İterasyon": list(range(len(results['history']))),
        "Toplam Ceza Puanı (Z)": results['history']
    })

    fig_conv = px.line(
        df_history,
        x="İterasyon",
        y="Toplam Ceza Puanı (Z)",
        labels={"İterasyon": "Arama İterasyonu (K)", "Toplam Ceza Puanı (Z)": "Ceza Puanı (Skor)"},
        color_discrete_sequence=["#d97706"]
    )
    fig_conv.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
    st.plotly_chart(fig_conv, width="stretch", key="t7_fig_conv")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 2️⃣ Hill Climbing Vardiya Dağılım Isı Haritası (Heatmap)")
        fig_hc_map = px.imshow(
            results['schedule'],
            labels=dict(x="Günler", y="Çalışanlar", color="Vardiya (0:OFF, 1:G, 2:A, 3:N)"),
            x=[f"G{d+1}" for d in range(num_days)],
            y=[w['name'] for w in results['workers']],
            color_continuous_scale=[[0, '#cbd5e1'], [0.33, '#fde047'], [0.66, '#f97316'], [1.0, '#1e3a8a']],
            aspect="auto"
        )
        fig_hc_map.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_hc_map, width="stretch", key="t7_fig_hc_map")

    with g_col2:
        st.markdown("##### 3️⃣ HC Personel Vardiya & Gece Nöbet Dağılımı")
        hc_worked = [np.sum(results['schedule'][i, :] > 0) for i in range(num_workers)]
        hc_night = [np.sum(results['schedule'][i, :] == 3) for i in range(num_workers)]
        
        df_hc_workload = pd.DataFrame({
            "İşçi": [w['name'] for w in results['workers']],
            "Toplam Çalışma": hc_worked,
            "Gece Nöbeti": hc_night
        })
        
        fig_hc_wl = px.bar(
            df_hc_workload,
            x="İşçi",
            y=["Toplam Çalışma", "Gece Nöbeti"],
            barmode="group",
            color_discrete_sequence=["#059669", "#dc2626"]
        )
        fig_hc_wl.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_hc_wl, width="stretch", key="t7_fig_hc_wl")

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        st.markdown("##### 4️⃣ HC Yumuşak Kısıt Ceza Puanı Dağılımı")
        df_hc_penalties = pd.DataFrame({
            "Kısıt Tipi": list(results['penalties'].keys()),
            "Ceza Puanı": list(results['penalties'].values())
        })
        fig_hc_pen = px.bar(
            df_hc_penalties,
            x="Ceza Puanı",
            y="Kısıt Tipi",
            orientation="h",
            text="Ceza Puanı",
            color="Ceza Puanı",
            color_continuous_scale="Reds"
        )
        fig_hc_pen.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=340, showlegend=False)
        st.plotly_chart(fig_hc_pen, width="stretch", key="t7_fig_hc_pen")

    with g_col4:
        st.markdown("##### 5️⃣ HC Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")
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
        st.plotly_chart(fig_posta, width="stretch", key="t7_posta_load")

    # 6. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI
    st.markdown("### 🏢 6️⃣ Gün ve Vardiya Bazında Posta Dağılımı (Gündüz, Akşam ve Gece Vardiyalarında Hangi Postadan Kaç Kişi Var?)")
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
    st.plotly_chart(fig_sp_all, width="stretch", key="t7_sp_all")

    st.divider()

    # --- FULL SCHEDULE MATRIX TABLE ---
    st.markdown("### 🗓️ Hill Climbing Tarafından İyileştirilen Vardiya Çizelgesi")
    shift_names = {0: "OFF", 1: "Gündüz", 2: "Akşam", 3: "Gece"}
    matrix_data = []
    
    for i, w in enumerate(results['workers']):
        row = {"İşçi": w['name'], "Posta": w['posta'], "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi"}
        for d in range(num_days):
            row[f"Gün {d+1}"] = shift_names[results['schedule'][i, d]]
        matrix_data.append(row)

    df_hc_view = pd.DataFrame(matrix_data)
    st.dataframe(df_hc_view, width="stretch", hide_index=True)
