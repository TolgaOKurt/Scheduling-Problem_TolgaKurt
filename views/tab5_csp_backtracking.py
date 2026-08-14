"""
================================================================================
  VIEWS/TAB5_CSP_BACKTRACKING.PY - SEKME 5: BACKTRACKING / CSP SOLVER & SİMÜLASYONU
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.csp_backtracking_solver import CSPBacktrackingSolver

def render_tab5(params):
    """Sekme 5 içeriğini çizer: Backtracking / CSP Teorisi, Simülasyonu, Grafikler ve Karşılaştırma Matrisi."""
    st.markdown("## 🔍 Backtracking / Constraint Satisfaction Problem (CSP) Yaklaşımı")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #059669;">
<div class="card-title" style="color: #047857;">🧩 CSP Çerçevesi (Değişkenler & Kümeler)</div>
<ul>
<li><b>Değişkenler (X<sub>i,t</sub>):</b> <i>i</i> çalışanının <i>t</i> günündeki vardiya ataması.</li>
<li><b>Etki Alanı (Domain D<sub>i,t</sub>):</b> {0: OFF, 1: Gündüz, 2: Akşam, 3: Gece}</li>
<li><b>Geri İzleme (Backtracking):</b> Bir vardiyada kadro eksik kaldığında algoritma durmaz; önceki günlere dönerek (backtrack) alternatif işçi kombinasyonlarını dener.</li>
<li><b>Pruning (Budama):</b> Sert kısıtları ihlal eden mantıksız arama dalları önceden budanarak arama ağacı küçültülür.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb;">
<div class="card-title" style="color: #1d4ed8;">⚡ Sezgisel Yönlendirme (Heuristics)</div>
<ul>
<li><b>MRV (Minimum Remaining Values):</b> En çok kısıtlanmış ve atama seçeneği en az kalan gün/işçiden aramaya başlar.</li>
<li><b>LCV (Least Constraining Value):</b> Gelecekteki günlerin seçeneklerini en az kısıtlayan vardiya atamasına öncelik verir.</li>
<li><b>Kadro Boşalması ve İzin Birikmesini Önleme:</b> Backtracking mekanizması sayesinde haftanın son günlerinde kadro yetersizliği yaşanmaz; haftalık izinler tüm haftaya %100 dengeli dağıtılır.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🎛️ Backtracking / CSP Arama Limiti Kontrolü")
    max_backtracks = st.number_input("Maksimum Backtrack Limiti", min_value=500, max_value=10000, value=3000, step=500, key="csp_mb")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    st.divider()

    run_btn = st.button("🚀 CSP Backtracking Çözücüsünü Çalıştır", type="primary", width="stretch", key="btn_run_t5")

    if not run_btn and "res_t5" not in st.session_state:
        st.warning("👈 Arama ağacını çalıştırmak için yukarıdaki **'🚀 CSP Backtracking Çözücüsünü Çalıştır'** butonuna basınız.")
        return

    if run_btn:
        with st.spinner("⏳ CSP Backtracking Arama Ağacı Taranıyor... Lütfen Bekleyiniz..."):
            solver = CSPBacktrackingSolver(
                n_workers=num_workers,
                n_days=num_days,
                r_day=req_day,
                r_eve=req_eve,
                r_night=req_night,
                max_backtracks=max_backtracks,
                custom_workers=custom_workers,
                weights=weights
            )
            results = solver.solve()
            st.session_state["res_t5"] = results
    else:
        results = st.session_state["res_t5"]

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        status_str = "✅ BAŞARILI (Geçerli)" if results['success'] else "❌ ÇÖZÜM YOK (Limit)"
        status_clr = "#059669" if results['success'] else "#dc2626"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">CSP Durumu</div>
        <div class="metric-value" style="color: {status_clr}; font-size: 1.2rem;">{status_str}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Geri İzleme (Backtrack)</div>
        <div class="metric-value" style="color: #d97706;">{results['backtracks']}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Gezilen Düğüm Sayısı</div>
        <div class="metric-value" style="color: #2563eb;">{results['nodes_explored']}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Çözüm Süresi</div>
        <div class="metric-value" style="color: #059669;">{results['exec_time_ms']} ms</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam Ceza Puanı</div>
        <div class="metric-value" style="color: #dc2626;">{results['total_penalty']}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- SERT VE YUMUŞAK KISIT DEĞERLENDİRME VE ZAYIFLIK ANALİZ UYARISI ---
    st.warning("""
    💡 **Yöneylem Araştırması Analizi (CSP / Backtracking Yaklaşımının Zayıf Yönleri ve Başarımı):**  
    Saf **Backtracking (CSP)** algoritması temel olarak **Sert Kısıtları %100 doğrulayan ilk geçerli çözümü (Satisfiability)** bulmayı hedefler. 
    Bu durum şu sonuçları doğurur:
    - **Güçlü Yönü:** Sert kısıtları (Pazar izin çöküşü, MYK ehliyet eksiklikleri, 11 saat dinlenme süreleri) %100 Kusursuz Garanti Eder.
    - **Kötü Yaptığı / İhmal Ettiği Yön:** İlk bulduğu geçerli çözüme kilitlendiği için Yumuşak Kısıtları (Örn: Gece nöbetinin postalar arasında tam adil dağılımı veya sirkadiyen vardiya dönüşleri) bir **MILP (Tam Sayılı Optimizasyon)** kadar mükemmel minimize etmeyebilir. Aşağıdaki grafiklerden kalan yumuşak ceza puanlarını gözlemleyebilirsiniz.
    """)

    st.divider()

    # --- PERSONEL YETKİNLİK VE MYK SERTİFİKA KADRO TABLOSU ---
    st.markdown("### 🪪 Personel Yetkinlik ve MYK Sertifika Kadro Listesi")
    profile_rows = []
    for w in results['workers']:
        skills_formatted = ", ".join(sorted(list(w['skills'])))
        profile_rows.append({
            "İşçi Adı": w['name'],
            "Posta": w['posta'],
            "Unvan": "Kıdemli Usta" if w['is_usta'] else "Operatör/İşçi",
            "Sahip Olduğu MYK Sertifika & Ehliyetler": skills_formatted,
            "Talep Ettiği İzin Günü": f"Gün {w['pref_off'] + 1}"
        })

    df_profiles = pd.DataFrame(profile_rows)
    st.dataframe(df_profiles, width="stretch", hide_index=True)



    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Backtracking CSP Çözüm Grafikleri & Ceza Analizi")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ CSP Vardiya Dağılım Isı Haritası (Heatmap)")
        fig_csp_map = px.imshow(
            results['schedule'],
            labels=dict(x="Günler", y="Çalışanlar", color="Vardiya (0:OFF, 1:G, 2:A, 3:N)"),
            x=[f"G{d+1}" for d in range(num_days)],
            y=[w['name'] for w in results['workers']],
            color_continuous_scale=[[0, '#cbd5e1'], [0.33, '#fde047'], [0.66, '#f97316'], [1.0, '#1e3a8a']],
            aspect="auto"
        )
        fig_csp_map.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_csp_map, width="stretch", key="t5_csp_map")

    with g_col2:
        st.markdown("##### 2️⃣ CSP Personel Vardiya & Gece Nöbet Dağılımı")
        csp_worked = [np.sum(results['schedule'][i, :] > 0) for i in range(num_workers)]
        csp_night = [np.sum(results['schedule'][i, :] == 3) for i in range(num_workers)]
        
        df_csp_workload = pd.DataFrame({
            "İşçi": [w['name'] for w in results['workers']],
            "Toplam Çalışma": csp_worked,
            "Gece Nöbeti": csp_night
        })
        
        fig_csp_wl = px.bar(
            df_csp_workload,
            x="İşçi",
            y=["Toplam Çalışma", "Gece Nöbeti"],
            barmode="group",
            color_discrete_sequence=["#059669", "#dc2626"]
        )
        fig_csp_wl.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_csp_wl, width="stretch", key="t5_csp_wl")

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        st.markdown("##### 3️⃣ CSP Yumuşak Kısıt Ceza Puanı Dağılımı")
        df_csp_penalties = pd.DataFrame({
            "Kısıt Tipi": list(results['penalties'].keys()),
            "Ceza Puanı": list(results['penalties'].values())
        })
        fig_csp_pen = px.bar(
            df_csp_penalties,
            x="Ceza Puanı",
            y="Kısıt Tipi",
            orientation="h",
            text="Ceza Puanı",
            color="Ceza Puanı",
            color_continuous_scale="Reds"
        )
        fig_csp_pen.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=340, showlegend=False)
        st.plotly_chart(fig_csp_pen, width="stretch", key="t5_csp_pen")

    with g_col4:
        st.markdown("##### 4️⃣ CSP Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")
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
        st.plotly_chart(fig_posta, width="stretch", key="t5_posta_load")

    # 5. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI
    st.markdown("### 🏢 5️⃣ Gün ve Vardiya Bazında Posta Dağılımı (Gündüz, Akşam ve Gece Vardiyalarında Hangi Postadan Kaç Kişi Var?)")
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

    v_tab1, v_tab2 = st.tabs(["📊 Tüm Günler & Vardiyalar Bütüncül Grafik", "🔍 Vardiya Türüne Göre Ayrıştırılmış"])

    with v_tab1:
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
        st.plotly_chart(fig_sp_all, width="stretch", key="t5_sp_all")

    with v_tab2:
        selected_shift = st.selectbox("İncelenecek Vardiyayı Seçin:", ["Gündüz (08-16)", "Akşam (16-24)", "Gece (24-08)"], key="t5_vselect")
        df_sub = df_shift_posta[df_shift_posta["Vardiya"] == selected_shift]
        
        fig_sp_sub = px.bar(
            df_sub,
            x="Gün",
            y="Çalışan Sayısı",
            color="Posta",
            barmode="group",
            text="Çalışan Sayısı",
            color_discrete_map={
                "Posta A": "#2563eb",
                "Posta B": "#059669",
                "Posta C": "#d97706",
                "Posta D": "#7c3aed"
            }
        )
        fig_sp_sub.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=400, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_sp_sub, width="stretch", key="t5_sp_sub")

    st.divider()

    # --- FULL SCHEDULE MATRIX TABLE ---
    st.markdown("### 🗓️ Backtracking / CSP Tarafından Üretilen Tam Vardiya Çizelgesi")
    shift_names = {0: "OFF", 1: "Gündüz", 2: "Akşam", 3: "Gece"}
    matrix_data = []
    
    for i, w in enumerate(results['workers']):
        row = {"İşçi": w['name'], "Posta": w['posta'], "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi"}
        for d in range(num_days):
            row[f"Gün {d+1}"] = shift_names[results['schedule'][i, d]]
        matrix_data.append(row)

    df_csp_view = pd.DataFrame(matrix_data)
    st.dataframe(df_csp_view, width="stretch", hide_index=True)
