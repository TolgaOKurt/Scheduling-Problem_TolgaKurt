"""
================================================================================
  VIEWS/TAB6_ILP_OPTIMIZATION.PY - SEKME 6: INTEGER LINEAR PROGRAMMING (ILP/MILP)
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.ilp_pulp_solver import solve_ilp_pulp

def render_tab6(params):
    """Sekme 6 içeriğini çizer: ILP / MILP Matematiksel Optimizasyon Çözücüsü ve Grafikler."""
    st.markdown("## 🎯 Integer Linear Programming (ILP / MILP - Tam Sayılı Programlama)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #1d4ed8;">
<div class="card-title" style="color: #1e40af;">📐 ILP Matematiksel Model Çerçevesi</div>
<ul>
<li><b>İkili Karar Değişkenleri (x<sub>i,t,k</sub> &isin; {0, 1}):</b> İşçi <i>i</i>, gün <i>t</i>, vardiya <i>k</i>.</li>
<li><b>Sert Kısıtlar (%100 Garanti):</b> 4857 Sayılı Kanun 11h dinlenme, 4-posta 1 gün hafta tatili, 7/24 kesintisiz fırın kadrosu ve MYK ehliyet zorunluluğu doğrusal eşitlik/eşitsizlik olarak eklenir.</li>
<li><b>CBC Branch-and-Bound Solver:</b> Dal-Sınır (Branch-and-Bound) algoritması ile tüm arama uzayını matematiksel olarak tarar.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #059669;">
<div class="card-title" style="color: #047857;">🏆 Küresel Optimizasyon (min Z)</div>
<ul>
<li><b>Amaç Fonksiyonu (min Z):</b> Sert kısıtların tamamını %100 sağlarken, 5 yumuşak ceza puanını küresel düzeyde minimuma indirir.</li>
<li><b>Garanti:</b> Eğer veri setine göre bir çözüm varsa, ILP en az ceza puanına sahip <b>EN İYİ ÇİZELGEYİ</b> bulur.</li>
<li><b>Performans:</b> Karmaşıklık yüksek olmasına rağmen PuLP/CBC ile saniyeler içinde çözülür.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- TAM MATEMATİKSEL AMAÇ FONKSİYONU DENKLEM KARTI ---
    st.markdown("### 🧮 MILP Solver Tarafından Optimize Edilen Amaç Fonksiyonu (Objective Function)")
    
    w_circ = params['weights'].get('circadian', 50)
    w_pref = params['weights'].get('pref_off', 40)
    w_posta = params['weights'].get('posta', 35)
    w_exp = params['weights'].get('exp_mix', 30)
    w_night = params['weights'].get('night_imb', 25)

    st.markdown(rf"""
    <div style="background-color: #f8fafc; border: 2px solid #2563eb; border-radius: 10px; padding: 18px; margin-bottom: 20px;">
    <h4 style="color: #1e40af; margin-top: 0;">min Z (Toplam Minimize Edilen Yumuşak Ceza Skoru)</h4>
    <p style="font-size: 1.05rem; color: #1e293b;">
    ILP / MILP Çözücüsü aşağıdaki 5 yumuşak kısıt ceza teriminin toplamını küresel düzeyde minimize etmektedir:
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.latex(rf"""
    \min Z = {w_circ} \cdot \sum_{{i=1}}^{{N}} \sum_{{t=1}}^{{D-1}} s_{{i,t}} 
    + {w_pref} \cdot \sum_{{i=1}}^{{N}} v_i 
    + {w_posta} \cdot \sum_{{p \in \{{\text{{A,B,C,D}}\}}}} \sum_{{t=1}}^{{D}} P_{{p,t}} 
    + {w_exp} \cdot \sum_{{t=1}}^{{D}} \sum_{{k=1}}^{{3}} U_{{t,k}} 
    + {w_night} \cdot \sum_{{i=1}}^{{N}} (d_i^+ + d_i^-)
    """)

    st.markdown("""
    **Değişken Açıklamaları:**
    - **$s_{i,t} \in \{0, 1\}$:** Sirkadiyen Ritim İhlali (Akşam vardiyasından Gündüz vardiyasına dinlenmesiz geçiş).
    - **$v_i \in \{0, 1\}$:** Kişisel İzin Talebi İhlali (İşçinin talep ettiği günde vardiyaya yazılması).
    - **$P_{p,t} \in \{0, 1\}$:** Posta Takım Bütünlüğü İhlali ($p$ postasının $t$ gününde farklı aktif vardiyalara bölünmesi).
    - **$U_{t,k} \in \{0, 1\}$:** Kıdemli Usta Eksikliği ($t$ günündeki $k$ vardiyasında en az 1 Kıdemli Usta bulunmaması).
    - **$d_i^+, d_i^- \ge 0$:** Gece Nöbet Dengesizliği (İşçi $i$'nin hedef ortalama gece nöbeti sayısından pozitif/negatif sapması).
    """)

    st.divider()

    st.markdown("### 🎛️ ILP / MILP Solver Zaman Limiti Kontrolü")
    time_limit = st.slider("Maksimum Solver Süre Limiti (Saniye)", 5, 60, 10, 5, key="ilp_tl")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # --- MILP MODEL KARMAŞIKLIĞI PANELİ ---
    total_vars = int(5 * num_workers * num_days + 40 * num_days + 3 * num_workers)
    total_constraints = int(5 * num_workers * num_days + 30 * num_days)

    if total_vars < 1000:
        model_class_str = "Küçük"
    elif total_vars <= 2500:
        model_class_str = "Orta"
    else:
        model_class_str = "Büyük NP-Hard"

    st.markdown("#### 🧮 MILP Matematiksel Model Boyutu & Karmaşıklık Paneli (Ön-Analiz)")

    pred_c1, pred_c2, pred_c3 = st.columns(3)
    with pred_c1:
        st.metric(
            label="Karar Değişkeni Sayısı (Decision Vars)",
            value=f"{total_vars:,} Değişken",
            help="İkili (Binary 0-1) atama ve kesikli yumuşak kısıt bağlayıcı karar değişkenleri sayısı."
        )
    with pred_c2:
        st.metric(
            label="Kısıt Denklem Sayısı (Constraints)",
            value=f"{total_constraints:,} Denklem",
            help="Modele eklenen tüm sert (hard) ve yumuşak (soft) eşitlik/eşitsizlik denklemleri sayısı."
        )
    with pred_c3:
        st.metric(
            label="Model Büyüklük Sınıfı",
            value=f"{model_class_str}",
            help="Problemin değişken ve kısıt boyutuna göre matematiksel ölçek sınıfı."
        )

    st.divider()

    run_btn = st.button("🚀 ILP / MILP Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t6")

    if not run_btn and "res_t6" not in st.session_state:
        st.warning("👈 Optimizasyonu başlatmak için yukarıdaki **'🚀 ILP / MILP Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        with st.spinner("⏳ ILP / MILP Dal-Sınır (Branch & Bound) Çözücüsü Çalıştırılıyor... Lütfen Bekleyiniz..."):
            results = solve_ilp_pulp(
                n_workers=num_workers,
                n_days=num_days,
                r_day=req_day,
                r_eve=req_eve,
                r_night=req_night,
                weights=weights,
                time_limit=time_limit,
                custom_workers=custom_workers
            )
            st.session_state["res_t6"] = results
    else:
        results = st.session_state["res_t6"]

    # --- DURUMA GÖRE NET VE DOĞRU HATA BİLDİRİMİ ---
    if results['is_infeasible']:
        st.error(f"""
        ❌ **MATEMATİKSEL OLARAK İMKANSIZ (Infeasible):**  
        Girilen personel sayısı ($N={num_workers}$) veya sertifikalı çalışan kadrosu, günlük vardiya ihtiyaçlarını ($R_G={req_day}, R_A={req_eve}, R_N={req_night}$) ve kanuni dinlenme kısıtlarını sağlamak için **fiziksel olarak yetersizdir.**  
        """)
    elif results['is_timeout'] and not results['is_hard_feasible']:
        st.error(f"""
        ⏱️ **ZAMAN LİMİTİ YETERSİZ (Timeout):**  
        ILP çözücüsü **{time_limit} saniyelik** süre sınırı içerisinde henüz geçerli bir vardiya çizelgesi üretemeden kesilmiştir.  
        👉 **Öneri:** Süreyi artırın (örn: 30-60 sn) veya bu boyutta çok daha hızlı çalışan **Sekme 7 (Hill Climbing)** çözücüsüne geçin.
        """)
    elif results['is_timeout'] and results['is_hard_feasible']:
        st.warning(f"""
        ⏱️ **ZAMAN LİMİTİ KESİNTİSİ UYARISI ({time_limit} Saniye Doldu):**  
        ILP çözücüsü belirlenen **{time_limit} saniyelik** zaman limitine ulaştığı için arama %100 tamamlanmadan durdurulmuştur.  
        📌 **Önemli:** Elde edilen bu çizelge **Sert Kısıtları %100 Sağlayan Geçerli (Feasible)** ve en iyi bulunan çözümdür. 
        Teorik alt sınıra ($Z_{{alt}} = {results['best_bound']}$) olan sapması sadece **%{results['mip_gap']}** kadardır.
        """)
    elif results['is_optimal']:
        st.success("🏆 **%100 KÜRESEL OPTİMAL ÇÖZÜM KANITLANDI:** Solver tüm arama uzayını tamamladı ve matematiksel olarak en az ceza puanına sahip çözümü buldu.")

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        status_clr = "#059669" if results['is_optimal'] else ("#dc2626" if (results['is_infeasible'] or not results['is_hard_feasible']) else "#d97706")
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">ILP Solver Durumu</div>
        <div class="metric-value" style="color: {status_clr}; font-size: 1.05rem;">{results['status_text']}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        obj_disp = results['objective_value'] if results['is_hard_feasible'] else "—"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Bulunan Amaç Skoru (Z)</div>
        <div class="metric-value" style="color: #1d4ed8;">{obj_disp}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        bb_disp = results['best_bound'] if results['is_hard_feasible'] else "—"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Teorik Alt Sınır (Best Bound)</div>
        <div class="metric-value" style="color: #7c3aed;">{bb_disp}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        gap_disp = f"%{results['mip_gap']}" if results['is_hard_feasible'] else "—"
        gap_clr = "#059669" if results['mip_gap'] == 0 else "#d97706"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">MIP Sapma Oranı (MIP Gap)</div>
        <div class="metric-value" style="color: {gap_clr};">{gap_disp}</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Hesaplama Süresi</div>
        <div class="metric-value" style="color: #059669;">{results['exec_time_ms']} ms</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- 3 YÖNTEMİN KARŞILAŞTIRMA MATRİSİ ---
    st.markdown("### 📊 3 Temel Yaklaşımın Bütüncül Karşılaştırma Matrisi (Greedy vs. CSP vs. ILP)")

    df_3way = pd.DataFrame({
        "Karşılaştırma Özelliği": [
            "Matematiksel Modelleme Temeli",
            "Sert Kısıt Garantisi (%100 Feasibility)",
            "Küresel En İyi (Global Optimal) Garantisi",
            "Geri İzleme / Budama Mekanizması",
            "Hesaplama Hızı"
        ],
        "Sekme 4: Greedy (Açgözlü)": [
            "Lokal Sezgisel Kurallar",
            "❌ Düşük (Pazar günü kadro çöker)",
            "❌ Yok (Yerel minimumda takılır)",
            "❌ Yok",
            "⚡ Çok Hızlı (< 5 ms)"
        ],
        "Sekme 5: Backtracking (CSP)": [
            "Kısıt Tatmin Çerçevesi (DFS Tree)",
            "✅ %100 Garanti (Geçerli Çözüm)",
            "❌ Yok (İlk geçerli çözüme kilitlenir)",
            "✅ Var (Arama ağacında geri adımlar)",
            "⏱️ Hızlı / Orta (Budama ile birkaç ms)"
        ],
        "Sekme 6: ILP / MILP (PuLP Solver)": [
            "Tam Sayılı Programlama (Branch & Bound)",
            "✅ %100 Kusursuz Garanti",
            "🏆 MATEMATİKSEL GARANTİ (min Z*)",
            "✅ Var (Dal-Sınır Ağacı)",
            "⏱️ Optimal Çözüm (Milisaniyeler)"
        ]
    })

    st.dataframe(df_3way, width="stretch", hide_index=True)

    st.divider()

    # --- PERSONEL YETKİNLİK VE MYK SERTİFİKA KADRO TABLOSU ---
    st.markdown("### 🪪 Aktif Personel Yetkinlik ve MYK Sertifika Kadro Listesi")
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

    # --- GÖRSEL GRAFİKLER VE ÇİZELGE ---
    if not results['is_hard_feasible']:
        st.warning("⚠️ **Geçerli Bir Çözüm Üretilemedi:** Solver imkansızlık (Infeasible) veya zaman aşımı nedeniyle geçerli bir atama oluşturamadığından grafikler ve çizelge gizlenmiştir.")
        return

    st.markdown("### 🎨 ILP / MILP Matematiksel Optimizasyon Grafikleri")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1️⃣ ILP Optimal Vardiya Matrisi Isı Haritası (Heatmap)")
        fig_ilp_map = px.imshow(
            results['schedule'],
            labels=dict(x="Günler", y="Çalışanlar", color="Vardiya (0:OFF, 1:G, 2:A, 3:N)"),
            x=[f"G{d+1}" for d in range(num_days)],
            y=[w['name'] for w in results['workers']],
            color_continuous_scale=[[0, '#cbd5e1'], [0.33, '#fde047'], [0.66, '#f97316'], [1.0, '#1e3a8a']],
            aspect="auto"
        )
        fig_ilp_map.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_ilp_map, width="stretch", key="t6_ilp_map")

    with g_col2:
        st.markdown("##### 2️⃣ ILP Personel Vardiya & Gece Nöbet Dağılımı")
        ilp_worked = [np.sum(results['schedule'][i, :] > 0) for i in range(num_workers)]
        ilp_night = [np.sum(results['schedule'][i, :] == 3) for i in range(num_workers)]
        
        df_ilp_workload = pd.DataFrame({
            "İşçi": [w['name'] for w in results['workers']],
            "Toplam Çalışma": ilp_worked,
            "Gece Nöbeti": ilp_night
        })
        
        fig_ilp_wl = px.bar(
            df_ilp_workload,
            x="İşçi",
            y=["Toplam Çalışma", "Gece Nöbeti"],
            barmode="group",
            color_discrete_sequence=["#059669", "#dc2626"]
        )
        fig_ilp_wl.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_ilp_wl, width="stretch", key="t6_ilp_wl")

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        st.markdown("##### 3️⃣ ILP Minimize Edilmiş Yumuşak Kısıt Cezaları (min Z)")
        df_ilp_penalties = pd.DataFrame({
            "Kısıt Tipi": list(results['penalties'].keys()),
            "Ceza Puanı": list(results['penalties'].values())
        })
        fig_ilp_pen = px.bar(
            df_ilp_penalties,
            x="Ceza Puanı",
            y="Kısıt Tipi",
            orientation="h",
            text="Ceza Puanı",
            color="Ceza Puanı",
            color_continuous_scale="Reds"
        )
        fig_ilp_pen.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=340, showlegend=False)
        st.plotly_chart(fig_ilp_pen, width="stretch", key="t6_ilp_pen")

    with g_col4:
        st.markdown("##### 4️⃣ ILP Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")
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
        st.plotly_chart(fig_posta, width="stretch", key="t6_posta_load")

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
        st.plotly_chart(fig_sp_all, width="stretch", key="t6_sp_all")

    with v_tab2:
        selected_shift = st.selectbox("İncelenecek Vardiyayı Seçin:", ["Gündüz (08-16)", "Akşam (16-24)", "Gece (24-08)"], key="t6_vselect")
        df_sub = df_shift_posta[df_shift_posta["Vardiya"] == selected_shift]
        
        fig_sp_sub = px.bar(
            df_sub,
            x="Gün",
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
        fig_sp_sub.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=400, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_sp_sub, width="stretch", key="t6_sp_sub")

    st.divider()

    # --- FULL OPTIMAL SCHEDULE MATRIX TABLE ---
    st.markdown("### 🗓️ ILP / MILP Tarafından Üretilen Matematiksel Olarak Optimal Vardiya Çizelgesi")
    shift_names = {0: "OFF", 1: "Gündüz", 2: "Akşam", 3: "Gece"}
    matrix_data = []
    
    for i, w in enumerate(results['workers']):
        row = {"İşçi": w['name'], "Posta": w['posta'], "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi"}
        for d in range(num_days):
            row[f"Gün {d+1}"] = shift_names[results['schedule'][i, d]]
        matrix_data.append(row)

    df_ilp_view = pd.DataFrame(matrix_data)
    st.dataframe(df_ilp_view, width="stretch", hide_index=True)
