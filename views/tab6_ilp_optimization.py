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
from views.common_components import (
    render_schedule_heatmap,
    render_workload_chart,
    render_penalties_chart,
    render_posta_load_chart,
    render_shift_posta_stacked_chart,
    render_schedule_matrix_table,
    render_request_details_expander,
    render_worker_profiles_table
)

def render_tab6(params):
    """Sekme 6 içeriğini çizer: ILP / MILP Matematiksel Optimizasyon Çözücüsü ve Grafikler."""
    st.markdown("## 🎯 Integer Linear Programming (ILP / MILP - Tam Sayılı Programlama)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #1d4ed8;">
<div class="card-title" style="color: #1e40af;">📐 ILP / MILP Matematiksel Model Çerçevesi</div>
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
<li><b>Garanti:</b> Eğer veri setine göre bir çözüm varsa, solver en az ceza puanına sahip <b>EN İYİ ÇİZELGEYİ</b> bulur.</li>
<li><b>Performans:</b> Karmaşıklık yüksek olmasına rağmen PuLP/CBC ile saniyeler içinde çözülür.</li>
</ul>
</div>""", unsafe_allow_html=True)

    # --- ILP VE MILP FARKI BİLGİ KUTUSU ---
    st.markdown("""
    <div class="card-box" style="border-left: 5px solid #6366f1; background-color: #f8fafc; padding: 18px; margin-top: 15px; margin-bottom: 20px;">
        <div style="font-size: 1.15rem; font-weight: 700; color: #4338ca; margin-bottom: 10px;">
            📚 Yöneylem Araştırması Rehberi: ILP ile MILP Arasındaki Fark Nedir?
        </div>
        <div style="color: #334155; font-size: 1rem; line-height: 1.65;">
            Matematiksel optimizasyon literatüründe karar değişkenlerinin tiplerine göre iki temel doğrusal modelleme sınıfı kullanılır:
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 12px;">
            <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
                <b style="color: #1e40af; font-size: 1.05rem;">🔹 ILP (Integer Linear Programming - Saf Tam Sayılı Programlama):</b><br>
                <span style="font-size: 0.95rem; color: #475569;">
                Modeldeki <b>TÜM</b> karar değişkenleri istisnasız tam sayı (kesikli) veya ikili (<i>Binary</i> &isin; {0, 1}) olmak zorundadır. Hiçbir sürekli (ondalıklı / reel &Ropf;) değişken barındırmaz.
                </span>
            </div>
            <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px;">
                <b style="color: #047857; font-size: 1.05rem;">🔸 MILP (Mixed Integer Linear Programming - Karışık Tam Sayılı Programlama):</b><br>
                <span style="font-size: 0.95rem; color: #475569;">
                Modelde <b>hem tam sayılı / ikili değişkenler hem de sürekli (Continuous &isin; &Ropf;) değişkenler bir arada (Mixed)</b> yer alır.
                </span>
            </div>
        </div>
        <div style="margin-top: 12px; font-size: 0.95rem; color: #1e293b; background-color: #e0e7ff; border-radius: 6px; padding: 10px 14px; line-height: 1.6;">
            🎯 <b>Bizim Çizelgeleme Modelimiz Neden Teknik Olarak Bir MILP'tir?</b><br>
            • <b>İkili (Binary) Kısım:</b> Vardiya atamaları (<i>x<sub>i,t,k</sub></i> &isin; {0, 1}), izin ihlalleri (<i>v<sub>i</sub></i> &isin; {0, 1}), posta bölünmesi (<i>P<sub>p,t</sub></i> &isin; {0, 1}) ve usta eksikliği (<i>U<sub>t,k</sub></i> &isin; {0, 1}) gibi karar değişkenleri ikilidir (0 veya 1).<br>
            • <b>Sürekli (Continuous) Kısım:</b> Gece nöbet adaleti kısıtındaki hedef ortalamadan sapma değişkenleri (<i>d<sub>i</sub><sup>+</sup>, d<sub>i</sub><sup>&minus;</sup></i> &ge; 0 &isin; &reals;) sürekli (ondalıklı) değerler alabildiği için modelimiz <b>MILP (Karışık Tam Sayılı)</b> sınıfındadır.
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    - <b><i>s<sub>i,t</sub></i> &isin; {0, 1} :</b> Sirkadiyen Ritim İhlali (Akşam vardiyasından Gündüz vardiyasına dinlenmesiz geçiş).
    - <b><i>v<sub>i</sub></i> &isin; {0, 1} :</b> Kişisel İzin Talebi İhlali (İşçinin talep ettiği günde vardiyaya yazılması).
    - <b><i>P<sub>p,t</sub></i> &isin; {0, 1} :</b> Posta Takım Bütünlüğü İhlali (<i>p</i> postasının <i>t</i> gününde farklı aktif vardiyalara bölünmesi).
    - <b><i>U<sub>t,k</sub></i> &isin; {0, 1} :</b> Kıdemli Usta Eksikliği (<i>t</i> günündeki <i>k</i> vardiyasında en az 1 Kıdemli Usta bulunmaması).
    - <b><i>d<sub>i</sub><sup>+</sup>, d<sub>i</sub><sup>&minus;</sup></i> &ge; 0 :</b> Gece Nöbet Dengesizliği (İşçi <i>i</i>'nin hedef ortalama gece nöbeti sayısından pozitif/negatif sapması).
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🎛️ ILP / MILP Solver Zaman Limiti Kontrolü")
    time_limit = st.slider("Maksimum Solver Süre Limiti (Saniye)", min_value=5, max_value=120, value=15, step=5, key="ilp_tl")

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

    pred_c1, pred_c2, pred_c3, pred_c4 = st.columns(4)
    with pred_c1:
        st.metric(
            label="Karar Değişkeni (Vars)",
            value=f"{total_vars:,} Adet",
            help="İkili (Binary 0-1) atama ve kesikli yumuşak kısıt bağlayıcı karar değişkenleri sayısı."
        )
    with pred_c2:
        st.metric(
            label="Kısıt Denklem Sayısı",
            value=f"{total_constraints:,} Denklem",
            help="Modele eklenen tüm sert (hard) ve yumuşak (soft) eşitlik/eşitsizlik denklemleri sayısı."
        )
    with pred_c3:
        st.metric(
            label="Model Büyüklük Sınıfı",
            value=f"{model_class_str}",
            help="Problemin değişken ve kısıt boyutuna göre matematiksel ölçek sınıfı."
        )
    with pred_c4:
        st.metric(
            label="Ceza Değerlendirme Tipi",
            value="1 Doğrulama Çağrısı",
            help="MILP katsayıları amaç fonksiyonunda tek hamlede optimize eder; sonuç raporu için 1 kez çağrılır."
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
    meta = results.get('meta', {})
    is_feas = results.get('is_feasible', results.get('is_hard_feasible', False))
    is_infeas = meta.get('is_infeasible', results.get('is_infeasible', False))
    is_timeout = meta.get('is_timeout', results.get('is_timeout', False))
    is_optimal = meta.get('is_optimal', results.get('is_optimal', False))
    status_text = meta.get('status_text', results.get('status_text', ''))
    best_bound = meta.get('best_bound', results.get('best_bound', 0))
    mip_gap = meta.get('mip_gap', results.get('mip_gap', 0.0))

    if is_infeas:
        st.error(f"""
        ❌ **MATEMATİKSEL OLARAK İMKANSIZ (Infeasible):**  
        Girilen personel sayısı (N = {num_workers}) veya sertifikalı çalışan kadrosu, günlük vardiya ihtiyaçlarını (Gündüz: {req_day}, Akşam: {req_eve}, Gece: {req_night}) ve kanuni dinlenme kısıtlarını sağlamak için **fiziksel olarak yetersizdir.**  
        """)
    elif is_timeout and not is_feas:
        st.error(f"""
        ⏱️ **ZAMAN LİMİTİ YETERSİZ (Timeout):**  
        ILP çözücüsü **{time_limit} saniyelik** süre sınırı içerisinde henüz geçerli bir vardiya çizelgesi üretemeden kesilmiştir.  
        👉 **Öneri:** Süreyi artırın (örn: 30-60 sn) veya bu boyutta çok daha hızlı çalışan **Sekme 7 (Hill Climbing)** çözücüsüne geçin.
        """)
    elif is_timeout and is_feas:
        st.warning(f"""
        ⏱️ **ZAMAN LİMİTİ KESİNTİSİ UYARISI ({time_limit} Saniye Doldu):**  
        ILP çözücüsü belirlenen **{time_limit} saniyelik** zaman limitine ulaştığı için arama %100 tamamlanmadan durdurulmuştur.  
        📌 **Önemli:** Elde edilen bu çizelge **Sert Kısıtları %100 Sağlayan Geçerli (Feasible)** ve en iyi bulunan çözümdür. 
        Teorik alt sınıra (Z_alt = {best_bound}) olan sapması sadece **%{mip_gap}** kadardır.
        """)
    elif is_optimal:
        st.success("🏆 **%100 KÜRESEL OPTİMAL ÇÖZÜM KANITLANDI:** Solver tüm arama uzayını tamamladı ve matematiksel olarak en az ceza puanına sahip çözümü buldu.")

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        status_clr = "#059669" if is_optimal else ("#dc2626" if (is_infeas or not is_feas) else "#d97706")
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">ILP Solver Durumu</div>
        <div class="metric-value" style="color: {status_clr}; font-size: 1.05rem;">{status_text}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        obj_disp = results['final_score'] if is_feas else "—"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Amaç Skoru (Z)</div>
        <div class="metric-value" style="color: #1d4ed8;">{obj_disp}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        bb_disp = best_bound if is_feas else "—"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Teorik Alt Sınır</div>
        <div class="metric-value" style="color: #7c3aed;">{bb_disp}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        gap_disp = f"%{mip_gap}" if is_feas else "—"
        gap_clr = "#059669" if mip_gap == 0 else "#d97706"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">MIP Gap</div>
        <div class="metric-value" style="color: {gap_clr};">{gap_disp}</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        eval_c = results.get('eval_count', 1)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Ceza Çağrısı</div>
        <div class="metric-value" style="color: #6366f1;">{eval_c} Adet</div>
        </div>""", unsafe_allow_html=True)

    with m6:
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
    render_worker_profiles_table(results['workers'], title="🪪 Aktif Personel Yetkinlik ve MYK Sertifika Kadro Listesi")

    st.divider()

    # --- GÖRSEL GRAFİKLER VE ÇİZELGE ---
    if not results['is_feasible']:
        st.warning("⚠️ **Geçerli Bir Çözüm Üretilemedi:** Solver imkansızlık (Infeasible) veya zaman aşımı nedeniyle geçerli bir atama oluşturamadığından grafikler ve çizelge gizlenmiştir.")
        return

    st.markdown("### 🎨 ILP / MILP Matematiksel Optimizasyon Grafikleri & Yakınsama Analizi")

    # 1. ZAMANA GÖRE CEZA PUANI İYİLEŞME VE YAKINSAMA GRAFİĞİ (BRANCH & BOUND CONVERGENCE CURVE)
    st.markdown("##### 1️⃣ Branch-and-Bound Zamana Göre Ceza Skoru İyileşme & Yakınsama Grafiği (Convergence Curve)")
    
    conv_hist = results['meta'].get('convergence_history', [])
    if conv_hist and len(conv_hist) > 0:
        df_conv = pd.DataFrame(conv_hist)
        
        fig_conv = go.Figure()
        
        # Trace 1: Bulunan En İyi Çözüm (Incumbent / Primal Bound)
        fig_conv.add_trace(go.Scatter(
            x=df_conv["time_sec"],
            y=df_conv["incumbent"],
            mode="lines+markers",
            name="Bulunan En İyi Çözüm (Incumbent / Primal Bound)",
            line=dict(color="#2563eb", width=3, shape="hv"),
            marker=dict(size=8, color="#1d4ed8"),
            hovertemplate="<b>Geçen Süre:</b> %{x:.2f} sn<br><b>Ceza Skoru:</b> %{y} Puan<extra></extra>"
        ))
        
        # Trace 2: Teorik Alt Sınır (Continuous LP Relaxation / Dual Bound)
        fig_conv.add_trace(go.Scatter(
            x=df_conv["time_sec"],
            y=df_conv["best_bound"],
            mode="lines",
            name=f"Teorik Alt Sınır (Z_alt = {best_bound})",
            line=dict(color="#059669", width=2, dash="dash"),
            hovertemplate="<b>Teorik Alt Sınır:</b> %{y} Puan<extra></extra>"
        ))
        
        fig_conv.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            height=360,
            xaxis_title="Arama Süresi (Saniye)",
            yaxis_title="Toplam Ceza Puanı (Z)",
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_conv, width="stretch", key="t6_fig_conv")
        
        # İyileşme Analiz Metrikleri (KPI Kutuları)
        init_score = df_conv.iloc[0]["incumbent"] if df_conv.iloc[0]["incumbent"] is not None else results['objective_value']
        final_score = results['objective_value']
        score_diff = max(0, init_score - final_score)
        pct_improvement = round((score_diff / max(1, init_score)) * 100, 1) if init_score > 0 else 0
        
        k_c1, k_c2, k_c3, k_c4 = st.columns(4)
        with k_c1:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Kök Düğüm Başlangıç Skoru</div>
            <div class="metric-value" style="color: #64748b;">{init_score:.0f} Puan</div>
            </div>""", unsafe_allow_html=True)
        with k_c2:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Nihai Optimize Skor</div>
            <div class="metric-value" style="color: #1d4ed8;">{final_score} Puan</div>
            </div>""", unsafe_allow_html=True)
        with k_c3:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Toplam Ceza İyileştirmesi</div>
            <div class="metric-value" style="color: #059669;">-{score_diff:.0f} (%{pct_improvement})</div>
            </div>""", unsafe_allow_html=True)
        with k_c4:
            t_opt_disp = f"{results['exec_time_ms']/1000:.2f} sn"
            st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Optimuma Ulaşma Süresi</div>
            <div class="metric-value" style="color: #7c3aed;">{t_opt_disp}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        render_schedule_heatmap(results['schedule'], results['workers'], num_days, key_prefix="t6_ilp", title="2️⃣ ILP Optimal Vardiya Matrisi Isı Haritası (Heatmap)")

    with g_col2:
        render_workload_chart(results['schedule'], results['workers'], key_prefix="t6_ilp", title="3️⃣ ILP Personel Vardiya & Gece Nöbet Dağılımı", color_seq=["#059669", "#dc2626"])

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        render_penalties_chart(results['penalties'], key_prefix="t6_ilp", title="4️⃣ ILP Minimize Edilmiş Yumuşak Kısıt Cezaları (min Z)")

    with g_col4:
        render_posta_load_chart(results['schedule'], results['workers'], key_prefix="t6_ilp", title="5️⃣ ILP Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")

    # 6. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI
    render_shift_posta_stacked_chart(results['schedule'], results['workers'], num_days, key_prefix="t6_ilp", title="6️⃣ Gün ve Vardiya Bazında Posta Dağılımı (Gündüz, Akşam ve Gece Vardiyalarında Hangi Postadan Kaç Kişi Var?)")

    st.divider()

    # --- FULL OPTIMAL SCHEDULE MATRIX TABLE ---
    render_schedule_matrix_table(results['schedule'], results['workers'], num_days, title="🗓️ ILP / MILP Tarafından Üretilen Matematiksel Olarak Optimal Vardiya Çizelgesi")

    # --- KİŞİSEL İZİN TALEPLERİ DETAY RAPORU ---
    if 'request_details' in results:
        render_request_details_expander(results['request_details'])
