"""
================================================================================
  VIEWS/TAB8_SIMULATED_ANNEALING.PY - SEKME 8: SIMULATED ANNEALING (TAVLAMA BENZETİMİ)
================================================================================
"""
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.simulated_annealing_solver import run_simulated_annealing

def render_tab8(params):
    """Sekme 8 içeriğini çizer: Simulated Annealing (Tavlama Benzetimi) Metasezgisel Çözücüsü ve Grafikler."""
    st.markdown("## ♨️ Simulated Annealing (Tavlama Benzetimi & Stokastik Arama)")


    # --- TEORİK KARTLAR ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #ea580c;">
<div class="card-title" style="color: #c2410c;">🔥 Metalurjik Tavlama Analojisi & Sıcaklık (T)</div>
<ul>
<li><b>Yüksek Sıcaklıkta Esneklik (Exploration):</b> Yüksek sıcaklıklarda (T &gt;&gt; 0) algoritma arama uzayını genişçe keşfeder.</li>
<li><b>Metropolis Kriteri:</b> P = exp(-&Delta;Z / T) formülü sayesinde <b>kötüleşen (ceza puanını artıran) hamleleri bile olasılıksal olarak kabul eder.</b></li>
<li><b>Tuzaklardan Kaçış:</b> Bu stokastik sıçramalar sayesinde Hill Climbing'in kilitlendiği yerel minimum (local optimum) çukurlarından kurtulur.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #0284c7;">
<div class="card-title" style="color: #0369a1;">❄️ Geometrik Soğuma (&alpha;) & Kararlı Yerleşim</div>
<ul>
<li><b>Aşamalı Soğuma (Cooling Schedule):</b> Sıcaklık her adımda T<sub>k+1</sub> = &alpha; &middot; T<sub>k</sub> formülü ile yavaşça düşürülür (&alpha; &isin; [0.85, 0.999]).</li>
<li><b>Düşük Sıcaklıkta Kararlılık (Exploitation):</b> Sıcaklık sıfıra yaklaştıkça (T &rarr; 0) kötü hamle kabul olasılığı sıfırlanır ve algoritma Hill Climbing'e dönüşerek en iyi vadiye yerleşir.</li>
<li><b>Denge:</b> Keşif (Exploration) ve Odaklanma (Exploitation) arasındaki mükemmel dengeyi kurar.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- PARAMETRE KONTROL PANELİ ---
    st.markdown("### 🎛️ Simulated Annealing Parametre Kontrol Paneli")
    
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        t_start = st.slider("Başlangıç Sıcaklığı (T_start)", 100.0, 5000.0, 1000.0, 100.0, key="sa_tstart")
    with p_col2:
        t_min = st.select_slider("Minimum Sıcaklık (T_min)", options=[0.1, 0.05, 0.01, 0.005, 0.001], value=0.01, key="sa_tmin")
    with p_col3:
        cooling_rate = st.slider("Soğuma Katsayısı (α)", 0.9800, 0.9999, 0.9900, 0.0005, format="%.4f", key="sa_alpha")
    with p_col4:
        max_iter = st.slider("Maksimum İterasyon (K)", 2000, 30000, 3000, 1000, key="sa_iter")

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # --- MATEMATİKSEL İTERASYON TAHMİN HESAPLAYICISI (ADIM ADIM KESİN FORMÜL) ---
    t_start_val = float(t_start)
    t_min_val = float(t_min)
    cooling_val = float(cooling_rate)

    # Matematiksel Denklem: k_soğuma = ceil( ln(T_min / T_start) / ln(alpha) )
    num_ratio = t_min_val / t_start_val
    log_num = float(np.log(num_ratio))
    log_den = float(np.log(cooling_val))
    
    k_cooling_needed = int(np.ceil(log_num / log_den))
    est_cooling_cpu_ms = round(k_cooling_needed * 1.5, 1)

    st.markdown("#### 🧮 Sıcaklık Parametrelerine Göre İterasyon Tahmin Paneli (Ön-Analiz)")

    pred_c1, pred_c2, pred_c3 = st.columns(3)
    with pred_c1:
        st.metric(
            label="Gerekli Tam Soğuma Adımı (k_soğuma)",
            value=f"{k_cooling_needed:,} İterasyon",
            help="Sıcaklığın T_start değerinden T_min değerine düşmesi için gereken matematiksel net adım sayısı."
        )
    with pred_c2:
        st.metric(
            label="Tam Soğuma İçin Tahmini Süre",
            value=f"~{est_cooling_cpu_ms:,} ms",
            help="Tam soğuma gerçekleşene kadar harcanacak tahmini CPU süresi."
        )
    with pred_c3:
        st.metric(
            label="Girilen İterasyon Sınırı (K_max)",
            value=f"{max_iter:,} İterasyon",
            delta=f"{max_iter - k_cooling_needed:+,} Adım Marj",
            delta_color="normal" if k_cooling_needed <= max_iter else "inverse"
        )

    with st.expander("📐 Matematiksel Formül ve Adım Adım İterasyon Hesabı Detayı", expanded=False):
        st.latex(r"k_{\text{soğuma}} = \left\lceil \frac{\ln\left(\frac{T_{\text{min}}}{T_{\text{başlangıç}}}\right)}{\ln(\alpha)} \right\rceil")
        st.write(f"- **Başlangıç Sıcaklığı (T_başlangıç):** `{t_start_val}`")
        st.write(f"- **Hedef Bitiş Sıcaklığı (T_min):** `{t_min_val}`")
        st.write(f"- **Soğuma Katsayısı (α):** `{cooling_val}`")
        st.write(f"- **Pay ln(T_min / T_başlangıç):** `ln({num_ratio})` = `{log_num:.6f}`")
        st.write(f"- **Payda ln(α):** `ln({cooling_val})` = `{log_den:.6f}`")
        st.write(f"- **Hesaplanan Net Adım Sayısı:** `{log_num:.6f} / {log_den:.6f}` = `{log_num/log_den:.2f}` → **{k_cooling_needed:,} İterasyon**")

    if k_cooling_needed <= max_iter:
        st.success(f"✅ **Soğuma Başarıyla Tamamlanacak:** Seçilen sıcaklık parametreleri ile algoritma **{k_cooling_needed:,}. adımda** hedef T <= {t_min} sıcaklığına ulaşarak soğumayı bitirecektir. Girilen K={max_iter:,} limiti soğumaya izin vermektedir.")
    else:
        st.warning(f"⚠️ **İterasyon Sınırı Sıcaklığı Yarıda Kesecek:** Tam soğuma için **{k_cooling_needed:,} adım** gerekirken, K={max_iter:,} limiti seçildiği için algoritma tam soğumadan {max_iter}. adımda kesilecektir! (Tam soğuma için K >= {k_cooling_needed:,} yapabilirsiniz).")

    st.divider()

    run_btn = st.button("🚀 Simulated Annealing Optimizasyonunu Başlat", type="primary", width="stretch", key="btn_run_t8")

    if not run_btn and "res_t8" not in st.session_state:
        st.warning("👈 Optimizasyonu başlatmak için yukarıdaki **'🚀 Simulated Annealing Optimizasyonunu Başlat'** butonuna basınız.")
        return

    if run_btn:
        with st.spinner("⏳ Simulated Annealing Stokastik Araması Çalıştırılıyor... Lütfen Bekleyiniz..."):
            results = run_simulated_annealing(
                n_workers=num_workers,
                n_days=num_days,
                r_day=req_day,
                r_eve=req_eve,
                r_night=req_night,
                weights=weights,
                t_start=t_start,
                t_min=t_min,
                cooling_rate=cooling_rate,
                max_iterations=max_iter,
                seed=42,
                custom_workers=custom_workers
            )
            st.session_state["res_t8"] = results
    else:
        results = st.session_state["res_t8"]

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
        <div class="metric-label">Kabul Edilen Kötü Hamle (Tuzak Kaçışı)</div>
        <div class="metric-value" style="color: #ea580c;">{results['worse_accepted_moves']} / {results['accepted_moves']}</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Arama Süresi</div>
        <div class="metric-value" style="color: #059669;">{results['exec_time_ms']} ms</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- GÖRSEL GRAFİKLER ---
    st.markdown("### 🎨 Simulated Annealing İterasyon, Sıcaklık & Yakınsama Grafikleri")

    # 1. ÇİFT EKSENLİ İTERASYON BAZLI CEZA PUANI & SICAKLIK DÜŞÜŞ GRAFİĞİ
    st.markdown("##### 1️⃣ İterasyon Bazlı Anlık Skor, En İyi Skor ve Sıcaklık T(k) Yakınsama Grafiği (Double Y-Axis)")
    
    iters = list(range(len(results['curr_score_history'])))
    fig_sa_conv = go.Figure()

    # Sol Y Ekseni 1: Anlık Arama Skoru Z_curr(k)
    fig_sa_conv.add_trace(go.Scatter(
        x=iters, y=results['curr_score_history'],
        name="Anlık Arama Skoru Z_curr(k)",
        line=dict(color="#f97316", width=1.5, dash="dot")
    ))

    # Sol Y Ekseni 2: En İyi Bulunan Skor Z_best(k)
    fig_sa_conv.add_trace(go.Scatter(
        x=iters, y=results['best_score_history'],
        name="En İyi Bulunan Skor Z_best(k)",
        line=dict(color="#059669", width=2.5)
    ))

    # Sağ Y Ekseni: Sıcaklık T(k)
    fig_sa_conv.add_trace(go.Scatter(
        x=iters, y=results['temp_history'],
        name="Sıcaklık T(k)",
        line=dict(color="#0284c7", width=2, dash="dash"),
        yaxis="y2"
    ))

    fig_sa_conv.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=400,
        xaxis=dict(title=dict(text="Arama İterasyonu (K)")),
        yaxis=dict(title=dict(text="Toplam Ceza Skoru Z(k)", font=dict(color="#059669")), tickfont=dict(color="#059669")),
        yaxis2=dict(title=dict(text="Sıcaklık T(k)", font=dict(color="#0284c7")), tickfont=dict(color="#0284c7"), overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.15)
    )
    st.plotly_chart(fig_sa_conv, width="stretch", key="t8_fig_sa_conv")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 2️⃣ SA Vardiya Dağılım Isı Haritası (Heatmap)")
        fig_sa_map = px.imshow(
            results['schedule'],
            labels=dict(x="Günler", y="Çalışanlar", color="Vardiya (0:OFF, 1:G, 2:A, 3:N)"),
            x=[f"G{d+1}" for d in range(num_days)],
            y=[w['name'] for w in results['workers']],
            color_continuous_scale=[[0, '#cbd5e1'], [0.33, '#fde047'], [0.66, '#f97316'], [1.0, '#1e3a8a']],
            aspect="auto"
        )
        fig_sa_map.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_sa_map, width="stretch", key="t8_fig_sa_map")

    with g_col2:
        st.markdown("##### 3️⃣ SA Personel Vardiya & Gece Nöbet Dağılımı")
        sa_worked = [np.sum(results['schedule'][i, :] > 0) for i in range(num_workers)]
        sa_night = [np.sum(results['schedule'][i, :] == 3) for i in range(num_workers)]
        
        df_sa_workload = pd.DataFrame({
            "İşçi": [w['name'] for w in results['workers']],
            "Toplam Çalışma": sa_worked,
            "Gece Nöbeti": sa_night
        })
        
        fig_sa_wl = px.bar(
            df_sa_workload,
            x="İşçi",
            y=["Toplam Çalışma", "Gece Nöbeti"],
            barmode="group",
            color_discrete_sequence=["#059669", "#dc2626"]
        )
        fig_sa_wl.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380, legend=dict(orientation="h", y=1.15))
        st.plotly_chart(fig_sa_wl, width="stretch", key="t8_fig_sa_wl")

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        st.markdown("##### 4️⃣ SA Yumuşak Kısıt Ceza Puanı Dağılımı")
        df_sa_penalties = pd.DataFrame({
            "Kısıt Tipi": list(results['penalties'].keys()),
            "Ceza Puanı": list(results['penalties'].values())
        })
        fig_sa_pen = px.bar(
            df_sa_penalties,
            x="Ceza Puanı",
            y="Kısıt Tipi",
            orientation="h",
            text="Ceza Puanı",
            color="Ceza Puanı",
            color_continuous_scale="Reds"
        )
        fig_sa_pen.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=340, showlegend=False)
        st.plotly_chart(fig_sa_pen, width="stretch", key="t8_fig_sa_pen")

    with g_col4:
        st.markdown("##### 5️⃣ SA Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")
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
        st.plotly_chart(fig_posta, width="stretch", key="t8_posta_load")

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
    st.plotly_chart(fig_sp_all, width="stretch", key="t8_sp_all")
    st.divider()

    # --- BÜTÜNCÜL KARŞILAŞTIRMA MATRİSİ VE ANALİZİ (SEKME 6, 7 VE 8) ---
    st.markdown("### 📊 Matematiksel ve Metasezgisel Çözücülerin Bütüncül Karşılaştırma Matrisi (Sekme 6 vs 7 vs 8)")
    st.info("💡 **Yöneylem Araştırması Rehberi:** Aşağıdaki tablo ve karşılaştırma analizi, işletmenizin ölçeğine ve hız / kesinlik gereksinimlerinize göre hangi algoritmayı seçmeniz gerektiğini özetler.")

    df_matrix = pd.DataFrame({
        "Karşılaştırma Kriteri": [
            "Matematiksel Çözüm Garantisi",
            "Hesaplama Hızı & Tepki Süresi",
            "Ölçeklenebilirlik (Büyük N ve D)",
            "Tuzaklardan (Local Optima) Kaçış",
            "Sert Kısıt Garantisi (Feasibility)",
            "Parametre Hassasiyeti & Ayar İhtiyacı",
            "Önerilen En İdeal Kullanım Alanı"
        ],
        "🏛️ Sekme 6: ILP / MILP (Tam Matematiksel)": [
            "🏆 %100 Küresel Optimum Garantisi (MIP Gap %0)",
            "⏱️ Yavaş / Üstel Artış (Dal-Sınır Ağacı)",
            "🔴 Zayıf (Personel/Gün arttıkça Timeout)",
            "❌ Yok (Kesin sınır budaması yapar)",
            "🛡️ Tam Sert Kısıt Garantisi (Feasibility Proof)",
            "🟢 Düşük (Sadece Zaman Limiti ayarı)",
            "Küçük/Orta ölçekli tesislerde kesin %100 optimal çözüm arandığında"
        ],
        "⚡ Sekme 7: Hill Climbing (Açgözlü Yerel Arama)": [
            "⚠️ Yerel Minimum (Local Optimum) Riski",
            "⚡ Yıldırım Hızında (Saliseler içinde dönüş)",
            "🟢 Mükemmel (Binlerce çalışanda dahi anlık)",
            "❌ Yok (İlk bulduğu tepede kilitlenir)",
            "⚠️ Başlangıç çözümüne bağımlı takip",
            "🟢 Düşük (Sadece İterasyon Sınırı ayarı)",
            "Çok büyük tesislerde anlık taslak çizelge oluşturmak istendiğinde"
        ],
        "♨️ Sekme 8: Simulated Annealing (Stokastik Metasezgi)": [
            "🌟 Yüksek Kaliteli Küresel Optimuma Yakın Çözüm",
            "🚀 Çok Hızlı (Stokastik Kabul / Soğuma)",
            "🟢 Mükemmel (Büyük tesislerde kesintisiz)",
            "✅ Mükemmel (Metropolis Kriteri ile sıçrama yapar)",
            "🛡️ %100 Sert Kısıt Korumalı Değişim Mantığı",
            "🟡 Orta/Yüksek (T_0, T_min, α, K_max hassas ayar)",
            "Orta ve dev ölçekli endüstriyel tesislerde en ideal dengeli çözüm için"
        ]
    })

    st.dataframe(df_matrix, width="stretch", hide_index=True)

    c_m1, c_m2, c_m3 = st.columns(3)

    with c_m1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb;">
<div class="card-title" style="color: #1d4ed8;">🏛️ Sekme 6: ILP / MILP (Neyde Güçlü?)</div>
<ul>
<li><b>Güçlü Tarafı:</b> %100 matematiksel kanıt ve küresel optimum garantisi verir. Bir çözüm imkansızsa (Infeasible) bunu kesin kanıtlar.</li>
<li><b>Neden Tercih Edilir?:</b> Yönetime <i>"Matematiksel olarak bundan daha az ceza puanı imkansızdır"</i> demek ve küçük kadrolarda kesin sonuç almak için.</li>
<li><b>Kısıtı:</b> Kadro N &gt; 40 üzerine çıktığında kombinatoryal patlama nedeniyle zaman aşımına girer.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_m2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #059669;">
<div class="card-title" style="color: #047857;">⚡ Sekme 7: Hill Climbing (Neyde Güçlü?)</div>
<ul>
<li><b>Güçlü Tarafı:</b> İşlemciyi yormayan yıldırım hızında arama hızı. Saliseler içinde binlerce hamle dener.</li>
<li><b>Neden Tercih Edilir?:</b> Canlı sistemde anlık karar vermek veya binlerce personeli olan dev tesislerde saniyeler içinde taslak çizelge üretmek için.</li>
<li><b>Kısıtı:</b> İlk bulduğu "Tepede" (Yerel Minimum) kilitlenir. Yanındaki tepe daha iyi olsa dahi yokuş aşağı hamle yapamadığı için tuzağa düşer.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_m3:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #ea580c;">
<div class="card-title" style="color: #c2410c;">♨️ Sekme 8: Simulated Annealing (Neyde Güçlü?)</div>
<ul>
<li><b>Güçlü Tarafı:</b> <b>Metropolis Kriteri (P = exp(-&Delta;Z / T))</b> sayesinde yerel tuzaklardan sıçrayarak kaçma yeteneği.</li>
<li><b>Neden Tercih Edilir?:</b> Çok hızlı çalışıp Hill Climbing'in tuzaklarına düşmeden ILP kalitesine (%98-99 yakınlıkta) çizelgeler üretmek için.</li>
<li><b>Kısıtı:</b> Soğuma katsayısı (&alpha;) ve sıcaklık parametrelerinin doğru ayarlanmasını gerektirir.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- FULL SCHEDULE MATRIX TABLE ---
    st.markdown("### 🗓️ Simulated Annealing Tarafından Üretilen Vardiya Çizelgesi")
    shift_names = {0: "OFF", 1: "Gündüz", 2: "Akşam", 3: "Gece"}
    matrix_data = []
    
    for i, w in enumerate(results['workers']):
        row = {"İşçi": w['name'], "Posta": w['posta'], "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi"}
        for d in range(num_days):
            row[f"Gün {d+1}"] = shift_names[results['schedule'][i, d]]
        matrix_data.append(row)

    df_sa_view = pd.DataFrame(matrix_data)
    st.dataframe(df_sa_view, width="stretch", hide_index=True)
