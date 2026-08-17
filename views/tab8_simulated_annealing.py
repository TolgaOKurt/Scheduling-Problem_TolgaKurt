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
from views.common_components import (
    render_schedule_heatmap,
    render_workload_chart,
    render_penalties_chart,
    render_posta_load_chart,
    render_shift_posta_stacked_chart,
    render_schedule_matrix_table,
    render_request_details_expander
)

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

    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    sa_est_calls = int(min(k_cooling_needed, max_iter) * 0.6)
    sa_total_ms = sa_est_calls * cost_ms
    sa_time_str = f"{sa_total_ms/1000:.2f} sn" if sa_total_ms >= 1000 else f"{sa_total_ms:.0f} ms"

    st.markdown("#### 🧮 Sıcaklık Parametrelerine Göre İterasyon Tahmin Paneli (Ön-Analiz)")

    pred_c1, pred_c2, pred_c3 = st.columns(3)
    with pred_c1:
        st.metric(
            label="Gerekli Soğuma Adımı",
            value=f"{k_cooling_needed:,} İterasyon",
            help="Sıcaklığın T_start değerinden T_min değerine düşmesi için gereken matematiksel net adım sayısı."
        )
    with pred_c2:
        st.metric(
            label="Tahmini Ceza Değerlendirme",
            value=f"~{sa_est_calls:,} Çağrı",
            delta=f"{cost_ms:.2f} ms * {sa_est_calls:,} = {sa_time_str}",
            delta_color="off",
            help="Geçerli komşu durumlarında hesaplanacak ceza değerlendirme sayısı."
        )
    with pred_c3:
        st.metric(
            label="İterasyon Sınırı (K_max)",
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
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Başlangıç Skoru</div>
        <div class="metric-value" style="color: #dc2626;">{results['initial_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İyileştirilmiş Skor</div>
        <div class="metric-value" style="color: #059669;">{results['final_score']}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İyileşme Oranı</div>
        <div class="metric-value" style="color: #1d4ed8;">%{results['improvement_rate']}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Kabul Edilen Kötü Hamle</div>
        <div class="metric-value" style="color: #ea580c;">{results['worse_accepted_moves']} / {results['accepted_moves']}</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        eval_c = results.get('eval_count', '-')
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Ceza Çağrısı (Evaluator)</div>
        <div class="metric-value" style="color: #6366f1;">{eval_c:,} Adet</div>
        </div>""", unsafe_allow_html=True)

    with m6:
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
        render_schedule_heatmap(results['schedule'], results['workers'], num_days, key_prefix="t8_sa", title="2️⃣ SA Vardiya Dağılım Isı Haritası (Heatmap)")

    with g_col2:
        render_workload_chart(results['schedule'], results['workers'], key_prefix="t8_sa", title="3️⃣ SA Personel Vardiya & Gece Nöbet Dağılımı", color_seq=["#059669", "#dc2626"])

    g_col3, g_col4 = st.columns(2)

    with g_col3:
        render_penalties_chart(results['penalties'], key_prefix="t8_sa", title="4️⃣ SA Yumuşak Kısıt Ceza Puanı Dağılımı")

    with g_col4:
        render_posta_load_chart(results['schedule'], results['workers'], key_prefix="t8_sa", title="5️⃣ SA Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")

    # 6. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI
    render_shift_posta_stacked_chart(results['schedule'], results['workers'], num_days, key_prefix="t8_sa", title="6️⃣ Gün ve Vardiya Bazında Posta Dağılımı (Gündüz, Akşam ve Gece Vardiyalarında Hangi Postadan Kaç Kişi Var?)")
    
    st.divider()

    # --- FULL SCHEDULE MATRIX TABLE ---
    render_schedule_matrix_table(results['schedule'], results['workers'], num_days, title="🗓️ Simulated Annealing Tarafından Üretilen Vardiya Çizelgesi")
    
    # --- KİŞİSEL İZİN TALEPLERİ DETAY RAPORU ---
    if 'request_details' in results:
        render_request_details_expander(results['request_details'])
