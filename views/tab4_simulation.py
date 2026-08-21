"""
================================================================================
  VIEWS/TAB4_SIMULATION.PY - SEKME 4: YAPICI SEZGİSELLER (GREEDY HEURISTICS)
================================================================================
  Bu sekme, Yöneylem Araştırması & Çizelgeleme literatüründeki
  Yapıcı Sezgiselleri (Constructive / Greedy Heuristics):
  1. Sıralı Miyopik Yaklaşım (Sequential Myopic Greedy)
  2. Kademeli Desen Yaklaşımı (Staggered Pattern-Based Greedy)
  3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)
  modellerini, Pazar İzin Çöküşünü ve kısıt analizlerini görselleştirir.
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.greedy_solver import run_greedy_algorithm
from views.common_components import (
    render_schedule_heatmap,
    render_workload_chart,
    render_penalties_chart,
    render_posta_load_chart,
    render_shift_posta_stacked_chart,
    render_schedule_matrix_table,
    render_request_details_expander,
    render_worker_profiles_table,
    render_evaluator_cost_badge
)

def render_tab4(params):
    """Sekme 4 içeriğini çizer: Greedy Simülasyonu, Sert İhlal Analizi, İzin Talepleri ve Grafikler."""
    st.markdown("## ⚡ Sekme 4: Yapıcı Sezgiseller (Constructive / Greedy Heuristics) Vardiya Simülasyonu")

    # --- TEORİK KAVRAM KARTLARI (LİTERATÜR TANITIMI) ---
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #059669;">
<div class="card-title" style="color: #047857;">🧩 Yapıcı Sezgisel (Constructive Heuristic) Çerçevesi</div>
<ul>
<li><b>Tek Geçişli İnşa (Single-Pass Construction):</b> Çizelge 1. günden son güne kadar sırayla, boş matris doldurularak inşa edilir.</li>
<li><b>Geri Dönüşsüz Karar (No Backtracking):</b> O anki vardiya için atanan işçi sabittir; sonraki günlerde kısıt çıkmazı yaşansa dahi geçmişe dönüp düzeltme yapılmaz.</li>
<li><b>Açgözlü (Greedy) Seçim Kuralı:</b> Her vardiya için o an uygun adaylar arasından hedef postaya ait olan ve sertifikayı sağlayan işçiye yerel öncelik verilir.</li>
<li><b>Yüksek Hesaplama Hızı [O(N · D)]:</b> Milisaniyeler içinde çizelge üretir; bu sayede metasezgisellere (Genetik, Tavlama vb.) kaliteli başlangıç çözümü sağlar.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with c_col2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb;">
<div class="card-title" style="color: #1d4ed8;">⚖️ Literatürdeki 3 Temel Yapıcı Yaklaşım Modu</div>
<ul>
<li><b>1. Sıralı / Miyopik Açgözlü (Sequential Myopic):</b> İleriye bakışsızdır. İşçilerin 6 gün çalışıp 7. gün kanuni izne çıkacağını öngöremez; bu yüzden pazar günleri toplu izin yığılması (<b>Pazar İzin Çöküşü / End-of-Horizon Shortage</b>) yaşanır.</li>
<li><b>2. Kademeli / Desen Tabanlı (Staggered Pattern):</b> Zorunlu haftalık izinleri işçilere modüler rotasyonla [<i>i mod 7</i>] kademeli dağıtır. Hafta sonu kadro çöküşünü önler.</li>
<li><b>3. Kısıt Öncelikli (MRV / LCV Tabanlı):</b> <b>MRV (Minimum Remaining Values)</b> ile darboğaz MYK sertifikalıları ve ustaları önceden güvenceye alır; <b>LCV (Least Constraining Value)</b> ile kişisel izin taleplerini korur ve joker ehliyetlileri saklar.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # Algoritma Modu Seçimi
    st.markdown("### 🎛️ Yapıcı Sezgisel (Greedy) Çözüm Modunu Seçin")
    solver_mode = st.radio(
        "🧠 Yaklaşım Modu:",
        options=[
            "1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)",
            "2. Kademeli / Desen Tabanlı Yapıcı Sezgisel (Staggered Pattern-Based Greedy)",
            "3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)"
        ],
        help="Sıralı mod pazar izin çöküşü yaşar. Kademeli mod zorunlu izinleri haftaya eşit yayar. MRV/LCV modu ise darboğaz sertifikaları ve izin taleplerini akıllıca önceliklendirir.",
        horizontal=True,
        key="t4_mode_radio"
    )

    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    penalty_weights_dict = params['weights']
    custom_workers = params['custom_workers']

    # HESAPLAMA YÜKÜ & ÖN-ANALİZ PANELİ
    cost_ms = params.get('call_cost_ms', round(0.001492 * (num_workers * num_days) + 0.1670, 3))
    t4_time_str = f"{cost_ms:.2f} ms"

    st.markdown("#### 🧮 Hesaplama Yükü & Ceza Değerlendirici Tahmin Paneli (Ön-Analiz)")
    c_est1, c_est2 = st.columns(2)
    with c_est1:
        render_evaluator_cost_badge(1, cost_ms, label="Tahmini Ceza Değerlendirme")
    with c_est2:
        st.metric(
            label="🔍 Zaman Karmaşıklığı (Time Complexity)",
            value=f"O(N · D) ≈ {num_workers * num_days} Adım",
            help="Her hücre için sabit sayıda işlemle tek yönlü atama yapılır."
        )

    st.divider()

    run_btn = st.button("🚀 Greedy Simülasyonu Çalıştır", type="primary", width="stretch", key="btn_run_t4")

    if not run_btn and "res_t4" not in st.session_state:
        st.warning("👈 Simülasyonu başlatmak için yukarıdaki **'🚀 Greedy Simülasyonu Çalıştır'** butonuna basınız.")
        return

    if run_btn:
        with st.spinner("⏳ Greedy Vardiya Simülasyonu Çalıştırılıyor..."):
            results = run_greedy_algorithm(
                num_workers, num_days, req_day, req_eve, req_night, penalty_weights_dict, solver_mode=solver_mode, custom_workers=custom_workers
            )
            st.session_state["res_t4"] = results
            if "Miyopik" in solver_mode or "Sıralı" in solver_mode:
                st.session_state["res_t4_myopic"] = results
            elif "Kademeli" in solver_mode:
                st.session_state["res_t4_staggered"] = results
            elif "MRV" in solver_mode or "Kısıt Öncelikli" in solver_mode:
                st.session_state["res_t4_mrv"] = results
    else:
        results = st.session_state["res_t4"]

    if isinstance(results, tuple):
        schedule_matrix, worker_list, penalty_dict, total_score, hard_viols_count, hard_logs, req_details = results
        is_feasible = (hard_viols_count == 0)
    else:
        schedule_matrix = results['schedule']
        worker_list = results['workers']
        penalty_dict = results['penalties']
        total_score = results['final_score']
        hard_viols_count = results['hard_violations_count']
        hard_logs = results['hard_violation_logs']
        req_details = results['request_details']
        is_feasible = results.get('is_feasible', hard_viols_count == 0)

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam Ceza Puanı</div>
        <div class="metric-value" style="color: #dc2626;">{total_score}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        status_text = "GEÇERLİ" if is_feasible else "❌ GEÇERSİZ"
        status_color = "#059669" if is_feasible else "#dc2626"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Sert Kısıt Durumu</div>
        <div class="metric-value" style="color: {status_color}; font-size: 1.15rem;">{status_text}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        hard_color = "#059669" if hard_viols_count == 0 else "#dc2626"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Sert Kısıt İhlali</div>
        <div class="metric-value" style="color: {hard_color};">{hard_viols_count}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        fulfilled_count = sum(1 for r in req_details if "Karşılandı" in r['durum'])
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İzin Talebi Başarısı</div>
        <div class="metric-value" style="color: #059669;">%{int((fulfilled_count/num_workers)*100)}</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Ceza Çağrısı (Evaluator)</div>
        <div class="metric-value" style="color: #2563eb;">1 Adet</div>
        </div>""", unsafe_allow_html=True)

    with m6:
        total_assignments = np.sum(schedule_matrix > 0)
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam Vardiya Ataması</div>
        <div class="metric-value" style="color: #475569;">{total_assignments}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # --- SERT KISIT İHLALLERİ VE UYARI PANELİ ---
    if hard_viols_count > 0:
        st.error(f"⚠️ **Sert Kısıt İhlali / Kadro Eksikliği:** {hard_viols_count} Adet Sert Kısıt İhlali veya Vardiya Kadro Eksikliği Tespit Edildi!")
        with st.expander("📌 Sert Kısıt İhlalleri ve Vardiya Kadro Eksiklikleri Detayı", expanded=True):
            df_logs = pd.DataFrame({"İhlal Açıklaması ve Vardiya Detayı": hard_logs})
            st.dataframe(df_logs, width="stretch", hide_index=True)
    else:
        st.success("✅ **BAŞARILI:** Hiçbir Sert Kısıt ihlali yaşanmadı! Vardiya kadro ve zorunlu sertifika ihtiyaçları (%100) eksiksiz karşılandı.")

    st.divider()

    # --- PERSONEL YETKİNLİK VE MYK SERTİFİKA KADRO TABLOSU ---
    render_worker_profiles_table(worker_list, title="🪪 Aktif Personel Yetkinlik ve MYK Sertifika Kadro Listesi")

    st.divider()

    # --- KİŞİSEL İZİN TALEPLERİ PERFORMANS VE YÜZDESEL ANALİZ PANOLARI ---
    st.markdown("### 📊 Kişisel İzin Talepleri Yüzdesel Karşılanma & İhlal Analizi")
    
    total_reqs = len(req_details)
    fulfilled_reqs = sum(1 for r in req_details if "Karşılandı" in r['durum'])
    violated_reqs = total_reqs - fulfilled_reqs
    fulfillment_rate = round((fulfilled_reqs / max(1, total_reqs)) * 100, 1)
    violation_rate = round((violated_reqs / max(1, total_reqs)) * 100, 1)
    total_pref_penalty = sum(r['ceza_puani'] for r in req_details)

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    with p_col1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam İzin Talebi</div>
        <div class="metric-value" style="color: #2563eb;">{total_reqs} Kişi</div>
        </div>""", unsafe_allow_html=True)

    with p_col2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Karşılanma Oranı (OFF Verilen)</div>
        <div class="metric-value" style="color: #059669;">%{fulfillment_rate}</div>
        </div>""", unsafe_allow_html=True)

    with p_col3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">İhlal Oranı (Vardiyaya Yazılan)</div>
        <div class="metric-value" style="color: #dc2626;">%{violation_rate}</div>
        </div>""", unsafe_allow_html=True)

    with p_col4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Eklenen Toplam İzin Cezası</div>
        <div class="metric-value" style="color: #d97706;">{total_pref_penalty} Puan</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    p_chart_col1, p_chart_col2 = st.columns([1, 1])

    with p_chart_col1:
        df_pref_pie = pd.DataFrame({
            "Durum": [f"✅ Karşılanan Talepler (%{fulfillment_rate})", f"❌ İhlal Edilen Talepler (%{violation_rate})"],
            "Sayı": [fulfilled_reqs, violated_reqs]
        })
        fig_pref_pie = px.pie(
            df_pref_pie,
            names="Durum",
            values="Sayı",
            color="Durum",
            color_discrete_map={
                f"✅ Karşılanan Talepler (%{fulfillment_rate})": "#059669",
                f"❌ İhlal Edilen Talepler (%{violation_rate})": "#dc2626"
            },
            hole=0.45
        )
        fig_pref_pie.update_layout(paper_bgcolor="#ffffff", height=300, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pref_pie, width="stretch", key="t4_pref_pie")

    with p_chart_col2:
        st.markdown(f"""
        <div class="card-box" style="border-top: 5px solid #2563eb;">
        <div class="card-title" style="color: #1d4ed8;">ℹ️ İzin Karşılanma Performans Özeti</div>
        <p style="font-size: 1.05rem; color: #1e293b;">
        Çalışanların talep ettiği toplam <b>{total_reqs} adet kişisel iznin %{fulfillment_rate} kadarı ({fulfilled_reqs} kişi)</b> başarıyla karşılanmış ve işçiye izin verilmiştir.
        </p>
        <p style="font-size: 1.05rem; color: #1e293b;">
        Kadro ihtiyaçları ve sertifika zorunlulukları nedeniyle <b>%{violation_rate} oranındaki izin talebi ({violated_reqs} kişi)</b> karşılanamayarak vardiyaya yazılmış ve toplam <b>{total_pref_penalty} ceza puanı</b> amaç fonksiyonuna eklenmiştir.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # --- GÖRSEL GÖSTERİMLER ---
    st.markdown("### 🎨 Görsel Gösterimler ve Posta Takım Bütünlüğü Analizleri")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("##### 1️⃣ Vardiya Dağılım Matrisi Isı Haritası (Heatmap)")
        fig_heatmap = px.imshow(
            schedule_matrix,
            labels=dict(x="Günler", y="Çalışanlar", color="Vardiya (0:OFF, 1:G, 2:A, 3:N)"),
            x=[f"G{d+1}" for d in range(num_days)],
            y=[w['name'] for w in worker_list],
            color_continuous_scale=[[0, '#e2e8f0'], [0.33, '#fde047'], [0.66, '#f97316'], [1.0, '#1e3a8a']],
            aspect="auto"
        )
        fig_heatmap.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380)
        st.plotly_chart(fig_heatmap, width="stretch", key="t4_heatmap")

    with viz_col2:
        st.markdown("##### 2️⃣ Toplam Vardiya Türü Dağılım Oranı (Pie Chart)")
        shift_counts = {
            "İzin (OFF)": np.sum(schedule_matrix == 0),
            "Gündüz (08-16)": np.sum(schedule_matrix == 1),
            "Akşam (16-24)": np.sum(schedule_matrix == 2),
            "Gece (24-08)": np.sum(schedule_matrix == 3)
        }
        df_pie = pd.DataFrame({"Vardiya": list(shift_counts.keys()), "Sayı": list(shift_counts.values())})
        
        fig_pie = px.pie(
            df_pie,
            names="Vardiya",
            values="Sayı",
            color="Vardiya",
            color_discrete_map={
                "İzin (OFF)": "#94a3b8",
                "Gündüz (08-16)": "#eab308",
                "Akşam (16-24)": "#f97316",
                "Gece (24-08)": "#1e3a8a"
            },
            hole=0.4
        )
        fig_pie.update_layout(paper_bgcolor="#ffffff", height=380, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig_pie, width="stretch", key="t4_pie")

    viz_col3, viz_col4 = st.columns(2)

    with viz_col3:
        render_penalties_chart(penalty_dict, key_prefix="t4", title="3️⃣ Yumuşak Kısıt Ceza Puanı Dağılımı")

    with viz_col4:
        render_posta_load_chart(schedule_matrix, worker_list, key_prefix="t4", title="4️⃣ Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı")

    # 5. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI
    render_shift_posta_stacked_chart(schedule_matrix, worker_list, num_days, key_prefix="t4", title="5️⃣ Gün ve Vardiya Bazında Posta Dağılımı (Gündüz, Akşam ve Gece Vardiyalarında Hangi Postadan Kaç Kişi Var?)")

    st.divider()

    # --- VARDİYA ÇİZELGESİ MATRIX TABLOSU ---
    render_schedule_matrix_table(schedule_matrix, worker_list, num_days, title="🗓️ Tam Vardiya Çizelge Tablosu")

    # --- KİŞİSEL İZİN TALEPLERİ DETAY RAPORU ---
    render_request_details_expander(req_details)

    st.divider()

    # --- LİTERATÜR VE YÖNEYLEM DEĞERLENDİRMESİ ---
    st.info("""
    💡 **Yöneylem Araştırması Analizi (Yapıcı Sezgisellerin Başarımı ve Sınırları):**
    - **Güçlü Yönü:** O(N × D) zaman karmaşıklığı ile 1 milisaniyeden kısa sürede tam bir çizelge kurar. Bu nedenle sezgisel ve metasezgisel algoritmalar için mükemmel bir **Başlangıç Çözümü Üreticisi (Initial Feasible Solution Generator)** olarak görev yapar.
    - **Sınırları:** İlk bulduğu yerel kararlara kilitlenir. Geri izleme (Backtracking) yapmadığı için miyopik modda pazar krizleri yaşayabilir; yerel arama yapmadığı için yumuşak kısıtları bir **MILP (Tam Sayılı Programlama)** veya **Metasezgisel (GA, SA, Tabu)** kadar minimize edemez.
    """)
