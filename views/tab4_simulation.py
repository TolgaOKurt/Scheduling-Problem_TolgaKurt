"""
================================================================================
  VIEWS/TAB4_SIMULATION.PY - SEKME 4: GREEDY SİMÜLASYONU, İZİN TALEPLERİ & GÖRSELLER
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
    render_worker_profiles_table
)

def render_tab4(params):
    """Sekme 4 içeriğini çizer: Greedy Simülasyonu, Sert İhlal Analizi, İzin Talepleri ve Grafikler."""
    st.markdown("## ⚡ Greedy / Heuristic (Açgözlü) Vardiya Çizelgeleme Simülasyonu")


    # Algoritma Modu Seçimi
    solver_mode = st.radio(
        "🧠 Greedy Yaklaşım Modu:",
        options=["Naif Greedy (Standart Açgözlü Yaklaşım)", "Akıllı Kademeli Greedy (İzinleri Günlere Yayan)"],
        help="Naif Greedy, vardiya atamalarını herhangi bir ileriye bakış (lookahead) veya izin yayılımı yapmadan sırayla gerçekleştirir. Akıllı Greedy ise haftalık izin günlerini dengeli biçimde yayar.",
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
        st.metric(
            label="📊 Tahmini Ceza Değerlendirme",
            value="1 Çağrı",
            delta=f"{cost_ms:.2f} ms * 1 = {t4_time_str}",
            delta_color="off",
            help="Doğrudan tek geçişli Greedy ataması sonrası 1 tam ceza değerlendirmesi yapılır."
        )
    with c_est2:
        st.metric(label="🔍 Arama Uzayı Boyutu", value=f"4^{num_days}", help="Tüm olası çizelgeler arasından tek yönlü sezgiyle seçim yapılır.")

    st.divider()

    run_btn = st.button("🚀 Greedy Simülasyonu Çalıştır", type="primary", width="stretch", key="btn_run_t4")

    if not run_btn and "res_t4" not in st.session_state:
        st.warning("👈 Simülasyonu başlatmak için yukarıdaki **'🚀 Greedy Simülasyonu Çalıştır'** butonuna basınız.")
        return

    if run_btn:
        with st.spinner("⏳ Greedy Vardiya Simülasyonu Çalıştırılıyor..."):
            schedule_matrix, worker_list, penalty_dict, total_score, hard_viols_count, hard_logs, req_details = run_greedy_algorithm(
                num_workers, num_days, req_day, req_eve, req_night, penalty_weights_dict, solver_mode=solver_mode, custom_workers=custom_workers
            )
            st.session_state["res_t4"] = (schedule_matrix, worker_list, penalty_dict, total_score, hard_viols_count, hard_logs, req_details)
    else:
        schedule_matrix, worker_list, penalty_dict, total_score, hard_viols_count, hard_logs, req_details = st.session_state["res_t4"]

    # --- METRİK KARTLARI ---
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam Ceza Puanı</div>
        <div class="metric-value" style="color: #dc2626;">{total_score}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        status_text = "GEÇERLİ" if hard_viols_count == 0 else "❌ GEÇERSİZ"
        status_color = "#059669" if hard_viols_count == 0 else "#dc2626"
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

    # --- SERT KISIT İHLALLERİ VE UYARI PANENELİ ---
    if "Naif" in solver_mode and hard_viols_count > 0:
        st.error(f"""
        ⚠️ **KRİTİK UYARI: {hard_viols_count} Adet Sert Kısıt (Hard Constraint) İhlali Tespit Edildi!**
        """)
        
        with st.expander("📌 Sert Kısıt İhlalleri ve Vardiya Kadro Eksiklikleri Detayı", expanded=True):
            df_logs = pd.DataFrame({"İhlal Açıklaması ve Vardiya Detayı": hard_logs})
            st.dataframe(df_logs, width="stretch", hide_index=True)

    elif hard_viols_count > 0:
        st.warning(f"⚠️ **Kadro Yetersizliği:** {hard_viols_count} Adet Vardiya İhtiyacı Eksik Kaldı. (Mevcut personel sayısı veya sertifika dağılımı seçilen vardiya ihtiyaçlarını karşılamıyor).")
        with st.expander("📌 Vardiya Kadro Eksiklikleri Detayı", expanded=False):
            df_logs = pd.DataFrame({"İhlal Açıklaması ve Vardiya Detayı": hard_logs})
            st.dataframe(df_logs, width="stretch", hide_index=True)

    else:
        st.success("✅ **TEBRİKLER:** Hiçbir Sert Kısıt ihlali yaşanmadı! Vardiya kadro ihtiyaçları (%100) eksiksiz karşılandı.")

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
        st.progress(fulfillment_rate / 100.0)

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
