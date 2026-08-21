"""
================================================================================
  VIEWS/TAB15_CP_SAT.PY - SEKME 15: CONSTRAINT PROGRAMMING / CP-SAT (GOOGLE OR-TOOLS)
================================================================================
  Bu modül, Google'ın ödüllü CP-SAT (Constraint Programming - Satisfiability)
  çözücü motorunun arayüzünü, kuramsal ilkelerini, SAT/LCG mimarisini ve
  çok çekirdekli paralel analitik görselleştirmelerini sunar.
================================================================================
"""

import time
import threading
import queue
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.cp_sat_solver import solve_cp_sat
from views.common_components import (
    render_schedule_heatmap,
    render_workload_chart,
    render_penalties_chart,
    render_posta_load_chart,
    render_shift_posta_stacked_chart,
    render_schedule_matrix_table,
    render_request_details_expander,
    render_hard_constraints_status_card,
    render_evaluator_cost_badge,
    render_metaheuristic_metric_cards,
    render_standard_schedule_analytics,
    LiveStreamTracker,
    render_live_stream_summary
)


def render_tab15(params):
    """Sekme 15 içeriğini çizer: Google OR-Tools CP-SAT Kısıt Programlama Çözücüsü ve Analiz Panelleri."""
    st.markdown("## ⚡ Sekme 15: Constraint Programming / CP-SAT (Google OR-Tools)")
    st.markdown("""
    <div style="background-color: #eff6ff; border-left: 5px solid #2563eb; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; color: #1e40af;">
        📌 <b>Google CP-SAT Paradigması (Google Operations Research Tools):</b> Google tarafından geliştirilen ödüllü 
        <b>CP-SAT (Constraint Programming - Satisfiability)</b> motoru, geleneksel kısıt programlamanın zengin mantıksal modelleme gücünü, 
        modern <b>Boolean SAT (Maddesel Sağlanabilirlik)</b> donanım hızı ve <b>Lazy Clause Generation (LCG)</b> teknolojisiyle birleştirir.
        Vardiya çizelgeleme (NSP) ve karmaşık zamanlama problemlerinde dünyanın en hızlı ve en yetenekli kesin optimizasyon çözücüsü kabul edilmektedir.
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 1. TEORİK BİLGİ VE GOOGLE CP-SAT MİMARİSİ
    # ==============================================================================
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #2563eb;">
            <div class="card-title" style="color: #1d4ed8;">🧠 3 Temel CP-SAT Teknolojisi</div>
            <ul style="line-height: 1.65; font-size: 0.92rem; color: #1e293b;">
                <li><b>1. Kısıt Yayılımı (Constraint Propagation & AC-3):</b> Bir değişkene değer atandığında etki alanındaki imkansız kombinasyonlar anında filtrelenir; arama ağacının %90'ından fazlası taranmadan budanır.</li>
                <li><b>2. Tembel Madde Üretimi (Lazy Clause Generation - LCG):</b> Tamsayı kısıtlar arka planda Boolean maddelerine dönüştürülür; donanım seviyesinde bit düzeyinde SAT çıkarımı yapılır.</li>
                <li><b>3. Çatışma Güdümlü Öğrenme (CDCL - Conflict-Driven Learning):</b> Çözücü bir çıkmaz sokakla (çatışma) karşılaştığında bunu öğrenir ve arama boyunca aynı hataya bir daha asla düşmez.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #10b981;">
            <div class="card-title" style="color: #047857;">🚀 Çok Çekirdekli Paralel Portföy Arama (LNS)</div>
            <ul style="line-height: 1.65; font-size: 0.92rem; color: #1e293b;">
                <li><b>Paralel İş Parçacıkları (Multi-threading):</b> 8 veya 16 CPU çekirdeğini eşzamanlı kullanarak farklı arama stratejilerini aynı anda koşturur.</li>
                <li><b>Büyük Komşuluk Araması (LNS - Large Neighborhood Search):</b> Çekirdeklerin bir kısmı küresel alt sınır kanıtlarken, diğerleri çözümleri dondurup-yeniden çözerek yerel iyileştirmeler yapar.</li>
                <li><b>Doğrusal Gevşetme (LP Relaxation):</b> Çözücü sürekli LP gevşetmesi yaparak kanıtlanmış teorik alt sınırı (Best Bound) anlık hesaplar.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Matematiksel Formül Gösterimi (LaTeX)
    st.markdown("##### 🧮 Google CP-SAT Mantıksal ve Matematiksel Kısıt Modeli:")
    st.latex(r"\text{Karar Değişkenleri: } x_{i, t, k} \in \{0, 1\} \quad \forall i \in \{1 \dots N\}, \; t \in \{1 \dots D\}, \; k \in \{0, 1, 2, 3\}")
    st.latex(r"\text{Günde Tek Vardiya: } \sum_{k=0}^{3} x_{i, t, k} = 1 \quad \forall i, t \qquad \text{Kadro: } \sum_{i=1}^{N} x_{i, t, k} \ge R_{t, k} \quad \forall t, k \in \{1, 2, 3\}")
    st.latex(r"\text{Gece } \rightarrow \text{ Gündüz Yasağı: } x_{i, t, 3} + x_{i, t+1, 1} \le 1 \qquad \text{Haftalık İzin: } \sum_{\tau=t}^{t+6} x_{i, \tau, 0} \ge 1 \quad \forall i, t")
    st.latex(r"\min Z = \sum_{j=1}^{5} w_j \cdot \text{SoftPenalty}_j(X)")

    st.divider()

    # ==============================================================================
    # 2. ÜÇLÜ KESİN YÖNTEM KARŞILAŞTIRMASI (MILP vs CSP vs CP-SAT)
    # ==============================================================================
    st.markdown("### ⚖️ Kesin Matematiksel Yöntemler Karşılaştırma Matrisi")
    
    comp_data = {
        "Özellik": ["Model Türü", "Çözüm Garantisi", "Doğrusal Olmayan Kısıtlar", "Paralel İşleme (Multi-threading)", "Vardiya Çizelgeleme Performansı", "Durdurulduğunda Durum"],
        "Klasik CSP (Backtracking)": ["Kısıt Arama Ağacı", "Yalnızca Feasibility", "Kolay", "❌ Tek Çekirdek", "Derin problemlerde yavaşlar", "İlk geçerli çözümü verir"],
        "MILP / ILP (PuLP - CBC)": ["Doğrusal Eşitsizlikler (Ax <= b)", "🏆 Küresel Optimum (Z*)", "Zor (Ek ikili değişken gerekir)", "❌ Tek Çekirdek (Varsayılan CBC)", "Orta / Büyük boyutta dal-sınır şişer", "En iyi tamsayı çözümü verir"],
        "Google CP-SAT (OR-Tools)": ["Zengin Kısıt + Boolean SAT", "🏆 Küresel Optimum (Z*)", "✅ Çok Kolay (Doğal Boolean Mantığı)", "🚀 8-16 Çekirdek Eşzamanlı", "⚡ Dünyanın En Hızlısı (Milisaniyeler)", "En iyi tamsayı çözümü + Alt Sınır"]
    }
    st.dataframe(pd.DataFrame(comp_data), width="stretch", hide_index=True)

    st.divider()

    # ==============================================================================
    # 3. KULLANICI KONTROL PANELİ VE PARAMETRELER
    # ==============================================================================
    st.markdown("### 🎛️ Google CP-SAT Optimizasyon Parametreleri")

    p_col1, p_col2, p_col3 = st.columns([1.5, 1.5, 1.2])

    with p_col1:
        time_limit = st.slider(
            "⏱️ Maksimum Süre Sınırı (Saniye):",
            min_value=1.0,
            max_value=60.0,
            value=10.0,
            step=1.0,
            help="CP-SAT çözücüsünün çözüm aramak için kullanacağı maksimum saniye."
        )

    with p_col2:
        num_threads = st.slider(
            "🧵 Paralel CPU Arama İş Parçacığı (Threads):",
            min_value=1,
            max_value=16,
            value=8,
            step=1,
            help="Eşzamanlı çalıştırılacak CPU çekirdeği sayısı. Ne kadar yüksekse arama o kadar paralel yürütülür."
        )

    with p_col3:
        live_stream = st.checkbox(
            "📡 Canlı Arama Akışı",
            value=True,
            help="Çözücünün bulduğu her ara çözümün canlı olarak ekranda güncellenmesini sağlar."
        )

    btn_run = st.button("⚡ Google CP-SAT Optimizasyonunu Başlat", type="primary", width="stretch")

    live_placeholder = st.empty()

    if btn_run:
        import threading
        import queue

        # 1. ÖNCE TEORİK ALT SINIR (LP RELAXATION) ARAMASI VE GÜNCELLEMESİ YAP
        from algorithms.ilp_pulp_solver import compute_lp_relaxation_bound
        active_workers = params.get('custom_workers') or params.get('workers')

        with live_placeholder.container():
            st.markdown("#### ⏳ 1. Aşama: Matematiksel Teorik Alt Sınır (LP Relaxation) Hesaplanıyor...")
            st.info("🔍 Tamsayılık şartları gevşetiliyor, kısıt ağları taranıyor ve analitik alt sınır belirleniyor...")

        theo_lb_pre, theo_breakdown_pre, _ = compute_lp_relaxation_bound(
            n_workers=params['n_workers'],
            n_days=params['n_days'],
            r_day=params['r_day'],
            r_eve=params['r_eve'],
            r_night=params['r_night'],
            weights=params['weights'],
            workers=active_workers,
            return_breakdown=True
        )

        sol_queue = queue.Queue()
        def queue_cb(step, best_score, cur_score, msg):
            sol_queue.put({
                "step": step,
                "best_score": best_score,
                "msg": msg,
                "time": time.time()
            })

        result_container = []
        def solve_worker():
            res = solve_cp_sat(
                n_workers=params['n_workers'],
                n_days=params['n_days'],
                r_day=params['r_day'],
                r_eve=params['r_eve'],
                r_night=params['r_night'],
                weights=params['weights'],
                time_limit=time_limit,
                num_threads=num_threads,
                custom_workers=active_workers,
                callback=queue_cb if live_stream else None,
                stream_interval=1
            )
            result_container.append(res)

        t_start = time.time()
        thread = threading.Thread(target=solve_worker)
        thread.start()

        best_score_live = None
        sol_count_live = 0
        history_x = []
        history_y = []
        initial_score_live = None

        if live_stream:
            while thread.is_alive():
                time.sleep(0.08)
                now = time.time()
                elapsed = max(0.01, now - t_start)
                pct = min(0.99, elapsed / max(1.0, float(time_limit)))

                # Kuyruktaki tüm yeni ara çözümleri oku
                while not sol_queue.empty():
                    item = sol_queue.get()
                    sol_count_live = item["step"]
                    best_score_live = item["best_score"]
                    if initial_score_live is None:
                        initial_score_live = best_score_live
                    history_x.append(round(item["time"] - t_start, 2))
                    history_y.append(best_score_live)

                with live_placeholder.container():
                    st.markdown("#### ⚡ Google CP-SAT Çok Çekirdekli Paralel Portföy Arama")
                    prog_text = f"🔄 %{int(pct*100)} Tamamlandı | Süre: {elapsed:.1f} sn / {time_limit:.0f} sn | 🏆 Bulunan Ara Çözüm: {sol_count_live} Adet"
                    st.progress(pct, text=prog_text)

                    c1, c2, c3, c4, c5 = st.columns(5)
                    with c1:
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Başlangıç Skoru</div>
                        <div class="metric-value" style="color: #64748b;">{initial_score_live if initial_score_live is not None else '-'}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Teorik Alt Sınır (LP)</div>
                        <div class="metric-value" style="color: #7c3aed;">{theo_lb_pre:g} Puan</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Paralel Çekirdek</div>
                        <div class="metric-value" style="color: #2563eb;">{num_threads} Threads</div>
                        </div>""", unsafe_allow_html=True)
                    with c4:
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Anlık En İyi Skor</div>
                        <div class="metric-value" style="color: #059669;">{best_score_live if best_score_live is not None else '-'} 🎯</div>
                        </div>""", unsafe_allow_html=True)
                    with c5:
                        drop_val = max(0, (initial_score_live - best_score_live)) if (initial_score_live and best_score_live) else 0
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">İyileşme</div>
                        <div class="metric-value" style="color: #d97706;">-{drop_val} Puan</div>
                        </div>""", unsafe_allow_html=True)

                    if len(history_x) > 1:
                        df_live = pd.DataFrame({"Ceza Skoru (Z)": history_y}, index=history_x)
                        st.line_chart(df_live, height=200)

        thread.join()
        results = result_container[0] if result_container else {}

        # Canlı paneli bitiş özeti olarak güncelle
        total_time = round(time.time() - t_start, 2)
        final_solutions = results.get('meta', {}).get('solutions_found', sol_count_live)
        with live_placeholder.container():
            st.markdown("#### ⚡ Google CP-SAT Optimizasyon Özeti (Arama Tamamlandı)")
            st.progress(1.0, text=f"✅ %100 Tamamlandı | Toplam Süre: {total_time:.2f} sn | 🏆 Toplam {final_solutions} Ara Çözüm Keşfedildi")

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Başlangıç Skoru</div>
                <div class="metric-value" style="color: #64748b;">{results.get('initial_score', '-')}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Teorik Alt Sınır (LP)</div>
                <div class="metric-value" style="color: #7c3aed;">{theo_lb_pre:g} Puan</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Paralel Çekirdek</div>
                <div class="metric-value" style="color: #2563eb;">{num_threads} Threads</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Nihai Skor</div>
                <div class="metric-value" style="color: #059669;">{results.get('final_score', '-')} 🎯</div>
                </div>""", unsafe_allow_html=True)
            with c5:
                drop_fin = max(0, (results.get('initial_score', 0) - results.get('final_score', 0)))
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Toplam İyileşme</div>
                <div class="metric-value" style="color: #d97706;">-{drop_fin} Puan (%{results.get('improvement_rate', 0)})</div>
                </div>""", unsafe_allow_html=True)

        results['stream_summary_data'] = {
            'total_time': total_time,
            'solutions_found': final_solutions,
            'initial_score': results.get('initial_score'),
            'final_score': results.get('final_score'),
            'num_threads': num_threads,
            'improvement_rate': results.get('improvement_rate')
        }
        st.session_state["res_t15"] = results
    else:
        has_valid_cp = (
            "res_t15" in st.session_state and 
            isinstance(st.session_state["res_t15"], dict) and 
            isinstance(st.session_state["res_t15"].get('schedule'), np.ndarray) and 
            st.session_state["res_t15"]['schedule'].shape == (params['n_workers'], params['n_days'])
        )
        if has_valid_cp:
            results = st.session_state["res_t15"]
            summary = results.get('stream_summary_data')
            if live_stream and summary:
                with live_placeholder.container():
                    st.markdown("#### ⚡ Google CP-SAT Optimizasyon Özeti (Önceki Çalıştırma)")
                    st.progress(1.0, text=f"✅ %100 Tamamlandı | Toplam Süre: {summary['total_time']:.2f} sn | 🏆 Toplam {summary['solutions_found']} Ara Çözüm Keşfedildi")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Başlangıç Skoru</div>
                        <div class="metric-value" style="color: #64748b;">{summary['initial_score']}</div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Paralel Çekirdek</div>
                        <div class="metric-value" style="color: #2563eb;">{summary['num_threads']} Threads</div>
                        </div>""", unsafe_allow_html=True)
                    with c3:
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Nihai Skor</div>
                        <div class="metric-value" style="color: #059669;">{summary['final_score']} 🎯</div>
                        </div>""", unsafe_allow_html=True)
                    with c4:
                        drop_s = max(0, (summary['initial_score'] - summary['final_score']))
                        st.markdown(f"""<div class="metric-card">
                        <div class="metric-label">Toplam İyileşme</div>
                        <div class="metric-value" style="color: #7c3aed;">-{drop_s} Puan (%{summary['improvement_rate']})</div>
                        </div>""", unsafe_allow_html=True)
        else:
            st.warning("👈 Google CP-SAT Optimizasyonunu başlatmak için yukarıdaki **'⚡ Google CP-SAT Optimizasyonunu Başlat'** butonuna basınız.")
            return

    # ==============================================================================
    # 4. SONUÇLAR VE ANALİTİK GÖSTERGELERİ
    # ==============================================================================
    meta = results.get("meta", {})

    st.divider()
    st.markdown("### 📊 Google CP-SAT Optimizasyon Sonuçları & Analitik Raporu")

    # Çözüm Durum Rozeti
    is_opt = meta.get("is_optimal", False)
    status_name = meta.get("solver_status", "UNKNOWN")
    wall_time = meta.get("wall_time", results.get("exec_time_ms", 0) / 1000)
    mip_gap = meta.get("mip_gap", 0.0)
    best_bound = meta.get("best_bound", 0)

    if is_opt:
        st.success(f"🏆 **ÇÖZÜCÜ DURUMU: OPTIMAL (Küresel Matematiksel Optimum Kanıtlandı)** | ⏱️ Süre: {wall_time:.2f} sn | Gap: %0.0 | Paralel Çekirdek: {num_threads}")
    elif results.get("is_feasible", False):
        st.info(f"⏱️ **ÇÖZÜCÜ DURUMU: {status_name} (%100 Geçerli Çözüm)** | ⏱️ Süre: {wall_time:.2f} sn | MIP Gap: %{mip_gap} | Kanıtlanan Alt Sınır: {best_bound}")
    else:
        st.error(f"❌ **ÇÖZÜCÜ DURUMU: {status_name}** | ⏱️ Süre: {wall_time:.2f} sn")

    st.info(f"📌 **Çözücünün Durma Gerekçesi:** {results['termination_reason']}")

    # Çok Çekirdekli Paralel Başlangıç Davranışı Teknik Bilgilendirme Notu
    st.markdown("""
    <div style="background-color: #f8fafc; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 12px 16px; margin: 12px 0 16px 0; font-size: 0.92rem; color: #334155; line-height: 1.5;">
        💡 <b>Teknik Not (Çok Çekirdekli Portföy ve Başlangıç Skoru Dinamiği):</b><br>
        Google CP-SAT çok iş parçacıklı (Multi-threaded Portfolio) mimarisinde eşzamanlı çalışan CPU çekirdeklerinden biri arama başladığı ilk 2-3 milisaniyede rastgele bir ham geçerli çözüm keşfeder (örneğin ~2,900 puan). Hemen ardından diğer çekirdekler sisteme verilen <b>Greedy Başlangıç Tohumunu (Warm-Start Hint)</b> devreye alarak <b>LNS (Büyük Komşuluk Araması)</b> ile arama ağacını budar. Canlı göstergedeki ilk skor ile grafiğin başlangıç noktasının çok hızlı güncellenmesi, CP-SAT'ın çok çekirdekli yarış (portfolio race) mekanizmasının doğal ve beklenen bir sonucudur.
    </div>
    """, unsafe_allow_html=True)

    # Analitik Teorik Alt Sınır (LP Relaxation) Dinamik Hesaplama
    active_workers = params.get('custom_workers') or results.get('workers')
    theo_lb_err = None
    try:
        from algorithms.ilp_pulp_solver import compute_lp_relaxation_bound
        theo_lb, theo_breakdown, _ = compute_lp_relaxation_bound(
            n_workers=params['n_workers'],
            n_days=params['n_days'],
            r_day=params['r_day'],
            r_eve=params['r_eve'],
            r_night=params['r_night'],
            weights=params['weights'],
            workers=active_workers,
            return_breakdown=True
        )
    except Exception as e:
        theo_lb_err = str(e)
        theo_lb = None

    if theo_lb_err:
        st.error(f"❌ **Teorik Alt Sınır (LP Relaxation) Hesaplanamadı:** {theo_lb_err}")

    # =========================================================================
    # CP-SAT ÖZGÜ MATEMATİKSEL METRİK KARTLARI (Tekrarsız ve Özgün)
    # =========================================================================
    # 1. Satır: Çözüm Kalitesi & İyileşme
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Başlangıç Skoru</div>
        <div class="metric-value" style="color: #64748b;">{results.get('initial_score', '-')}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Nihai Amaç Skoru (Z)</div>
        <div class="metric-value" style="color: #1d4ed8;">{results.get('final_score', '-')} 🎯</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        drop = max(0, results.get('initial_score', 0) - results.get('final_score', 0))
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam İyileşme</div>
        <div class="metric-value" style="color: #7c3aed;">-{drop} Puan (%{results.get('improvement_rate', 0)})</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        gap_clr = "#059669" if mip_gap == 0 else "#d97706"
        gap_txt = "🏆 %0.0 (Optimum)" if mip_gap == 0 else f"%{mip_gap}"
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">MIP Gap (Boşluk)</div>
        <div class="metric-value" style="color: {gap_clr};">{gap_txt}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

    # 2. Satır: Matematiksel Arama ve Çözücü İstatistikleri (5 Kolon)
    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
    m_c1.metric("🏆 Bulunan Ara Çözüm", f"{meta.get('solutions_found', 1):,} Adet")
    m_c2.metric("🎯 Kanıtlanan Alt Sınır", f"{best_bound:g} Puan", help="Google CP-SAT'ın arama ağacında imkansız dalları budayarak aşağıdan ördüğü dinamik matematiksel duvar.")
    theo_lb_txt = f"{theo_lb:g} Puan" if theo_lb is not None else "Hesaplanamadı"
    m_c3.metric("📌 Teorik Alt Sınır (2-Aşama)", theo_lb_txt, help="1. Aşama (Analitik Bölünemezlik) ve 2. Aşama (Sürekli LP) sentezinden doğan mutlak fiziksel taban.")
    m_c4.metric("🛑 Çatışma / Madde (Conflicts)", f"{meta.get('conflicts', 0):,}")
    m_c5.metric("⏱️ Hesaplama Süresi", f"{results.get('exec_time_ms', 0)} ms")

    # İki Alt Sınır Arasındaki Farkı Açıklayan Bilgilendirme Kartı
    theo_txt_card = f"{theo_lb:g} Puan" if theo_lb is not None else "Hesaplanamadı"
    st.markdown(rf"""
    <div style="background-color: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #059669; border-radius: 6px; padding: 14px 18px; margin: 10px 0 16px 0; font-size: 0.88rem; color: #166534; line-height: 1.55;">
        <b>🧱 Alt Sınır Göstergeleri Arasındaki Fark (2-Aşamalı Teorik Alt Sınır vs. CP-SAT Kanıtlanan Sınır):</b><br>
        • <b>📌 2-Aşamalı Birleşik Teorik Alt Sınır (max(Analitik, LP) = {theo_txt_card}):</b><br>
        &nbsp;&nbsp;&nbsp;&nbsp;<b>1️⃣ Aşama (Analitik Kaçınılmazlık Tabanı):</b> Gece nöbetinin işçi sayısına tam bölünememesi (Güvercin Yuvası İlkesi), usta kapasite açığı ve izin yığılmasından doğan fiziksel kaçınılmazlık tabanıdır.<br>
        &nbsp;&nbsp;&nbsp;&nbsp;<b>2️⃣ Aşama (Sürekli LP Gevşetmesi):</b> Tamsayılık şartı gevşetilerek (x ∈ [0, 1]) kısıt ağlarının sürekli uzayda Simplex ile taranmasıdır.<br>
        &nbsp;&nbsp;&nbsp;&nbsp;<i>Modelimiz bu iki aşamanın en güçlüsünü (max) alarak aşılması imkansız mutlak fiziksel tabanı belirler.</i><br>
        • <b>🎯 Kanıtlanan Alt Sınır (Best Bound = {best_bound:g} Puan):</b> Google CP-SAT'ın tamsayılı arama sürerken çelişkileri öğrenip imkansız dalları budayarak <b>aşağıdan yukarıya doğru ördüğü matematiksel duvardır</b>. Optimum kanıtlandığında (%0 Gap) yukarıdan inen tavanla (en iyi çözümle) tam çakışır.
    </div>
    """, unsafe_allow_html=True)

    # Sert Kısıt Uygunluk Kartı
    render_hard_constraints_status_card(
        is_feasible=results['is_feasible'],
        hard_violations_count=results['hard_violations_count'],
        hard_violation_logs=results['hard_violation_logs'],
        solver_name="Google CP-SAT",
        meta=results.get('meta')
    )

    # Yakınsama / Çözüm Gelişim Grafiği (Eğer birden fazla ara çözüm varsa)
    if len(results.get("score_history", [])) > 1:
        st.markdown("##### 📈 CP-SAT Çözüm Gelişim ve Yakınsama Eğrisi:")
        df_hist = pd.DataFrame({
            "Ara Çözüm #": list(range(1, len(results["score_history"]) + 1)),
            "Ceza Skoru (Z)": results["score_history"]
        })
        fig_hist = px.line(
            df_hist,
            x="Ara Çözüm #",
            y="Ceza Skoru (Z)",
            title="Google CP-SAT İyileştirme Grafiği (İteratif Ceza Düşüşü)",
            markers=True,
            color_discrete_sequence=["#2563eb"]
        )
        fig_hist.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_hist, width="stretch")

    # Standart Analitik Görselleri
    render_standard_schedule_analytics(
        results=results,
        num_days=params['n_days'],
        key_prefix="t15_cpsat",
        solver_name="Google CP-SAT",
        start_chart_num=2,
        show_request_details=True
    )
