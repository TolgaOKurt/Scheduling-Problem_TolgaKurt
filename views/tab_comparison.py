"""
================================================================================
  VIEWS/TAB_COMPARISON.PY - BÜTÜNCÜL KARŞILAŞTIRMA & BENCHMARK ANALİZİ
================================================================================
  Bu modül; Matematiksel (ILP/MILP, CSP), Sezgisel (Greedy) ve Metasezgisel
  (Hill Climbing, Simulated Annealing, Genetic Algorithm, Memetic Algorithm,
  Tabu Search) yaklaşımların teorik ve pratik karşılaştırmalarını sunar.
================================================================================
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.worker_manager import generate_worker_profiles
from algorithms.greedy_solver import run_greedy_algorithm
from algorithms.csp_backtracking_solver import CSPBacktrackingSolver
from algorithms.ilp_pulp_solver import solve_ilp_pulp, compute_lp_relaxation_bound
from algorithms.hill_climbing_solver import run_hill_climbing
from algorithms.simulated_annealing_solver import run_simulated_annealing
from algorithms.genetic_algorithm_solver import run_genetic_algorithm
from algorithms.memetic_algorithm_solver import run_memetic_algorithm
from algorithms.tabu_search_solver import run_tabu_search
from algorithms.vns_solver import run_variable_neighborhood_search

def render_tab_comparison(params):
    """Son sekme içeriğini çizer: Tüm Çözücülerin Bütüncül Karşılaştırması ve Benchmark Analizi."""
    num_workers = params['n_workers']
    num_days = params['n_days']
    req_day = params['r_day']
    req_eve = params['r_eve']
    req_night = params['r_night']
    weights = params['weights']
    custom_workers = params['custom_workers']

    # --- CANLI BENCHMARK ÇALIŞTIRMA BUTONU ---
    st.markdown("### 🚀 Canlı Benchmark & Karşılaştırma Merkezi")
    
    b_col1, b_col2 = st.columns([3, 1])
    with b_col1:
        st.info("💡 **İpucu:** Algoritmaları kendi sekmelerinde çalıştırdıysanız sonuçları doğrudan aşağıda görebilirsiniz. Dilerseniz aşağıdaki buton ile **tüm çözücüleri tek tıkla standart ayarlarda sırayla çalıştırıp** canlı olarak kıyaslayabilirsiniz.")
    with b_col2:
        run_all_btn = st.button("⚡ Tüm Çözücüleri Çalıştır (Benchmark)", type="primary", width="stretch", key="btn_run_all_bench")

    if run_all_btn:
        progress_bar = st.progress(0, text="Benchmark başlatılıyor...")
        
        # 1. Greedy 1: Sıralı / Miyopik
        progress_bar.progress(8, text="1/11: Greedy (1. Sıralı / Miyopik) çalıştırılıyor...")
        res_g1 = run_greedy_algorithm(
            num_workers, num_days, req_day, req_eve, req_night, weights,
            solver_mode="1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)", custom_workers=custom_workers
        )
        st.session_state["res_t4_myopic"] = res_g1

        # 2. Greedy 2: Kademeli / Desen Tabanlı
        progress_bar.progress(16, text="2/11: Greedy (2. Kademeli Desen) çalıştırılıyor...")
        res_g2 = run_greedy_algorithm(
            num_workers, num_days, req_day, req_eve, req_night, weights,
            solver_mode="2. Kademeli / Desen Tabanlı Yapıcı Sezgisel (Staggered Pattern-Based Greedy)", custom_workers=custom_workers
        )
        st.session_state["res_t4_staggered"] = res_g2
        st.session_state["res_t4"] = res_g2

        # 3. Greedy 3: Kısıt Öncelikli (MRV / LCV)
        progress_bar.progress(24, text="3/11: Greedy (3. Kısıt Öncelikli MRV/LCV) çalıştırılıyor...")
        res_g3 = run_greedy_algorithm(
            num_workers, num_days, req_day, req_eve, req_night, weights,
            solver_mode="3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)", custom_workers=custom_workers
        )
        st.session_state["res_t4_mrv"] = res_g3

        # 4. CSP Backtracking
        progress_bar.progress(33, text="4/11: CSP Backtracking çalıştırılıyor...")
        csp_solver = CSPBacktrackingSolver(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            max_backtracks=2000,
            custom_workers=custom_workers,
            weights=weights
        )
        res_t5 = csp_solver.solve()
        st.session_state["res_t5"] = res_t5

        # 5. ILP
        progress_bar.progress(42, text="5/11: ILP / MILP Optimizasyonu çözülüyor...")
        res_t6 = solve_ilp_pulp(num_workers, num_days, req_day, req_eve, req_night, weights, time_limit=10, custom_workers=custom_workers)
        st.session_state["res_t6"] = res_t6

        # 6. Hill Climbing
        progress_bar.progress(51, text="6/11: Hill Climbing Yerel Araması çalıştırılıyor...")
        res_t7 = run_hill_climbing(num_workers, num_days, req_day, req_eve, req_night, weights, max_iterations=3000, seed=42, custom_workers=custom_workers)
        st.session_state["res_t7"] = res_t7

        # 7. Simulated Annealing
        progress_bar.progress(60, text="7/11: Simulated Annealing Tavlama çalıştırılıyor...")
        res_t8 = run_simulated_annealing(num_workers, num_days, req_day, req_eve, req_night, weights, t_start=1000.0, t_min=0.01, cooling_rate=0.990, max_iterations=3000, seed=42, custom_workers=custom_workers)
        st.session_state["res_t8"] = res_t8

        # 8. Genetic Algorithm
        progress_bar.progress(70, text="8/11: Genetik Algoritma Popülasyonu evrimleştiriliyor...")
        res_t9 = run_genetic_algorithm(num_workers, num_days, req_day, req_eve, req_night, weights, pop_size=50, generations=80, crossover_rate=0.85, mutation_rate=0.05, elitism_count=2, seed=42, custom_workers=custom_workers)
        st.session_state["res_t9"] = res_t9

        # 9. Memetic Algorithm
        progress_bar.progress(80, text="9/11: Memetik Algoritma (GA + HC) çözülüyor...")
        res_t10 = run_memetic_algorithm(num_workers, num_days, req_day, req_eve, req_night, weights, pop_size=40, generations=60, crossover_rate=0.85, mutation_rate=0.05, local_search_depth=5, elitism_count=2, seed=42, custom_workers=custom_workers)
        st.session_state["res_t10"] = res_t10

        # 10. Tabu Search
        progress_bar.progress(90, text="10/11: Tabu Search Hafıza Tabanlı Arama çözülüyor...")
        res_t11 = run_tabu_search(num_workers, num_days, req_day, req_eve, req_night, weights, max_iterations=750, tabu_tenure=15, neighborhood_size=20, use_aspiration=True, seed=42, custom_workers=custom_workers)
        st.session_state["res_t11"] = res_t11

        # 11. Variable Neighborhood Search (VNS)
        progress_bar.progress(100, text="11/11: Variable Neighborhood Search (VNS) çözülüyor...")
        res_t12 = run_variable_neighborhood_search(num_workers, num_days, req_day, req_eve, req_night, weights, max_iterations=1000, max_neighborhoods=3, local_search_depth=15, seed=42, custom_workers=custom_workers)
        st.session_state["res_t12"] = res_t12

        st.success("✅ **Benchmark Tamamlandı:** Tüm 11 çözücü (3 Greedy + CSP + ILP + 6 Metasezgisel) aynı parametreler ve kadro üzerinde başarıyla çalıştırıldı!")

    st.divider()

    # --- SESSION STATE SONUÇLARINI TOPLAMA ---
    workers = custom_workers if (custom_workers is not None and len(custom_workers) == num_workers) else generate_worker_profiles(num_workers, num_days, randomize=False)

    # 1. Greedy Çözücüleri (3 Farklı Yaklaşım)
    res_g1 = st.session_state.get('res_t4_myopic')
    if not res_g1:
        res_g1 = run_greedy_algorithm(num_workers, num_days, req_day, req_eve, req_night, weights, solver_mode="1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)", custom_workers=workers)
        st.session_state['res_t4_myopic'] = res_g1

    res_g2 = st.session_state.get('res_t4_staggered')
    if not res_g2:
        res_g2 = run_greedy_algorithm(num_workers, num_days, req_day, req_eve, req_night, weights, solver_mode="2. Kademeli / Desen Tabanlı Yapıcı Sezgisel (Staggered Pattern-Based Greedy)", custom_workers=workers)
        st.session_state['res_t4_staggered'] = res_g2

    res_g3 = st.session_state.get('res_t4_mrv')
    if not res_g3:
        res_g3 = run_greedy_algorithm(num_workers, num_days, req_day, req_eve, req_night, weights, solver_mode="3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)", custom_workers=workers)
        st.session_state['res_t4_mrv'] = res_g3

    res5 = st.session_state.get('res_t5')
    res6 = st.session_state.get('res_t6')
    res7 = st.session_state.get('res_t7')
    res8 = st.session_state.get('res_t8')
    res9 = st.session_state.get('res_t9')
    res10 = st.session_state.get('res_t10')
    res11 = st.session_state.get('res_t11')
    res12 = st.session_state.get('res_t12')

    # --- TEORİK ALT SINIR (THEORETICAL LOWER BOUND / BEST BOUND) ---
    try:
        theo_lb, theo_breakdown, theo_reasons = compute_lp_relaxation_bound(
            num_workers, num_days, req_day, req_eve, req_night, weights, workers, return_breakdown=True
        )
    except Exception:
        theo_lb = 0.0
        theo_breakdown = {}
        theo_reasons = []

    def render_leaderboard_card(title, res_obj, bg_color, border_color, title_color, is_meta=False):
        if not res_obj:
            s_html = "<div style='font-weight:bold; font-size:1.02rem; color:#64748b; margin-top:4px;'>Çalıştırılmadı</div>"
            gap_html = ""
            time_txt = "-"
            eval_txt = "-"
        else:
            s_val = res_obj.get('final_score')
            is_feas = res_obj.get('is_feasible', True)
            h_count = res_obj.get('hard_violations_count', 0)
            meta_dict = res_obj.get('meta', {})
            
            if meta_dict.get('is_infeasible', False):
                s_html = "<div style='font-weight:bold; font-size:1.02rem; color:#dc2626; margin-top:4px;'>❌ İMKANSIZ</div>"
                gap_html = ""
            elif not is_feas:
                s_html = f"<div style='font-weight:bold; font-size:1.08rem; color:#dc2626; margin-top:4px;'>{s_val} Puan</div><div style='font-size:0.75rem; color:#dc2626; font-weight:700;'>(🚨 {h_count} İhlal)</div>"
                gap_html = ""
            elif meta_dict.get('is_optimal', False):
                s_html = f"<div style='font-weight:bold; font-size:1.08rem; color:#059669; margin-top:4px;'>{s_val} Puan</div><div style='font-size:0.75rem; color:#059669; font-weight:700;'>🏆 Optimal</div>"
                diff = round(s_val - theo_lb, 1)
                gap_pct = round((diff / max(1.0, theo_lb)) * 100, 1) if theo_lb > 0 else 0.0
                gap_html = "<div style='font-size:0.75rem; color:#059669; font-weight:700; margin-top:2px;'>🎯 Alt Sınırda (%0 Gap)</div>" if diff <= 0 else f"<div style='font-size:0.74rem; color:#047857; font-weight:600; margin-top:2px; background-color:#ecfdf5; border-radius:4px; padding:2px 4px;'>🎯 Fark: +{diff:g} (%{gap_pct} Gap)</div>"
            elif meta_dict.get('is_timeout', False):
                s_html = f"<div style='font-weight:bold; font-size:1.08rem; color:#0284c7; margin-top:4px;'>{s_val} Puan</div><div style='font-size:0.75rem; color:#0284c7; font-weight:600;'>⏱️ Zaman Sınırı</div>"
                diff = round(s_val - theo_lb, 1)
                gap_pct = round((diff / max(1.0, theo_lb)) * 100, 1) if theo_lb > 0 else 0.0
                gap_html = f"<div style='font-size:0.74rem; color:#047857; font-weight:600; margin-top:2px; background-color:#ecfdf5; border-radius:4px; padding:2px 4px;'>🎯 Fark: +{diff:g} (%{gap_pct} Gap)</div>"
            else:
                s_html = f"<div style='font-weight:bold; font-size:1.08rem; color:#1e293b; margin-top:4px;'>{s_val} Puan</div>"
                diff = round(s_val - theo_lb, 1)
                gap_pct = round((diff / max(1.0, theo_lb)) * 100, 1) if theo_lb > 0 else 0.0
                gap_html = "<div style='font-size:0.75rem; color:#059669; font-weight:700; margin-top:2px;'>🎯 Alt Sınırda (%0 Gap)</div>" if diff <= 0 else f"<div style='font-size:0.74rem; color:#047857; font-weight:600; margin-top:2px; background-color:#ecfdf5; border-radius:4px; padding:2px 4px;'>🎯 Fark: +{diff:g} (%{gap_pct} Gap)</div>"

            time_txt = f"{res_obj.get('exec_time_ms', '-')} ms"
            eval_val = res_obj.get('eval_count', '-')
            eval_txt = f"{eval_val:,} Adet" if isinstance(eval_val, (int, float)) else f"{eval_val} Adet"

        sub_info = f"<div style='font-size:0.78rem; color:#64748b; margin-top:4px;'>Süre: {time_txt}</div><div style='font-size:0.72rem; color:#475569;'>Çağrı: {eval_txt}</div>" if is_meta else f"<div style='font-size:0.82rem; color:#64748b; margin-top:4px;'>Süre: {time_txt} &bull; Çağrı: {eval_txt}</div>"
        h_tag = "h6" if is_meta else "h5"
        font_sz = "0.82rem" if is_meta else "0.92rem"

        card_html = (
            f"<div style='background-color:{bg_color}; border:1px solid {border_color}; border-radius:8px; padding:10px; text-align:center; min-height:140px; display:flex; flex-direction:column; justify-content:space-between;'>"
            f"<{h_tag} style='color:{title_color}; margin:0; font-size:{font_sz};'>{title}</{h_tag}>"
            f"<div>{s_html}{gap_html}</div>"
            f"{sub_info}"
            f"</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # --- CANLI SKOR KARTLARI (LEADERBOARD) ---
    st.markdown("### 🏆 Canlı Algoritma Skor Tablosu & Liderlik Panosu")

    # 0. HEDEF TEORİK ALT SINIR KARTI (HERO BANNER)
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border: 2px solid #059669; border-radius: 10px; padding: 14px 20px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 6px rgba(5, 150, 105, 0.08);">
        <div>
            <div style="color: #047857; font-weight: 800; font-size: 1.12rem; display: flex; align-items: center; gap: 8px;">
                🎯 <span>Matematiksel / Fiziksel Teorik Alt Sınır (Theoretical Lower Bound / <i>Z</i><sub>LB</sub>)</span>
            </div>
            <div style="color: #334155; font-size: 0.88rem; margin-top: 4px; line-height: 1.45;">
                Sürekli LP Gevşetmesi (Continuous Relaxation) ve fiziksel kaçınılmazlık kısıtlarına (Güvercin Yuvası, Usta & İzin çakışmaları) göre 
                bu problem için ulaşılabilecek <b>mutlak en düşük (aşılması imkansız) ceza puanı tabanıdır</b>.
            </div>
        </div>
        <div style="text-align: center; background-color: #ffffff; border: 2px solid #10b981; border-radius: 8px; padding: 8px 18px; min-width: 150px;">
            <div style="font-size: 0.72rem; color: #64748b; font-weight: 700; text-transform: uppercase;">HEDEF ALT TABAN</div>
            <div style="font-size: 1.55rem; font-weight: 900; color: #047857;">{theo_lb} <span style="font-size: 0.85rem; font-weight: 600;">Puan</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔍 Teorik Alt Sınır Ceza Kırılımı ve Kaçınılmazlık Nedenleri (Detay)", expanded=False):
        b_c1, b_c2 = st.columns([1, 2])
        with b_c1:
            st.markdown("##### 📌 Kısıt Bazında Alt Sınır Puanları:")
            df_lb_break = pd.DataFrame([
                {"Kısıt Tipi": k, "Kaçınılmaz Ceza (Puan)": f"{v:g} Puan"}
                for k, v in theo_breakdown.items()
            ])
            st.dataframe(df_lb_break, width="stretch", hide_index=True)
        with b_c2:
            st.markdown("##### 🧮 Matematiksel ve Fiziksel Kaçınılmazlık Nedenleri:")
            if len(theo_reasons) > 0:
                for reason in theo_reasons:
                    st.markdown(f"- {reason}")
            else:
                st.write("Tüm kısıtlar için teorik alt sınır 0 ceza puanı ile tam karşılanabilir durumdadır.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Satır: ⚡ Yapıcı Sezgiseller (Constructive / Greedy Heuristics - 3 Literatür Varyantı)
    st.markdown("##### ⚡ 1. Yapıcı Sezgiseller (Greedy Heuristics - 3 Literatür Yaklaşımı)")
    g_c1, g_c2, g_c3 = st.columns(3)
    with g_c1:
        g1_bg = "#fef2f2" if (res_g1 and not res_g1.get('is_feasible', True)) else "#f1f5f9"
        g1_bd = "#ef4444" if (res_g1 and not res_g1.get('is_feasible', True)) else "#94a3b8"
        g1_tc = "#991b1b" if (res_g1 and not res_g1.get('is_feasible', True)) else "#475569"
        render_leaderboard_card("⚡ Greedy (1. Sıralı Miyopik)", res_g1, g1_bg, g1_bd, g1_tc, is_meta=False)
    with g_c2:
        g2_bg = "#fef2f2" if (res_g2 and not res_g2.get('is_feasible', True)) else "#f1f5f9"
        g2_bd = "#ef4444" if (res_g2 and not res_g2.get('is_feasible', True)) else "#94a3b8"
        g2_tc = "#991b1b" if (res_g2 and not res_g2.get('is_feasible', True)) else "#475569"
        render_leaderboard_card("⚡ Greedy (2. Kademeli Desen)", res_g2, g2_bg, g2_bd, g2_tc, is_meta=False)
    with g_c3:
        g3_bg = "#fef2f2" if (res_g3 and not res_g3.get('is_feasible', True)) else "#f1f5f9"
        g3_bd = "#ef4444" if (res_g3 and not res_g3.get('is_feasible', True)) else "#94a3b8"
        g3_tc = "#991b1b" if (res_g3 and not res_g3.get('is_feasible', True)) else "#475569"
        render_leaderboard_card("⚡ Greedy (3. Kısıt Öncelikli MRV)", res_g3, g3_bg, g3_bd, g3_tc, is_meta=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Satır: 🏛️ Kısıt Tatmin & Matematiksel Kesin Çözücüler (Sekme 5 & 6)
    st.markdown("##### 🏛️ 2. Kısıt Tatmin & Matematiksel Kesin Çözücüler (Sekme 5 & 6)")
    c1, c2 = st.columns(2)
    with c1:
        render_leaderboard_card("🔍 Sekme 5: CSP Backtracking (Kısıt Tatmin)", res5, "#f8fafc", "#64748b", "#334155", is_meta=False)
    with c2:
        render_leaderboard_card("🎯 Sekme 6: ILP / MILP (PuLP Matematiksel Optimizasyon)", res6, "#eff6ff", "#2563eb", "#1d4ed8", is_meta=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Satır: 🧬 Metasezgisel Optimizasyon Çözücüleri (Sekme 7, 8, 9, 10, 11, 12)
    st.markdown("##### 🧬 3. Metasezgisel Optimizasyon Çözücüleri (Sekme 7, 8, 9, 10, 11, 12)")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        render_leaderboard_card("🏔️ Sekme 7: Hill Climbing", res7, "#fff7ed", "#f97316", "#c2410c", is_meta=True)
    with m2:
        render_leaderboard_card("🔥 Sekme 8: Sim. Annealing", res8, "#f0fdf4", "#16a34a", "#15803d", is_meta=True)
    with m3:
        render_leaderboard_card("🧬 Sekme 9: Genetic Algo.", res9, "#fef2f2", "#ef4444", "#b91c1c", is_meta=True)
    with m4:
        render_leaderboard_card("🏆 Sekme 10: Memetic Algo.", res10, "#eff6ff", "#2563eb", "#1d4ed8", is_meta=True)
    with m5:
        render_leaderboard_card("🤫 Sekme 11: Tabu Search", res11, "#f0f9ff", "#0284c7", "#0369a1", is_meta=True)
    with m6:
        render_leaderboard_card("🔄 Sekme 12: VNS", res12, "#f0fdfa", "#0d9488", "#0f766e", is_meta=True)

    st.divider()

    # --- KARŞILAŞTIRMALI CANLI PLOTLY GRAFİKLERİ ---
    st.markdown("### 📊 Çözücülerin Canlı Grafiksel Kıyaslama Analizi")

    solvers_data = []
    solvers_catalog = [
        ("Greedy: Sıralı Miyopik", res_g1, "Doğrudan Sezgi"),
        ("Greedy: Kademeli Desen", res_g2, "Doğrudan Sezgi"),
        ("Greedy: Kısıt Öncelikli (MRV)", res_g3, "Doğrudan Sezgi"),
        ("Sekme 5: CSP Backtracking", res5, "Kısıt Tatmin"),
        ("Sekme 6: ILP / MILP", res6, "Matematiksel"),
        ("Sekme 7: Hill Climbing", res7, "Metasezgisel"),
        ("Sekme 8: Simulated Annealing", res8, "Metasezgisel"),
        ("Sekme 9: Genetic Algorithm", res9, "Metasezgisel"),
        ("Sekme 10: Memetic Algorithm", res10, "Metasezgisel"),
        ("Sekme 11: Tabu Search", res11, "Metasezgisel"),
        ("Sekme 12: VNS", res12, "Metasezgisel"),
    ]
    for s_name, s_res, s_type in solvers_catalog:
        if s_res and s_res.get('final_score') is not None:
            f_score = s_res['final_score']
            diff_lb = round(f_score - theo_lb, 1)
            gap_pct = round((diff_lb / max(1.0, theo_lb)) * 100, 1) if theo_lb > 0 else 0.0
            solvers_data.append({
                "Algoritma": s_name,
                "Ceza Puanı": f_score,
                "Alt Sınıra Fark": diff_lb,
                "Gap (%)": gap_pct,
                "Süre (ms)": s_res.get('exec_time_ms', 0.0),
                "Ceza Çağrısı": s_res.get('eval_count', 1),
                "Tür": s_type,
                "Geçerli": s_res.get('is_feasible', True)
            })

    df_solvers = pd.DataFrame(solvers_data)

    if len(df_solvers) > 0:
        g_c1, g_c2 = st.columns(2)

        with g_c1:
            st.markdown("##### 1️⃣ Toplam Ceza Puanı Karşılaştırması & Teorik Alt Sınır (min Z - Düşük Olan İyidir)")
            fig_bar_scores = px.bar(
                df_solvers,
                x="Algoritma",
                y="Ceza Puanı",
                color="Ceza Puanı",
                color_continuous_scale="Tealgrn",
                text="Ceza Puanı",
                hover_data=["Alt Sınıra Fark", "Gap (%)", "Tür"]
            )
            # TEORİK ALT SINIR YATAY ÇİZGİSİ
            if theo_lb > 0:
                fig_bar_scores.add_hline(
                    y=theo_lb,
                    line_dash="dash",
                    line_color="#dc2626",
                    line_width=2.5,
                    annotation_text=f"🎯 Teorik Alt Sınır (Z_LB = {theo_lb} Puan)",
                    annotation_position="top left",
                    annotation_font=dict(color="#dc2626", size=11, family="sans-serif")
                )
            fig_bar_scores.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380, xaxis_tickangle=-30)
            st.plotly_chart(fig_bar_scores, width="stretch", key="bench_fig_scores")

        with g_c2:
            st.markdown("##### 2️⃣ Hesaplama Süresi Karşılaştırması (CPU ms)")
            fig_bar_times = px.bar(
                df_solvers,
                x="Algoritma",
                y="Süre (ms)",
                color="Tür",
                text="Süre (ms)",
                color_discrete_map={
                    "Doğrudan Sezgi": "#94a3b8",
                    "Kısıt Tatmin": "#64748b",
                    "Matematiksel": "#2563eb",
                    "Metasezgisel": "#059669"
                }
            )
            fig_bar_times.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380, xaxis_tickangle=-30, legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig_bar_times, width="stretch", key="bench_fig_times")

        g_c3, g_c4 = st.columns(2)

        with g_c3:
            st.markdown("##### 3️⃣ Ceza Hesaplama Çağrı Sayısı (Evaluator Calls - Logaritmik Ölçek)")
            fig_bar_evals = px.bar(
                df_solvers,
                x="Algoritma",
                y="Ceza Çağrısı",
                color="Ceza Çağrısı",
                color_continuous_scale="Purples",
                text="Ceza Çağrısı",
                log_y=True
            )
            fig_bar_evals.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380, xaxis_tickangle=-30)
            st.plotly_chart(fig_bar_evals, width="stretch", key="bench_fig_evals")

        with g_c4:
            st.markdown("##### 4️⃣ Çözücülerin Ceza Türü Dağılımı ve Kırılımı (Penalty Breakdown)")
            penalty_breakdown_data = []
            for s_name, res_obj in [
                ("Greedy: Sıralı", res_g1),
                ("Greedy: Kademeli", res_g2),
                ("Greedy: MRV/LCV", res_g3),
                ("Sekme 7: Hill Climbing", res7),
                ("Sekme 8: Sim. Annealing", res8),
                ("Sekme 9: Genetic Algo", res9),
                ("Sekme 10: Memetic Algo", res10),
                ("Sekme 11: Tabu Search", res11),
                ("Sekme 12: VNS", res12)
            ]:
                if res_obj and 'penalties' in res_obj:
                    for k_type, p_val in res_obj['penalties'].items():
                        penalty_breakdown_data.append({
                            "Algoritma": s_name,
                            "Kısıt Tipi": k_type,
                            "Ceza Puanı": p_val
                        })

            if len(penalty_breakdown_data) > 0:
                df_pbreak = pd.DataFrame(penalty_breakdown_data)
                fig_pbreak = px.bar(
                    df_pbreak,
                    x="Algoritma",
                    y="Ceza Puanı",
                    color="Kısıt Tipi",
                    barmode="stack",
                    text="Ceza Puanı"
                )
                fig_pbreak.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc", height=380, legend=dict(orientation="h", y=1.2))
                st.plotly_chart(fig_pbreak, width="stretch", key="bench_fig_pbreak")
            else:
                st.info("Çözücüler henüz çalıştırılmadı.")

        st.markdown("##### 📋 Canlı Benchmark & Teorik Alt Sınıra Göre Sapma (Optimality Gap) Tablosu")
        df_summary_table = pd.DataFrame([
            {
                "Algoritma": r["Algoritma"],
                "Tür": r["Tür"],
                "Elde Edilen Skor (Z)": f"{r['Ceza Puanı']} Puan",
                "Teorik Alt Sınır (Z_LB)": f"{theo_lb} Puan",
                "Alt Sınıra Fark (Δ)": f"+{r['Alt Sınıra Fark']} Puan" if r['Alt Sınıra Fark'] > 0 else "0 Puan (Alt Sınır)",
                "Optimality Gap (%)": f"%{r['Gap (%)']}",
                "Çözüm Süresi": f"{r['Süre (ms)']} ms",
                "Sert Kısıt Durumu": "✅ Geçerli" if r["Geçerli"] else "🚨 İhlal Var"
            }
            for r in solvers_data
        ])
        st.dataframe(df_summary_table, width="stretch", hide_index=True)
    else:
        st.warning("⚠️ Henüz hiçbir algoritma çalıştırılmadı. Canlı grafikleri görüntülemek için yukarıdaki **'⚡ Tüm Çözücüleri Çalıştır (Benchmark)'** butonuna basınız.")

    st.divider()

    # ==============================================================================
    # BÖLÜM 1: ALGORİTMA TAKSONOMİSİ VE 3 TEMEL PARADİGMANIN KARŞILAŞTIRMASI
    # ==============================================================================
    st.markdown("### 🏛️ Bölüm 1: Çözücü Sınıflandırması & 3 Temel Paradigmanın Bütüncül Karşılaştırması")
    st.markdown("""
    Vardiya Çizelgeleme Problemi (NSP / Hard NP-Complete) literatüründe kullanılan çözücüler, arama uzayını tarama ve matematiksel kesinlik modellerine göre **3 temel paradigma** altında toplanır:
    """)

    # 3 Temel Sınıfın Görsel Kartları
    k_col1, k_col2, k_col3 = st.columns(3)

    with k_col1:
        st.markdown("""
        <div style="background-color: #eff6ff; border: 2px solid #2563eb; border-radius: 10px; padding: 15px; height: 100%;">
        <h4 style="color: #1d4ed8; margin-top: 0;">🏛️ 1. Matematiksel & Kesin (Exact)</h4>
        <p style="font-size: 0.88rem; color: #1e293b;"><b>Dahil Olan Yöntemler:</b></p>
        <ul style="font-size: 0.85rem; color: #334155; padding-left: 18px;">
            <li><b>ILP / MILP (Sekme 6):</b> Tam matematiksel formülasyon, Dal-Sınır (Branch & Bound) ile %100 küresel optimum (MIP Gap %0).</li>
            <li><b>CSP Backtracking (Sekme 5):</b> Kısıt tatmin çerçevesi, arama ağacı budama (Pruning) ve geri izleme.</li>
        </ul>
        <div style="font-size: 0.82rem; background-color: #dbeafe; padding: 6px 10px; border-radius: 6px; color: #1e40af; font-weight: bold;">
            🏆 Güçlü Yönü: %100 Matematiksel Garanti
        </div>
        </div>
        """, unsafe_allow_html=True)

    with k_col2:
        st.markdown("""
        <div style="background-color: #f1f5f9; border: 2px solid #64748b; border-radius: 10px; padding: 15px; height: 100%;">
        <h4 style="color: #334155; margin-top: 0;">⚡ 2. Sezgisel (Heuristics)</h4>
        <p style="font-size: 0.88rem; color: #1e293b;"><b>Dahil Olan Yöntemler:</b></p>
        <ul style="font-size: 0.85rem; color: #334155; padding-left: 18px;">
            <li><b>Greedy / Açgözlü Simülasyon (Sekme 4):</b> Adım adım yerel en iyiyi seçen, geriye dönmeyen kural tabanlı atama.</li>
            <li><b>Kademeli Kural Sezgiselleri:</b> Günlük vardiya yayılımı, izin dengeleme kuralları.</li>
        </ul>
        <div style="font-size: 0.82rem; background-color: #e2e8f0; padding: 6px 10px; border-radius: 6px; color: #334155; font-weight: bold;">
            ⚡ Güçlü Yönü: Anlık Hız (~5 ms) & Basitlik
        </div>
        </div>
        """, unsafe_allow_html=True)

    with k_col3:
        st.markdown("""
        <div style="background-color: #f0fdf4; border: 2px solid #059669; border-radius: 10px; padding: 15px; height: 100%;">
        <h4 style="color: #047857; margin-top: 0;">🧬 3. Metasezgisel (Metaheuristics)</h4>
        <p style="font-size: 0.88rem; color: #1e293b;"><b>Dahil Olan Yöntemler:</b></p>
        <ul style="font-size: 0.85rem; color: #334155; padding-left: 18px;">
            <li><b>Tek Noktalı (Yörünge):</b> Hill Climbing (Sekme 7), Simulated Annealing (Sekme 8), Tabu Search (Sekme 11).</li>
            <li><b>Popülasyon & Hibrit:</b> Genetic Algorithm (Sekme 9), Memetic Algorithm (Sekme 10).</li>
        </ul>
        <div style="font-size: 0.82rem; background-color: #dcfce7; padding: 6px 10px; border-radius: 6px; color: #166534; font-weight: bold;">
            🌟 Güçlü Yönü: Büyük Tesislerde Üstün Çözüm
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3 Temel Paradigma Makro Karşılaştırma Tablosu
    st.markdown("##### 📊 3 Temel Yaklaşımın (Matematiksel vs. Sezgisel vs. Metasezgisel) Makro Karşılaştırma Matrisi")
    df_paradigm = pd.DataFrame({
        "Karşılaştırma Boyutu": [
            "Matematiksel Optimum Güvencesi",
            "Hesaplama Hızı & Karmaşıklık",
            "Ölçeklenebilirlik (Büyük N ve D)",
            "Yerel Tuzaklardan (Local Optima) Kaçış",
            "Sert Kısıt (Feasibility) Sağlama",
            "Arama Uzayı Stratejisi",
            "Hafıza & Öğrenme Yeteneği"
        ],
        "🏛️ Matematiksel / Kesin (ILP / CSP)": [
            "🏆 %100 Küresel Optimum Garantisi (MIP Gap %0)",
            "⏱️ Boyuta Duyarlı / Üstel Artış (ILP: 500 ms - 15+ sn, CSP: 10 - 300 ms)",
            "🔴 Zayıf (N > 40 personelde Dal-Sınır kombinatoryal patlama ve Timeout)",
            "❌ İhtiyaç Yok (Kesin sınır budaması yapar)",
            "🛡️ %100 Matematiksel Feasibility Kanıtı",
            "Arama ağacı taraması ve Simplex gevşetmesi",
            "Dal-sınır ağaç düğümleri belleği"
        ],
        "⚡ Sezgisel / Açgözlü (Greedy)": [
            "⚠️ Garanti Yok (Miyop / Local Myopic karar)",
            "⚡ Yıldırım Hızında / Anlık (~1 - 10 ms, O(N·D) Polinomial)",
            "🟢 Mükemmel (Binlerce çalışanda dahi anlık dönüş)",
            "❌ Kaçamaz (İlk bulduğu yola sapar, geriye dönemez)",
            "⚠️ Hafta sonuna doğru kadro krizine girebilir",
            "Adım adım doğrudan inşa (Constructive)",
            "❌ Yok (Hafızasız, kural tabanlı)"
        ],
        "🧬 Metasezgisel (HC / SA / GA / MA / TS)": [
            "🌟 Yüksek Kaliteli Optimuma Çok Yakın Çözüm (%98-99)",
            "🚀 Hızlı & Öngörülebilir (~50 ms - 8,000 ms, O(K·Komşuluk))",
            "🟢 Mükemmel (Devasa endüstriyel tesislerde timeout olmadan kesintisiz)",
            "🏆 Üstün (Metropolis, Çaprazlama ve Tabu Hafızası ile kaçar)",
            "🛡️ Sert Kısıt Korumalı Akıllı Takas Operatörleri",
            "Komşuluk araştırması, vadi aşımı ve genetik evrim",
            "Popülasyon gen havuzu veya Tabu Listesi hafızası"
        ]
    })
    st.dataframe(df_paradigm, width="stretch", hide_index=True)

    # --- TABLO DIŞINA ALINAN EN İDEAL KULLANIM ALANLARI KARTLARI ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 🎯 3 Temel Yaklaşım İçin En İdeal Kullanım Alanları ve Karar Kılavuzu")

    u_col1, u_col2, u_col3 = st.columns(3)

    with u_col1:
        st.markdown("""
        <div style="background-color: #eff6ff; border-left: 4px solid #2563eb; padding: 14px; border-radius: 8px; height: 100%;">
        <div style="font-weight: bold; color: #1d4ed8; font-size: 0.95rem; margin-bottom: 6px;">🏛️ Matematiksel / Kesin (ILP & CSP)</div>
        <p style="font-size: 0.88rem; color: #334155; line-height: 1.5; margin: 0;">
        <b>En İdeal Kullanım Alanı:</b> Küçük ve orta ölçekli kadrolarda (<i>N</i> &le; 30), yasal denetimlerde ve yönetim kuruluna <i>"Matematiksel olarak bundan daha az ceza puanı imkansızdır"</i> diyebilmek için %100 kanıtlanmış küresel optimum arandığında.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with u_col2:
        st.markdown("""
        <div style="background-color: #f1f5f9; border-left: 4px solid #64748b; padding: 14px; border-radius: 8px; height: 100%;">
        <div style="font-weight: bold; color: #334155; font-size: 0.95rem; margin-bottom: 6px;">⚡ Sezgisel / Açgözlü (Greedy)</div>
        <p style="font-size: 0.88rem; color: #334155; line-height: 1.5; margin: 0;">
        <b>En İdeal Kullanım Alanı:</b> Saliseler içinde (~5 ms) anlık taslak çizelge üretmek veya Metasezgisel algoritmalara kaliteli bir başlangıç tohumu (initial seed) sağlamak istendiğinde.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with u_col3:
        st.markdown("""
        <div style="background-color: #f0fdf4; border-left: 4px solid #059669; padding: 14px; border-radius: 8px; height: 100%;">
        <div style="font-weight: bold; color: #047857; font-size: 0.95rem; margin-bottom: 6px;">🧬 Metasezgisel (HC, SA, GA, MA, TS)</div>
        <p style="font-size: 0.88rem; color: #334155; line-height: 1.5; margin: 0;">
        <b>En İdeal Kullanım Alanı:</b> Yüzlerce personelin ve haftaların olduğu dev ağır sanayi tesislerinde, ILP'nin kombinatoryal patlama ile zaman aşımına girdiği durumlarda saniyeler içinde %98-99 kalitede dengeli ve çalışan memnuniyeti yüksek çizelgeler üretmek için.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # BÖLÜM 2: SEZGİSEL (HEURISTIC) YÖNTEMLERİN KENDİ ARASINDA KARŞILAŞTIRILMASI
    # ==============================================================================
    st.markdown("### ⚡ Bölüm 2: Sezgisel (Heuristic) Yöntemlerin Kendi Arasında Karşılaştırması")
    st.markdown("""
    Sezgisel yaklaşımlar (Sekme 4); arama uzayında geriye dönük arama yapmadan, belirlenen **işletme kurallarına ve öncelik sıralamalarına** göre boş çizelgeyi adım adım dolduran (inşa eden) yöntemlerdir.
    """)

    df_heuristics = pd.DataFrame({
        "Sezgisel Strateji": [
            "Standart Sıralı Greedy (Sequential Greedy)",
            "Akıllı Kademeli Greedy (Shift-Spread Heuristic)",
            "Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)"
        ],
        "Çalışma Mantığı & Atama Prensibi": [
            "İşçileri listedeki sırasına göre 1. günden itibaren doldurur; izin taleplerini doğrudan verir.",
            "İzinleri tüm haftaya dengeli yayar, vardiya kotalarını aşamalı doldurur.",
            "En çok kısıtlanmış işçiden (MYK belgeli / Usta) başlar, sonra diğerlerini atar."
        ],
        "Hafta Sonu Tıkanma Riski": [
            "🔴 Çok Yüksek (İzinler birikir, son günlerde vardiyalar eksik kalır)",
            "🟡 Düşük / Dengeli (İzinler önceden dengelendiği için kriz önlenir)",
            "🟢 Çok Düşük (Kritik roller önceden güvenceye alınır)"
        ],
        "Posta Bütünlüğü Koruma": [
            "🔴 Zayıf (Kişi bazlı atama postaları dağıtır)",
            "🟡 Orta (Posta önceliği gözetilir)",
            "🟢 İyi (Posta blokları halinde doldurma yapılır)"
        ],
        "Hesaplama Hızı": [
            "⚡ ~1 ms",
            "⚡ ~5 ms",
            "⚡ ~10 ms"
        ],
        "Genel Değerlendirme": [
            "Sadece temel kurguyu test etmek için uygundur.",
            "Sanayide anlık taslak çizelge için en pratik sezgiseldir.",
            "Sert kısıtları riske atmayan en güvenli sezgiseldir."
        ]
    })
    st.dataframe(df_heuristics, width="stretch", hide_index=True)

    h_box1, h_box2 = st.columns(2)
    with h_box1:
        st.markdown("""
        <div style="background-color: #fffbeb; border: 1px solid #f59e0b; border-radius: 8px; padding: 14px;">
        <h5 style="color: #b45309; margin-top: 0;">⚠️ Sezgisellerin Temel Zayıflığı (Miyop / Local Myopia):</h5>
        <p style="font-size: 0.88rem; color: #334155; margin-bottom: 0;">
        Açgözlü sezgiseller geriye dönük arama yapmaz. Pazartesi ve Salı günleri tüm izin taleplerine "EVET" dediklerinde, Pazar günü kanuni dinlenme kuralı yüzünden çalışacak işçi kalmaz. Bu durum <b>"Miyop Karar Hatası"</b> olarak adlandırılır.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with h_box2:
        st.markdown("""
        <div style="background-color: #f0fdf4; border: 1px solid #16a34a; border-radius: 8px; padding: 14px;">
        <h5 style="color: #15803d; margin-top: 0;">💡 Sezgisellerin En Büyük Gücü (Tohumlama / Seeding):</h5>
        <p style="font-size: 0.88rem; color: #334155; margin-bottom: 0;">
        Sezgisel yöntemlerin ürettiği taslak çizelge, <b>Metasezgisel çözücülere (Hill Climbing, Tabu Search, GA) başlangıç tohumu (initial seed)</b> olarak verildiğinde, arama süresi %80 oranında kısalır ve çok daha hızlı küresel optimuma ulaşılır.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # BÖLÜM 3: METASEZGİSEL (METAHEURISTIC) YÖNTEMLERİN KENDİ ARASINDA KARŞILAŞTIRILMASI
    # ==============================================================================
    st.markdown("### 🧬 Bölüm 3: Metasezgisel (Metaheuristic) Yöntemlerin Kendi Arasında Karşılaştırması (6-Yönlü Derin Matris)")
    st.markdown("""
    Metasezgisel yöntemler (Sekme 7, 8, 9, 10, 11, 12); arama uzayındaki **yerel minimum (Local Optimum) çukurlarından kurtulmak** ve küresel en iyiye ulaşmak için farklı zeka mekanizmaları (termodinamik, genetik evrim, insan hafızası, hiyerarşik komşuluk değişimi) kullanır.
    """)

    comp_html = """
    <div style="overflow-x: auto;">
    <table style="width:100%; border-collapse:collapse; margin-top:10px; font-size:0.85rem; background-color:#ffffff; border:1px solid #cbd5e1; border-radius:8px; overflow:hidden;">
        <thead>
            <tr style="background-color:#0f172a; color:#ffffff; text-align:left;">
                <th style="padding:10px 12px; width:14%;">Karşılaştırma Kriteri</th>
                <th style="padding:10px 12px; width:14%; color:#fdba74;">🏔️ Sekme 7: Hill Climbing</th>
                <th style="padding:10px 12px; width:14%; color:#86efac;">🔥 Sekme 8: Sim. Annealing</th>
                <th style="padding:10px 12px; width:14%; color:#fca5a5;">🧬 Sekme 9: Genetic Algo.</th>
                <th style="padding:10px 12px; width:15%; color:#93c5fd;">🏆 Sekme 10: Memetic Algo.</th>
                <th style="padding:10px 12px; width:14%; color:#7dd3fc;">🤫 Sekme 11: Tabu Search</th>
                <th style="padding:10px 12px; width:15%; color:#2dd4bf;">🔄 Sekme 12: VNS</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:9px 12px; font-weight:bold; color:#334155;">Arama Paradigması</td>
                <td style="padding:9px 12px;">Tek Noktalı Yöresel Arama</td>
                <td style="padding:9px 12px;">Stokastik Termodinamik Kabul</td>
                <td style="padding:9px 12px;">Popülasyon Bazlı Evrimsel</td>
                <td style="padding:9px 12px; font-weight:bold; color:#1d4ed8;">Hibrit: Global GA + Lokal HC</td>
                <td style="padding:9px 12px; color:#0369a1; font-weight:bold;">Hafıza Tabanlı Deterministik</td>
                <td style="padding:9px 12px; color:#0f766e; font-weight:bold;">Hiyerarşik Çoklu Komşuluk (N1→N2→N3)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:9px 12px; font-weight:bold; color:#334155;">Hafıza Mekanizması</td>
                <td style="padding:9px 12px; color:#dc2626;">❌ Yok (Hafızasız)</td>
                <td style="padding:9px 12px; color:#dc2626;">❌ Yok (Yalnızca Sıcaklık T)</td>
                <td style="padding:9px 12px;">Popülasyon Gen Havuzu</td>
                <td style="padding:9px 12px;">Popülasyon Gen Havuzu</td>
                <td style="padding:9px 12px; color:#059669; font-weight:bold;">✅ Var (Kısa Vadeli Tabu Listesi)</td>
                <td style="padding:9px 12px; color:#0d9488;">Komşuluk İndeksi Kademesi (k=1,2,3)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:9px 12px; font-weight:bold; color:#334155;">Çevrim / Döngü Engelleme</td>
                <td style="padding:9px 12px; color:#dc2626;">Zayıf (Döngüye girebilir)</td>
                <td style="padding:9px 12px;">Kısmi (Sıcaklık sıçraması)</td>
                <td style="padding:9px 12px;">İyi (Mutasyon)</td>
                <td style="padding:9px 12px; color:#15803d;">Çok İyi (Çeşitlilik koruma)</td>
                <td style="padding:9px 12px; color:#059669; font-weight:bold;">🏆 Mükemmel (Tabu Tenure kilidi)</td>
                <td style="padding:9px 12px; color:#059669; font-weight:bold;">🏆 Çok Başarılı (Shaking ile uzay değiştirme)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:9px 12px; font-weight:bold; color:#334155;">Yerel Tuzaktan Kaçış</td>
                <td style="padding:9px 12px; color:#c2410c;">Zayıf (Tepede kilitlenir)</td>
                <td style="padding:9px 12px; color:#15803d;">İyi (Metropolis kabulü)</td>
                <td style="padding:9px 12px; color:#1d4ed8;">Çok Üstün (Çaprazlama)</td>
                <td style="padding:9px 12px; color:#059669; font-weight:bold;">🏆 Mükemmel (Global + Lokal)</td>
                <td style="padding:9px 12px; color:#0369a1; font-weight:bold;">Çok Güçlü (Aspirasyon)</td>
                <td style="padding:9px 12px; color:#0f766e; font-weight:bold;">🏆 Mükemmel (N2 ve N3'e sıçrama)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:9px 12px; font-weight:bold; color:#334155;">Posta Bütünlüğü Yaklaşımı</td>
                <td style="padding:9px 12px;">Yavaş (Mikro takas)</td>
                <td style="padding:9px 12px;">Rastgele</td>
                <td style="padding:9px 12px; color:#dc2626;">Dikiş noktalarında bozulabilir</td>
                <td style="padding:9px 12px; color:#059669;">İyi (Lokal arama onarır)</td>
                <td style="padding:9px 12px;">Hızlı (Hafıza yönlendirmeli)</td>
                <td style="padding:9px 12px; color:#059669; font-weight:bold;">🏆 Üstün (N3 doğrudan postayı taşır)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:9px 12px; font-weight:bold; color:#334155;">Hesaplama Hızı</td>
                <td style="padding:9px 12px; color:#15803d; font-weight:bold;">⚡ Yıldırım (~50-500 ms)</td>
                <td style="padding:9px 12px; color:#0284c7; font-weight:bold;">🚀 Çok Hızlı (~200-2,000 ms)</td>
                <td style="padding:9px 12px; color:#d97706;">⏱️ Orta (~2-8 sn | Popülasyon)</td>
                <td style="padding:9px 12px; color:#2563eb;">⏱️ Dengeli (~3-12 sn)</td>
                <td style="padding:9px 12px; color:#059669; font-weight:bold;">🚀 Çok Hızlı (~300-3,000 ms)</td>
                <td style="padding:9px 12px; color:#0f766e; font-weight:bold;">⚡ Yıldırım Hızında (~200-1,500 ms)</td>
            </tr>
        </tbody>
    </table>
    </div>
    """
    st.markdown(comp_html, unsafe_allow_html=True)
