"""
================================================================================
  VIEWS/TAB_COMPARISON.PY - BÜTÜNCÜL KARŞILAŞTIRMA & BENCHMARK ANALİZİ
================================================================================
  Bu modül; Matematiksel , Sezgisel  ve Metasezgisel
  yaklaşımların teorik ve pratik karşılaştırmalarını sunar.
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
from algorithms.pso_solver import run_particle_swarm_optimization
from algorithms.aco_solver import run_ant_colony_optimization
from algorithms.cp_sat_solver import solve_cp_sat

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
        progress_bar.progress(7, text="1/13: Greedy (1. Sıralı / Miyopik) çalıştırılıyor...")
        res_g1 = run_greedy_algorithm(
            num_workers, num_days, req_day, req_eve, req_night, weights,
            solver_mode="1. Sıralı / Miyopik Açgözlü Sezgisel (Sequential Myopic Greedy)", custom_workers=custom_workers
        )
        st.session_state["res_t4_myopic"] = res_g1

        # 2. Greedy 2: Kademeli / Desen Tabanlı
        progress_bar.progress(15, text="2/13: Greedy (2. Kademeli Desen) çalıştırılıyor...")
        res_g2 = run_greedy_algorithm(
            num_workers, num_days, req_day, req_eve, req_night, weights,
            solver_mode="2. Kademeli / Desen Tabanlı Yapıcı Sezgisel (Staggered Pattern-Based Greedy)", custom_workers=custom_workers
        )
        st.session_state["res_t4_staggered"] = res_g2
        st.session_state["res_t4"] = res_g2

        # 3. Greedy 3: Kısıt Öncelikli (MRV / LCV)
        progress_bar.progress(23, text="3/13: Greedy (3. Kısıt Öncelikli MRV/LCV) çalıştırılıyor...")
        res_g3 = run_greedy_algorithm(
            num_workers, num_days, req_day, req_eve, req_night, weights,
            solver_mode="3. Kısıt Öncelikli Sezgisel (MRV / LCV Tabanlı Heuristic)", custom_workers=custom_workers
        )
        st.session_state["res_t4_mrv"] = res_g3

        # 4. CSP Backtracking
        progress_bar.progress(31, text="4/13: CSP Backtracking çalıştırılıyor...")
        csp_mb = max(st.session_state.get('csp_mb', 3000), 10000)
        csp_solver = CSPBacktrackingSolver(
            n_workers=num_workers,
            n_days=num_days,
            r_day=req_day,
            r_eve=req_eve,
            r_night=req_night,
            max_backtracks=csp_mb,
            custom_workers=custom_workers,
            weights=weights
        )
        res_t5 = csp_solver.solve()
        st.session_state["res_t5"] = res_t5

        # 5. ILP
        progress_bar.progress(38, text="5/13: ILP / MILP Optimizasyonu çözülüyor...")
        res_t6 = solve_ilp_pulp(num_workers, num_days, req_day, req_eve, req_night, weights, time_limit=10, custom_workers=custom_workers)
        st.session_state["res_t6"] = res_t6

        # 6. Hill Climbing
        progress_bar.progress(46, text="6/13: Hill Climbing Yerel Araması çalıştırılıyor...")
        res_t7 = run_hill_climbing(num_workers, num_days, req_day, req_eve, req_night, weights, max_iterations=3000, seed=42, custom_workers=custom_workers)
        st.session_state["res_t7"] = res_t7

        # 7. Simulated Annealing
        progress_bar.progress(54, text="7/13: Simulated Annealing Tavlama çalıştırılıyor...")
        res_t8 = run_simulated_annealing(num_workers, num_days, req_day, req_eve, req_night, weights, t_start=1000.0, t_min=0.01, cooling_rate=0.990, max_iterations=3000, seed=42, custom_workers=custom_workers)
        st.session_state["res_t8"] = res_t8

        # 8. Genetic Algorithm
        progress_bar.progress(62, text="8/13: Genetik Algoritma Popülasyonu evrimleştiriliyor...")
        res_t9 = run_genetic_algorithm(num_workers, num_days, req_day, req_eve, req_night, weights, pop_size=50, generations=80, crossover_rate=0.85, mutation_rate=0.05, elitism_count=2, seed=42, custom_workers=custom_workers)
        st.session_state["res_t9"] = res_t9

        # 9. Memetic Algorithm
        progress_bar.progress(70, text="9/13: Memetik Algoritma (GA + HC) çözülüyor...")
        res_t10 = run_memetic_algorithm(num_workers, num_days, req_day, req_eve, req_night, weights, pop_size=40, generations=60, crossover_rate=0.85, mutation_rate=0.05, local_search_depth=5, elitism_count=2, seed=42, custom_workers=custom_workers)
        st.session_state["res_t10"] = res_t10

        # 10. Tabu Search
        progress_bar.progress(78, text="10/13: Tabu Search Hafıza Tabanlı Arama çözülüyor...")
        res_t11 = run_tabu_search(num_workers, num_days, req_day, req_eve, req_night, weights, max_iterations=750, tabu_tenure=15, neighborhood_size=20, use_aspiration=True, seed=42, custom_workers=custom_workers)
        st.session_state["res_t11"] = res_t11

        # 11. Variable Neighborhood Search (VNS)
        progress_bar.progress(85, text="11/13: Variable Neighborhood Search (VNS) çözülüyor...")
        res_t12 = run_variable_neighborhood_search(num_workers, num_days, req_day, req_eve, req_night, weights, max_iterations=1000, max_neighborhoods=3, local_search_depth=15, seed=42, custom_workers=custom_workers)
        st.session_state["res_t12"] = res_t12

        # 12. Particle Swarm Optimization (Discrete PSO)
        progress_bar.progress(92, text="12/13: Particle Swarm Optimization (Discrete PSO) çözülüyor...")
        res_t13 = run_particle_swarm_optimization(num_workers, num_days, req_day, req_eve, req_night, weights, swarm_size=30, max_iterations=100, w_inertia=0.72, c1_cognitive=1.49, c2_social=1.49, seed=42, custom_workers=custom_workers)
        st.session_state["res_t13"] = res_t13

        # 13. Ant Colony Optimization (ACO)
        progress_bar.progress(93, text="13/14: Ant Colony Optimization (ACO) çözülüyor...")
        res_t14 = run_ant_colony_optimization(num_workers, num_days, req_day, req_eve, req_night, weights, n_ants=20, max_iterations=60, evaporation_rate=0.15, alpha=1.0, beta=2.0, seed=42, custom_workers=custom_workers)
        st.session_state["res_t14"] = res_t14

        # 14. Google CP-SAT (Constraint Programming)
        progress_bar.progress(100, text="14/14: Google CP-SAT Optimizasyonu çözülüyor...")
        res_t15 = solve_cp_sat(num_workers, num_days, req_day, req_eve, req_night, weights, time_limit=10.0, num_threads=8, custom_workers=custom_workers)
        st.session_state["res_t15"] = res_t15

        st.success("✅ **Benchmark Tamamlandı:** Tüm 14 çözücü (3 Greedy + CSP + ILP + 8 Metasezgisel + Google CP-SAT) aynı parametreler ve kadro üzerinde başarıyla çalıştırıldı!")

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
    res13 = st.session_state.get('res_t13')
    res14 = st.session_state.get('res_t14')
    res15 = st.session_state.get('res_t15')

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

    # 2. Satır: 🏛️ Kısıt Tatmin & Matematiksel Kesin Çözücüler (Sekme 5, 6 & 15)
    st.markdown("##### 🏛️ 2. Kısıt Tatmin & Matematiksel Kesin Çözücüler (Sekme 5, 6 & 15)")
    c1, c2, c3 = st.columns(3)
    with c1:
        render_leaderboard_card("🔍 Sekme 5: CSP Backtracking", res5, "#f8fafc", "#64748b", "#334155", is_meta=False)
    with c2:
        render_leaderboard_card("🎯 Sekme 6: ILP / MILP (PuLP CBC)", res6, "#eff6ff", "#2563eb", "#1d4ed8", is_meta=False)
    with c3:
        render_leaderboard_card("⚡ Sekme 15: Google CP-SAT (OR-Tools)", res15, "#eff6ff", "#3b82f6", "#1e40af", is_meta=False)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Satır: 🧬 Metasezgisel Optimizasyon Çözücüleri (Sekme 7 - 14)
    st.markdown("##### 🧬 3. Metasezgisel Optimizasyon Çözücüleri (Sekme 7, 8, 9, 10, 11, 12, 13, 14)")
    
    # 1. Metasezgisel Alt Satırı (4 Kolon: HC, SA, GA, MA)
    m1_1, m1_2, m1_3, m1_4 = st.columns(4)
    with m1_1:
        render_leaderboard_card("🏔️ Sekme 7: Hill Climbing", res7, "#fff7ed", "#f97316", "#c2410c", is_meta=True)
    with m1_2:
        render_leaderboard_card("🔥 Sekme 8: Sim. Annealing", res8, "#f0fdf4", "#16a34a", "#15803d", is_meta=True)
    with m1_3:
        render_leaderboard_card("🧬 Sekme 9: Genetic Algo.", res9, "#fef2f2", "#ef4444", "#b91c1c", is_meta=True)
    with m1_4:
        render_leaderboard_card("🏆 Sekme 10: Memetic Algo.", res10, "#eff6ff", "#2563eb", "#1d4ed8", is_meta=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 2. Metasezgisel Alt Satırı (4 Kolon: TS, VNS, PSO, ACO)
    m2_1, m2_2, m2_3, m2_4 = st.columns(4)
    with m2_1:
        render_leaderboard_card("🤫 Sekme 11: Tabu Search", res11, "#f0f9ff", "#0284c7", "#0369a1", is_meta=True)
    with m2_2:
        render_leaderboard_card("🔄 Sekme 12: VNS", res12, "#f0fdfa", "#0d9488", "#0f766e", is_meta=True)
    with m2_3:
        render_leaderboard_card("🐝 Sekme 13: Discrete PSO", res13, "#fefce8", "#ca8a04", "#a16207", is_meta=True)
    with m2_4:
        render_leaderboard_card("🐜 Sekme 14: Ant Colony (ACO)", res14, "#fffbeb", "#d97706", "#b45309", is_meta=True)

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
        ("Sekme 13: Discrete PSO", res13, "Metasezgisel"),
        ("Sekme 14: Ant Colony (ACO)", res14, "Metasezgisel"),
        ("Sekme 15: Google CP-SAT", res15, "Matematiksel"),
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
                ("Sekme 12: VNS", res12),
                ("Sekme 13: Discrete PSO", res13),
                ("Sekme 14: Ant Colony (ACO)", res14),
                ("Sekme 15: Google CP-SAT", res15)
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
            <li><b>ILP / MILP (Sekme 6):</b> Klasik Doğrusal Programlama, Simplex ve Dal-Sınır (Branch & Bound) ile %100 küresel optimum.</li>
            <li><b>Google CP-SAT (Sekme 15):</b> Kısıt Programlama, SAT tabanlı LCG/CDCL ve çok çekirdekli LNS portföy motoru.</li>
            <li><b>CSP Backtracking (Sekme 5):</b> Kısıt tatmin çerçevesi, arama ağacı budama (Pruning) ve geri izleme.</li>
        </ul>
        <div style="font-size: 0.82rem; background-color: #dbeafe; padding: 6px 10px; border-radius: 6px; color: #1e40af; font-weight: bold;">
            🏆 Güçlü Yönü: %100 Matematiksel Garanti & Kanıtlanmış Alt Sınır
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
            <li><b>Tek Noktalı (Yörünge):</b> Hill Climbing (Sekme 7), Simulated Annealing (Sekme 8), Tabu Search (Sekme 11), VNS (Sekme 12).</li>
            <li><b>Popülasyon, Sürü & Hibrit:</b> Genetic Algorithm (Sekme 9), Memetic Algorithm (Sekme 10), PSO (Sekme 13), ACO (Sekme 14).</li>
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
        "🏛️ Matematiksel / Kesin (ILP / CP-SAT / CSP)": [
            "🏆 %100 Küresel Optimum Garantisi (MIP Gap %0)",
            "⏱️ Boyuta Duyarlı / Üstel Artış (ILP: 500 ms - 15+ sn, CP-SAT: 100 ms - 10 sn)",
            "🟢 CP-SAT Güçlü (LNS ile), ILP Orta (N > 40 personelde Dal-Sınır yavaşlaması)",
            "❌ İhtiyaç Yok (Kesin sınır budaması ve SAT maddeleri yapar)",
            "🛡️ %100 Matematiksel Feasibility Kanıtı",
            "Simplex gevşetmesi (ILP) veya SAT Çatışma Öğrenimi & LNS (CP-SAT)",
            "Dal-sınır ağacı (ILP) ve CDCL Çatışma Maddesi Hafızası (CP-SAT)"
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
        "🧬 Metasezgisel (HC / SA / GA / MA / TS / VNS / PSO / ACO)": [
            "🌟 Yüksek Kaliteli Optimuma Çok Yakın Çözüm (%98-99)",
            "🚀 Hızlı & Öngörülebilir (~50 ms - 8,000 ms, O(K·Komşuluk / Sürü))",
            "🟢 Mükemmel (Devasa endüstriyel tesislerde timeout olmadan kesintisiz)",
            "🏆 Üstün (Metropolis, Çaprazlama, Tabu Hafızası ve Sürü Zekası ile kaçar)",
            "🛡️ Sert Kısıt Korumalı Akıllı Takas Operatörleri",
            "Komşuluk araştırması, vadi aşımı, genetik evrim ve feromon/sürü uçuşu",
            "Popülasyon gen havuzu, Tabu Listesi veya Feromon İzi / Sürü Hafızası"
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
        <div style="font-weight: bold; color: #1d4ed8; font-size: 0.95rem; margin-bottom: 6px;">🏛️ Matematiksel / Kesin (ILP & CP-SAT)</div>
        <p style="font-size: 0.88rem; color: #334155; line-height: 1.5; margin: 0;">
        <b>En İdeal Kullanım Alanı:</b> Yasal denetimlerde ve yönetim kuruluna <i>"Matematiksel olarak bundan daha az ceza puanı imkansızdır"</i> diyebilmek için %100 kanıtlanmış küresel optimum veya kanıtlanmış alt sınır (Best Bound) arandığında.
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
        <div style="font-weight: bold; color: #047857; font-size: 0.95rem; margin-bottom: 6px;">🧬 Metasezgisel (HC, SA, GA, MA, TS, VNS, PSO, ACO)</div>
        <p style="font-size: 0.88rem; color: #334155; line-height: 1.5; margin: 0;">
        <b>En İdeal Kullanım Alanı:</b> Yüzlerce personelin ve ayların olduğu dev ağır sanayi tesislerinde, ILP'nin kombinatoryal patlama ile zaman aşımına girdiği durumlarda saniyeler içinde %98-99 kalitede dengeli çizelgeler üretmek için.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # BÖLÜM 1.1: MATEMATİKSEL DEVLERİN KIYASI: SEKME 6 (ILP) vs. SEKME 15 (GOOGLE CP-SAT)
    # ==============================================================================
    st.markdown("### ⚖️ Bölüm 1.1: Matematiksel Çözücülerin Kıyası: Sekme 6 (ILP / MILP) vs. Sekme 15 (Google CP-SAT)")
    st.markdown("""
    Endüstriyel optimizasyon literatüründe hem **ILP (Mixed Integer Linear Programming)** hem de **Google CP-SAT (Constraint Programming)** matematiksel kesinlik ve kanıtlanmış alt sınır (Best Bound) sunar. Ancak bu iki motorun iç mimarileri, mantıksal kısıtları ele alış biçimleri ve çok çekirdek optimizasyonları kökten farklıdır:
    """)

    # 4 Temel Ayrım Kartı
    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        st.markdown("""
        <div style="background-color: #eff6ff; border: 1.5px solid #3b82f6; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <div style="font-weight: bold; color: #1d4ed8; font-size: 0.95rem; margin-bottom: 6px;">🎯 1. Çözücü Motoru ve Temel Arama Paradigması</div>
            <p style="font-size: 0.86rem; color: #334155; line-height: 1.5; margin: 0;">
                <b>Sekme 6 (ILP / PuLP CBC):</b> Sürekli uzayda <b>LP Relaxation (Doğrusal Gevşetme)</b> ve <b>Simplex Algoritması</b> çözer. Tamsayılık için Gomory Kesme Düzlemleri ve standart Dal-Sınır (Branch-and-Bound) ağacı budar.<br><br>
                <b>Sekme 15 (Google CP-SAT):</b> Saf bir Doğrusal Programlama motoru değildir. <b>Boolean SAT (Sağlanabilirlik)</b> ile <b>Kısıt Yayılımı (Constraint Propagation)</b> motorunu birleştirir. <b>Lazy Clause Generation (LCG)</b> ve <b>CDCL (Çatışma Güdümlü Madde Öğrenimi)</b> ile arama yapar.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #f8fafc; border: 1.5px solid #64748b; border-radius: 8px; padding: 14px;">
            <div style="font-weight: bold; color: #334155; font-size: 0.95rem; margin-bottom: 6px;">🧩 2. Mantıksal ve Ayrık Kısıtların Modellenmesi</div>
            <p style="font-size: 0.86rem; color: #334155; line-height: 1.5; margin: 0;">
                <b>Sekme 6 (ILP):</b> Mantıksal kuralları (Örn: <i>"A postası bölünmesin"</i> veya <i>"Akşamdan gündüze ters dönüş olmasın"</i>) doğrudan anlayamaz. Bu kurallar için yapay ikili/sürekli değişkenler ve <b>Big-M (Büyük M Katsayısı)</b> eşitsizlikleri gerektirir. Big-M yöntemi gevşetmeyi zayıflatarak çözümü yavaşlatabilir.<br><br>
                <b>Sekme 15 (Google CP-SAT):</b> Mantıksal ifadeleri (<code>AddBoolAnd</code>, <code>AddBoolOr</code>, <code>OnlyEnforceIf</code>) doğrudan yerel Boolean önermeleri olarak anlar. Big-M yapay değişkenlerine ihtiyaç duymaz.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with comp_c2:
        st.markdown("""
        <div style="background-color: #f0fdf4; border: 1.5px solid #10b981; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <div style="font-weight: bold; color: #047857; font-size: 0.95rem; margin-bottom: 6px;">⚡ 3. Çok Çekirdek (Multi-Threading) & Portföy Paralelliği</div>
            <p style="font-size: 0.86rem; color: #334155; line-height: 1.5; margin: 0;">
                <b>Sekme 6 (ILP / CBC):</b> Varsayılan olarak tek iş parçacığında (Single Thread) seri Dal-Sınır ağacı yürütür.<br><br>
                <b>Sekme 15 (Google CP-SAT):</b> <b>8-16 CPU Çekirdeğini</b> aynı anda farklı stratejilerle (bir çekirdek LNS - Büyük Komşuluk Araması, biri rastgele arama, biri LP gevşetmesi, biri çatışma öğrenimi) eşzamanlı bir portföy yarışı olarak koşturur. Çekirdekler buldukları iyi maddeleri anlık birbirine aktarır.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #fff7ed; border: 1.5px solid #f97316; border-radius: 8px; padding: 14px;">
            <div style="font-weight: bold; color: #c2410c; font-size: 0.95rem; margin-bottom: 6px;">📈 4. Büyük Ölçekli Tesislerde Performans Farkı</div>
            <p style="font-size: 0.86rem; color: #334155; line-height: 1.5; margin: 0;">
                <b>Sekme 6 (ILP):</b> Personel sayısı (<i>N</i> &gt; 35) veya gün sayısı (<i>D</i> &gt; 14) olduğunda karar değişkeni sayısı on binleri bulur ve Simplex matrisi şişerek süre sınırında (timeout) takılabilir.<br><br>
                <b>Sekme 15 (Google CP-SAT):</b> <b>LNS (Büyük Komşuluk Araması)</b> sayesinde devasa kısıt problemlerinde saniyeler içinde mükemmel çözümlere ulaşır ve karmaşık kısıt kombinasyonlarında günümüzün altın standardıdır.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detaylı Karşılaştırma Matrisi Tablosu
    st.markdown("##### 📊 Sekme 6 (ILP / PuLP CBC) vs. Sekme 15 (Google CP-SAT) Karşılaştırma Matrisi")
    df_ilp_vs_cpsat = pd.DataFrame({
        "Özellik / Karşılaştırma Kriteri": [
            "Temel Matematiksel Paradigma",
            "Kullanılan Çözücü Motoru",
            "Mantıksal Kısıt Temsili (AND / OR / IF)",
            "Çok Çekirdek (Multi-Threading) Desteği",
            "Arama İçi Öğrenme (Learning Capability)",
            "Teorik Alt Sınır (Best Bound / Gap)",
            "Küresel Optimum Kanıtlama Hızı",
            "Büyük Kısıtlı NSP Problemlerinde Güç"
        ],
        "🎯 Sekme 6: ILP / MILP (PuLP CBC)": [
            "Karışık Tamsayılı Doğrusal Programlama (MILP)",
            "COIN-OR CBC (Coin-or branch and cut)",
            "Big-M Yöntemi ve Yapay Sürekli Değişkenler",
            "Tek Çekirdekli (Seri Dal-Sınır Ağacı)",
            "Gomory Kesme Düzlemleri (Cuts)",
            "✅ LP Relaxation tabanlı sürekli alt sınır",
            "Küçük problemlerde hızlı, büyüklerde üstel yavaşlar",
            "Saf doğrusal maliyetlerde çok iyi, mantıksal kısıtlarda zorlanır"
        ],
        "⚡ Sekme 15: Google CP-SAT (OR-Tools)": [
            "Kısıt Programlama (CP) + Boolean SAT",
            "Google CP-SAT Engine (Ödüllü SAT Motoru)",
            "Doğrudan Yerel Boolean Önermeleri (Big-M gerektirmez)",
            "✅ 8-16 Çekirdekli Çok İş Parçacıklı Portföy Yarışı",
            "CDCL (Çatışma Güdümlü Madde Öğrenimi)",
            "✅ LCG tabanlı tam ölçekli alt sınır",
            "Çok çekirdekli LNS portföyü ile belirgin derecede hızlı",
            "🏆 Vardiya, rotasyon ve karmaşık kural ağlarında dünya lideri"
        ]
    })
    st.dataframe(df_ilp_vs_cpsat, width="stretch", hide_index=True)

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
        <div style="background-color: #fffbeb; border 1px solid #f59e0b; border-radius: 8px; padding: 14px;">
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
        Sezgisel yöntemlerin ürettiği taslak çizelge, <b>Metasezgisel çözücülere (Hill Climbing, Tabu Search, GA, PSO) başlangıç tohumu (initial seed)</b> olarak verildiğinde, arama süresi %80 oranında kısalır ve çok daha hızlı küresel optimuma ulaşılır.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # BÖLÜM 3: METASEZGİSEL (METAHEURISTIC) YÖNTEMLERİN KENDİ ARASINDA KARŞILAŞTIRILMASI
    # ==============================================================================
    st.markdown("### 🧬 Bölüm 3: Metasezgisel (Metaheuristic) Yöntemlerin Kendi Arasında Karşılaştırması (8-Yönlü Objektif Mühendislik Matrisi)")
    st.markdown("""
    > ⚖️ **No Free Lunch (NFL) Teoremi (Wolpert & Macready, 1997):**  
    > *"Tüm optimizasyon problemlerinde diğer yöntemlerden mutlak olarak üstün olan tek bir metasezgisel algoritma YOKTUR."*  
    > Her yöntemin **küresel keşif (Exploration)**, **yerel ince arama (Exploitation)**, **hesaplama hızı** ve **hiperparametre ayar hassasiyeti** arasında farklı mühendislik ödünleşimleri (trade-offs) bulunur.
    """)

    comp_html = """
    <div style="overflow-x: auto;">
    <table style="width:100%; border-collapse:collapse; margin-top:10px; font-size:0.82rem; background-color:#ffffff; border:1px solid #cbd5e1; border-radius:8px; overflow:hidden;">
        <thead>
            <tr style="background-color:#0f172a; color:#ffffff; text-align:left;">
                <th style="padding:10px 8px; width:12%;">Mühendislik Kriteri</th>
                <th style="padding:10px 8px; width:11%; color:#fdba74;">🏔️ Sekme 7: HC</th>
                <th style="padding:10px 8px; width:11%; color:#86efac;">🔥 Sekme 8: SA</th>
                <th style="padding:10px 8px; width:11%; color:#fca5a5;">🧬 Sekme 9: GA</th>
                <th style="padding:10px 8px; width:11%; color:#93c5fd;">🏆 Sekme 10: MA</th>
                <th style="padding:10px 8px; width:11%; color:#7dd3fc;">🤫 Sekme 11: TS</th>
                <th style="padding:10px 8px; width:11%; color:#2dd4bf;">🔄 Sekme 12: VNS</th>
                <th style="padding:10px 8px; width:11%; color:#fde047;">🐝 Sekme 13: PSO</th>
                <th style="padding:10px 8px; width:12%; color:#facc15;">🐜 Sekme 14: ACO</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:8px 8px; font-weight:bold; color:#334155;">Arama Tipi & Paradigma</td>
                <td style="padding:8px 8px;">Tek Noktalı Yöresel</td>
                <td style="padding:8px 8px;">Tek Noktalı Stokastik (Fiziksel Tavlama)</td>
                <td style="padding:8px 8px;">Popülasyon Tabanlı Evrimsel</td>
                <td style="padding:8px 8px; font-weight:bold; color:#1d4ed8;">Hibrit (GA Evrim + HC Onarım)</td>
                <td style="padding:8px 8px; color:#0369a1; font-weight:bold;">Tek Noktalı Deterministik Hafıza</td>
                <td style="padding:8px 8px; color:#0f766e; font-weight:bold;">Çoklu Komşuluk Değişimi</td>
                <td style="padding:8px 8px; color:#a16207; font-weight:bold;">Popülasyon Sürü Zekası (Bilişsel+Sosyal)</td>
                <td style="padding:8px 8px; color:#ca8a04; font-weight:bold;">Popülasyon Feromon Çizge Tabanlı</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:8px 8px; font-weight:bold; color:#334155;">Temel Güçlü Yönü (Avantaj)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">⚡ Yıldırım hızında (&lt;50 ms), 0 parametre ayarı</td>
                <td style="padding:8px 8px; color:#0369a1; font-weight:bold;">Metropolis sıçramalarıyla çukurdan kaçış</td>
                <td style="padding:8px 8px; color:#2563eb; font-weight:bold;">Geniş uzayı paralel tarama (Exploration)</td>
                <td style="padding:8px 8px; color:#1d4ed8; font-weight:bold;">🎯 En yüksek ortalama çözüm kalitesi</td>
                <td style="padding:8px 8px; color:#059669; font-weight:bold;">Tabu listesiyle döngüye girmeyen arama</td>
                <td style="padding:8px 8px; color:#0f766e; font-weight:bold;">N1, N2, N3 ile üstün posta takası</td>
                <td style="padding:8px 8px; color:#a16207; font-weight:bold;">Ortak hafıza (p_best + g_best) ile hızlı yönelim</td>
                <td style="padding:8px 8px; color:#ca8a04; font-weight:bold;">Feromon takviyesi ile kolektif çizelge inşası</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:8px 8px; font-weight:bold; color:#334155;">Temel Zayıf Yönü (Risk / Dezavantaj)</td>
                <td style="padding:8px 8px; color:#b91c1c; font-weight:bold;">⚠️ İlk yerel çukurda kesin kilitlenir</td>
                <td style="padding:8px 8px; color:#b91c1c;">⚠️ Soğuma hızına (&alpha;) aşırı duyarlıdır</td>
                <td style="padding:8px 8px; color:#b91c1c;">⚠️ İnce ayarda yavaş; dikiş hatası riski</td>
                <td style="padding:8px 8px; color:#b91c1c; font-weight:bold;">⏱️ En yüksek CPU maliyeti (~3-12 sn)</td>
                <td style="padding:8px 8px; color:#b91c1c;">⚠️ Tenure yanlışsa çözümleri hapseder</td>
                <td style="padding:8px 8px; color:#b91c1c;">⚠️ Probleme özel Nk tasarımı zorunludur</td>
                <td style="padding:8px 8px; color:#b91c1c;">⚠️ Çeşitlilik biterse erken yakınsama riski</td>
                <td style="padding:8px 8px; color:#b91c1c;">⚠️ Buharlaşma katsayısı yanlışsa erken kilitlenme</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:8px 8px; font-weight:bold; color:#334155;">Hiperparametre Hassasiyeti</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">🟢 Çok Kolay (0 Parametre)</td>
                <td style="padding:8px 8px; color:#d97706;">🟡 Orta (T_start, &alpha;)</td>
                <td style="padding:8px 8px; color:#dc2626; font-weight:bold;">🔴 Yüksek (Pop, Cross, Mut)</td>
                <td style="padding:8px 8px; color:#dc2626; font-weight:bold;">🔴 Çok Yüksek (GA + HC)</td>
                <td style="padding:8px 8px; color:#d97706;">🟡 Orta / Yüksek (Tenure)</td>
                <td style="padding:8px 8px; color:#d97706;">🟡 Orta (k_max, Shaking)</td>
                <td style="padding:8px 8px; color:#d97706;">🟡 Orta / Yüksek (w, c1, c2)</td>
                <td style="padding:8px 8px; color:#d97706;">🟡 Orta / Yüksek (m, &rho;, &alpha;, &beta;)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:8px 8px; font-weight:bold; color:#334155;">Arama Uzayı Keşfi (Exploration)</td>
                <td style="padding:8px 8px; color:#64748b;">Düşük (Lokal komşuluk)</td>
                <td style="padding:8px 8px; color:#0284c7;">Orta / Yüksek (Sıcakken)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">Çok Yüksek (Popülasyon)</td>
                <td style="padding:8px 8px; color:#15803d;">Yüksek (Popülasyon)</td>
                <td style="padding:8px 8px; color:#0284c7;">Orta (Tabu itişiyle)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">Yüksek (Shaking sıçraması)</td>
                <td style="padding:8px 8px; color:#15803d;">Yüksek (Hız vektörleri)</td>
                <td style="padding:8px 8px; color:#15803d;">Yüksek (Buharlaşma ve çoklu karınca)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#f8fafc;">
                <td style="padding:8px 8px; font-weight:bold; color:#334155;">Yerel İnce Arama (Exploitation)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">Çok Hızlı (Lokal gradyan)</td>
                <td style="padding:8px 8px; color:#64748b;">Yavaş (Stokastik)</td>
                <td style="padding:8px 8px; color:#64748b;">Düşük (Tek başına yetersiz)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">Çok Üstün (Özel HC ile)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">Çok Güçlü (En iyi komşu)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">Çok Güçlü (N1 mikro takas)</td>
                <td style="padding:8px 8px; color:#15803d;">Hızlı / Güçlü (Çekim)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">Çok Güçlü (MMAS + Daemon Action)</td>
            </tr>
            <tr style="border-bottom:1px solid #e2e8f0; background-color:#ffffff;">
                <td style="padding:8px 8px; font-weight:bold; color:#334155;">Hesaplama Hızı & CPU Yükü</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">⚡ Yıldırım (~50 - 300 ms)</td>
                <td style="padding:8px 8px; color:#0284c7; font-weight:bold;">🚀 Hızlı (~200 - 1,500 ms)</td>
                <td style="padding:8px 8px; color:#d97706;">⏱️ Orta (~2 - 6 sn)</td>
                <td style="padding:8px 8px; color:#b91c1c; font-weight:bold;">⏱️ Yoğun (~3 - 12 sn)</td>
                <td style="padding:8px 8px; color:#0284c7; font-weight:bold;">🚀 Hızlı (~300 - 2,500 ms)</td>
                <td style="padding:8px 8px; color:#15803d; font-weight:bold;">⚡ Çok Hızlı (~200 - 1,500 ms)</td>
                <td style="padding:8px 8px; color:#0284c7; font-weight:bold;">🚀 Hızlı (~300 - 2,500 ms)</td>
                <td style="padding:8px 8px; color:#0284c7; font-weight:bold;">🚀 Hızlı (~300 - 2,500 ms)</td>
            </tr>
        </tbody>
    </table>
    </div>
    """
    st.markdown(comp_html, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # BÖLÜM 4: ÇÖZÜCÜ KARŞILAŞTIRMALI VARDİYA & İZİN MUTABAKATI (CONSENSUS & ALIGNMENT)
    # ==============================================================================
    st.markdown("### 🔬 Bölüm 4: Çözücü Karşılaştırmalı Vardiya & İzin Mutabakatı (Schedule Alignment & Consensus Matrix)")
    st.markdown("""
    Bu modülde, **Matematiksel Kesin Çözücü (ILP)** ile **Metasezgisel Yöntemlerin (GA, MA, TS, VNS, PSO, SA, HC)** ve **Sezgisel Çözücülerin** ürettiği nihai vardiya matrisleri üst üste bindirilir.
    Algoritmaların hangi personel ve günlerde **tam mutabakata vardığı** (birebir aynı vardiyayı atadığı), hangi noktalarda **ayrıştığı** (izin vs çalışma çatışması veya vardiya tipi farkı) görsel ısı haritası ve analitik metriklerle ortaya konur.
    """)

    # Mevcut / Çalıştırılmış Çözücüler Kataloğu
    available_solvers = {}
    if res15: available_solvers["⚡ Sekme 15: Google CP-SAT (OR-Tools Portfolio)"] = res15
    if res6: available_solvers["🏛️ Sekme 6: ILP (Matematiksel Küresel Optimum)"] = res6
    if res5: available_solvers["🌲 Sekme 5: CSP Backtracking"] = res5
    if res10: available_solvers["🏆 Sekme 10: Memetic Algorithm (MA)"] = res10
    if res9: available_solvers["🧬 Sekme 9: Genetic Algorithm (GA)"] = res9
    if res14: available_solvers["🐜 Sekme 14: Ant Colony Optimization (ACO)"] = res14
    if res13: available_solvers["🐝 Sekme 13: Discrete PSO"] = res13
    if res12: available_solvers["🔄 Sekme 12: Variable Neighborhood Search (VNS)"] = res12
    if res11: available_solvers["🤫 Sekme 11: Tabu Search (TS)"] = res11
    if res8: available_solvers["🔥 Sekme 8: Simulated Annealing (SA)"] = res8
    if res7: available_solvers["🏔️ Sekme 7: Hill Climbing (HC)"] = res7
    if res_g3: available_solvers["⚡ Sekme 4: Kısıt Öncelikli Sezgisel (MRV/LCV)"] = res_g3
    if res_g2: available_solvers["⚡ Sekme 4: Kademeli Sezgisel (Staggered)"] = res_g2
    if res_g1: available_solvers["⚡ Sekme 4: Sıralı Miyopik Sezgisel (Sequential)"] = res_g1

    if len(available_solvers) < 2:
        st.warning("⚠️ Karşılaştırma yapabilmek için en az 2 farklı çözücünün çalıştırılmış olması gerekir. Yukarıdaki **'⚡ Tüm Çözücüleri Çalıştır (Benchmark)'** butonuna basarak tüm sonuçları tek tıkla üretebilirsiniz.")
    else:
        solver_names = list(available_solvers.keys())
        default_idx_A = 0
        default_idx_B = min(1, len(solver_names) - 1)
        for i, name in enumerate(solver_names):
            if "GA" in name or "Memetic" in name or "PSO" in name:
                default_idx_B = i
                break

        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            name_A = st.selectbox(
                "📌 Referans Çözücü A (Baz Alınan Çözüm):",
                options=solver_names,
                index=default_idx_A,
                key="cmp_sel_a"
            )
        with c_sel2:
            name_B = st.selectbox(
                "🔄 Karşılaştırılan Çözücü B:",
                options=solver_names,
                index=default_idx_B,
                key="cmp_sel_b"
            )

        sol_A = available_solvers[name_A]
        sol_B = available_solvers[name_B]
        sched_A = sol_A['schedule']
        sched_B = sol_B['schedule']

        s_map = {0: "OFF", 1: "Gündüz", 2: "Akşam", 3: "Gece"}
        shift_full_names = {0: "İzin (OFF)", 1: "Gündüz (08-16)", 2: "Akşam (16-24)", 3: "Gece (24-08)"}
        total_slots = num_workers * num_days

        # 1. Metrik Hesaplamaları
        exact_matches = int((sched_A == sched_B).sum())
        exact_pct = round((exact_matches / total_slots) * 100.0, 1)

        # İzin (OFF) uyumu
        off_A = (sched_A == 0)
        off_B = (sched_B == 0)
        off_both = int((off_A & off_B).sum())
        off_either = int((off_A | off_B).sum())
        off_agree_state = int((off_A == off_B).sum())
        off_agree_pct = round((off_agree_state / total_slots) * 100.0, 1)

        # Gece Nöbeti uyumu
        night_A = (sched_A == 3)
        night_B = (sched_B == 3)
        night_both = int((night_A & night_B).sum())
        night_either = int((night_A | night_B).sum())
        night_match_pct = round((night_both / max(1, night_either)) * 100.0, 1) if night_either > 0 else 100.0

        # Skor ve İyileştirme Farkı
        score_A = sol_A['final_score']
        score_B = sol_B['final_score']
        score_diff = abs(score_A - score_B)

        # KPI KARTLARI
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #10b981;">
                <div class="metric-label">🎯 Tam Vardiya Mutabakatı</div>
                <div class="metric-value" style="color: #059669;">%{exact_pct}</div>
                <div style="font-size:0.80rem; color:#64748b; margin-top:2px;">{exact_matches} / {total_slots} Hücre Birebir Aynı</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #3b82f6;">
                <div class="metric-label">🏖️ İzin (OFF) Kararı Uyumu</div>
                <div class="metric-value" style="color: #2563eb;">%{off_agree_pct}</div>
                <div style="font-size:0.80rem; color:#64748b; margin-top:2px;">{off_both} Ortak İzin Günü</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #8b5cf6;">
                <div class="metric-label">🌙 Gece Vardiyası Mutabakatı</div>
                <div class="metric-value" style="color: #7c3aed;">%{night_match_pct}</div>
                <div style="font-size:0.80rem; color:#64748b; margin-top:2px;">{night_both} Ortak Gece Nöbeti</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #f59e0b;">
                <div class="metric-label">⚖️ Skor Farkı (|ΔZ|)</div>
                <div class="metric-value" style="color: #d97706;">{score_diff} Puan</div>
                <div style="font-size:0.80rem; color:#64748b; margin-top:2px;">A: {score_A} | B: {score_B}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. Mutabakat Isı Haritası (Overlay Consensus Heatmap)
        st.markdown("##### 🗺️ Karşılaştırmalı Vardiya & İzin Mutabakat Haritası (Overlay Matrix)")
        st.markdown("""
        - 🟩 **Yeşil (Tam Mutabakat):** İki yöntem de aynı personele aynı gün **aynı vardiyayı** atamıştır.
        - 🟨 **Sarı (Vardiya Tipi Farkı):** İki yöntem de personeli çalıştırmış, ancak biri *Gündüz/Akşam/Gece* farklılığı seçmiştir.
        - 🟥 **Kırmızı (İzin Çatışması):** Bir yöntem personele *İzin (OFF)* verirken, diğeri *Çalışma* yazmıştır.
        """)

        # Matris Hazırlığı
        diff_matrix = np.zeros((num_workers, num_days), dtype=int)
        text_matrix = []
        hover_matrix = []

        day_cols = [f"{d+1}. Gün" for d in range(num_days)]
        worker_rows = [f"{w['name']} ({w['posta']})" for w in workers]

        for w_idx, w in enumerate(workers):
            t_row = []
            h_row = []
            for d in range(num_days):
                v_A = sched_A[w_idx, d]
                v_B = sched_B[w_idx, d]
                s_A_txt = s_map[v_A]
                s_B_txt = s_map[v_B]
                pref_txt = f" (Talep Edilen İzin: {w.get('pref_off', '-')}. Gün)" if w.get('pref_off', 0) == (d+1) else ""

                if v_A == v_B:
                    diff_matrix[w_idx, d] = 0
                    t_row.append(f"✅ {s_A_txt}")
                    h_row.append(f"<b>{w['name']}</b> ({w['posta']})<br><b>Gün:</b> {d+1}. Gün{pref_txt}<br><b>Karar:</b> Birebir Aynı: <b>{shift_full_names[v_A]}</b><br><b>Durum:</b> 🟢 Tam Mutabakat")
                elif v_A != 0 and v_B != 0:
                    diff_matrix[w_idx, d] = 1
                    t_row.append(f"⚡ {s_A_txt}/{s_B_txt}")
                    h_row.append(f"<b>{w['name']}</b> ({w['posta']})<br><b>Gün:</b> {d+1}. Gün{pref_txt}<br><b>{name_A.split(':')[0]}:</b> {shift_full_names[v_A]}<br><b>{name_B.split(':')[0]}:</b> {shift_full_names[v_B]}<br><b>Durum:</b> 🟡 Vardiya Tipi Farkı")
                else:
                    diff_matrix[w_idx, d] = 2
                    t_row.append(f"⚠️ {s_A_txt}/{s_B_txt}")
                    h_row.append(f"<b>{w['name']}</b> ({w['posta']})<br><b>Gün:</b> {d+1}. Gün{pref_txt}<br><b>{name_A.split(':')[0]}:</b> {shift_full_names[v_A]}<br><b>{name_B.split(':')[0]}:</b> {shift_full_names[v_B]}<br><b>Durum:</b> 🔴 İzin vs Çalışma Çatışması")
            text_matrix.append(t_row)
            hover_matrix.append(h_row)

        colorscale_comp = [
            [0.0, '#10b981'], [0.33, '#10b981'],   # Yeşil: Tam Mutabakat
            [0.34, '#f59e0b'], [0.66, '#f59e0b'],  # Sarı: Vardiya Tipi Farkı
            [0.67, '#ef4444'], [1.0, '#ef4444']    # Kırmızı: İzin Çatışması
        ]

        fig_overlay = go.Figure(data=go.Heatmap(
            z=diff_matrix,
            x=day_cols,
            y=worker_rows,
            text=text_matrix,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white", "family": "Inter, sans-serif"},
            hovertext=hover_matrix,
            hoverinfo="text",
            colorscale=colorscale_comp,
            zmin=0,
            zmax=2,
            showscale=False
        ))

        fig_overlay.update_layout(
            height=max(480, num_workers * 26 + 120),
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis=dict(tickangle=0, side='top', showgrid=False),
            yaxis=dict(autorange='reversed', showgrid=False)
        )
        st.plotly_chart(fig_overlay, width="stretch", key="bench_fig_overlay")

        # 3. Ayrışma Detayları & Yan Yana İnceleme Sekmeleri
        st.markdown("<br>", unsafe_allow_html=True)
        tab_discrepancy, tab_side_by_side, tab_ensemble = st.tabs([
            "📋 Ayrışan Vardiyalar ve İzin Farkları Detay Tablosu",
            "📊 Yan Yana Çizelge Matrisleri (Side-by-Side)",
            "🌐 Çoklu Çözücü Konsensüs Analizi (Ensemble Consensus)"
        ])

        with tab_discrepancy:
            discrepancy_rows = []
            for w_idx, w in enumerate(workers):
                for d in range(num_days):
                    v_A = sched_A[w_idx, d]
                    v_B = sched_B[w_idx, d]
                    if v_A != v_B:
                        is_pref = "🎯 EVET (Talep Edilen Gün)" if (w.get('pref_off', 0) == (d + 1)) else "Hayır"
                        diff_type = "🟡 Vardiya Tipi Farkı" if (v_A != 0 and v_B != 0) else "🔴 İzin vs Çalışma Çatışması"
                        discrepancy_rows.append({
                            "Personel": w['name'],
                            "Posta": w['posta'],
                            "Unvan": "Kıdemli Usta" if w['is_usta'] else "İşçi",
                            "Gün": f"{d+1}. Gün",
                            f"{name_A.split(':')[0]} Kararı": shift_full_names[v_A],
                            f"{name_B.split(':')[0]} Kararı": shift_full_names[v_B],
                            "Ayrışma Türü": diff_type,
                            "Kişisel İzin Günü mü?": is_pref
                        })

            if discrepancy_rows:
                df_disc = pd.DataFrame(discrepancy_rows)
                st.markdown(f"Toplam **{len(discrepancy_rows)}** hücrede iki yöntem farklı karar vermiştir:")
                st.dataframe(df_disc, width="stretch", hide_index=True)
            else:
                st.success("🎉 **Kusursuz Birebir Eşleşme:** İki çözücü de tüm personel ve günler için %100 birebir aynı çizelgeyi üretmiştir!")

        with tab_side_by_side:
            st.markdown("""
            <div style="display:flex; align-items:center; gap:14px; margin-bottom:12px; font-size:0.85rem; font-weight:600; background-color:#f8fafc; padding:8px 14px; border-radius:8px; border:1px solid #e2e8f0;">
                <span style="color:#475569;">🎨 Vardiya Renk Kodları:</span>
                <span style="background-color:#cbd5e1; color:#0f172a; padding:3px 10px; border-radius:4px; border:1px solid #94a3b8;">⬜ OFF (İzin)</span>
                <span style="background-color:#fde047; color:#713f12; padding:3px 10px; border-radius:4px; border:1px solid #eab308;">🟨 Gündüz (08-16)</span>
                <span style="background-color:#f97316; color:#ffffff; padding:3px 10px; border-radius:4px;">🟧 Akşam (16-24)</span>
                <span style="background-color:#1e3a8a; color:#ffffff; padding:3px 10px; border-radius:4px;">🟦 Gece (24-08)</span>
            </div>
            """, unsafe_allow_html=True)

            def create_schedule_color_heatmap(sched, w_list, name_title, score_val):
                short_labels = {0: "OFF", 1: "Gün", 2: "Akş", 3: "Gec"}
                text_grid = [[short_labels[sched[w_i, d_i]] for d_i in range(num_days)] for w_i in range(num_workers)]
                hover_grid = []
                for w_i, w in enumerate(w_list):
                    h_row = []
                    for d_i in range(num_days):
                        v = sched[w_i, d_i]
                        h_row.append(
                            f"<b>{w['name']}</b> ({w['posta']})<br>"
                            f"<b>Gün:</b> {d_i+1}. Gün<br>"
                            f"<b>Vardiya:</b> {shift_full_names[v]}<br>"
                            f"<b>Kıdem:</b> {'Kıdemli Usta' if w['is_usta'] else 'İşçi'}"
                        )
                    hover_grid.append(h_row)

                colorscale_shifts = [
                    [0.0, '#cbd5e1'], [0.25, '#cbd5e1'],
                    [0.26, '#fde047'], [0.50, '#fde047'],
                    [0.51, '#f97316'], [0.75, '#f97316'],
                    [0.76, '#1e3a8a'], [1.0, '#1e3a8a']
                ]

                fig = go.Figure(data=go.Heatmap(
                    z=sched,
                    x=day_cols,
                    y=worker_rows,
                    text=text_grid,
                    texttemplate="%{text}",
                    textfont=dict(size=10, family="Inter, sans-serif"),
                    hovertext=hover_grid,
                    hoverinfo="text",
                    colorscale=colorscale_shifts,
                    zmin=0,
                    zmax=3,
                    showscale=False
                ))

                fig.update_layout(
                    title=dict(text=f"<b>{name_title}</b> (Skor: {score_val} Puan)", font=dict(size=12, color="#1e293b")),
                    height=max(480, num_workers * 25 + 100),
                    margin=dict(l=10, r=10, t=35, b=10),
                    xaxis=dict(tickangle=0, side='top', showgrid=False),
                    yaxis=dict(autorange='reversed', showgrid=False)
                )
                return fig

            s_col1, s_col2 = st.columns(2)
            with s_col1:
                fig_A = create_schedule_color_heatmap(sched_A, workers, name_A, sol_A['final_score'])
                st.plotly_chart(fig_A, width="stretch", key="bench_fig_side_a")

            with s_col2:
                fig_B = create_schedule_color_heatmap(sched_B, workers, name_B, sol_B['final_score'])
                st.plotly_chart(fig_B, width="stretch", key="bench_fig_side_b")

            with st.expander("📄 Ham Metin Tablolarını Görüntüle (Kopyalama & Dışa Aktarma İçin)", expanded=False):
                c_tbl1, c_tbl2 = st.columns(2)
                with c_tbl1:
                    st.markdown(f"**{name_A}**")
                    df_A = pd.DataFrame(sched_A, columns=day_cols, index=[w['name'] for w in workers]).replace(shift_full_names)
                    df_A.reset_index(names=["Personel"], inplace=True)
                    st.dataframe(df_A, width="stretch", hide_index=True)
                with c_tbl2:
                    st.markdown(f"**{name_B}**")
                    df_B = pd.DataFrame(sched_B, columns=day_cols, index=[w['name'] for w in workers]).replace(shift_full_names)
                    df_B.reset_index(names=["Personel"], inplace=True)
                    st.dataframe(df_B, width="stretch", hide_index=True)

        with tab_ensemble:
            st.markdown("##### 🌐 Tüm Çalıştırılmış Çözücülerin Konsensüs (Ortak Akıl) Analizi")
            st.markdown("""
            Bu analiz; sistemde çalıştırılmış olan tüm çözücülerin (<i>K</i> adet) her bir hücre (<i>w, d</i>) için **çoğunluk oyu konsensüs oranını (%)** hesaplar.
            Konsensüs oranı %100 olan hücreler, problemin matematiksel yapısı gereği tüm sezgisel ve matematiksel modellerin **kesin olarak aynı vardiyaya mecbur kaldığı kilit düğümleri** gösterir.
            """)

            ran_schedules = [sol['schedule'] for sol in available_solvers.values() if sol and 'schedule' in sol]
            if len(ran_schedules) >= 3:
                stacked = np.stack(ran_schedules, axis=0) # (K, N, D)
                k_solvers = stacked.shape[0]

                consensus_matrix = np.zeros((num_workers, num_days), dtype=float)
                consensus_text = []
                consensus_hover = []

                for w_idx, w in enumerate(workers):
                    c_row_txt = []
                    c_row_hov = []
                    for d in range(num_days):
                        cell_vals = stacked[:, w_idx, d]
                        counts = np.bincount(cell_vals, minlength=4)
                        majority_shift = int(np.argmax(counts))
                        maj_count = int(counts[majority_shift])
                        agreement_ratio = round((maj_count / k_solvers) * 100.0, 1)
                        consensus_matrix[w_idx, d] = agreement_ratio

                        c_row_txt.append(f"%{int(agreement_ratio)}")
                        c_row_hov.append(f"<b>{w['name']}</b> ({w['posta']})<br><b>Gün:</b> {d+1}. Gün<br><b>Çoğunluk Kararı:</b> {shift_full_names[majority_shift]} ({maj_count}/{k_solvers} Çözücü)<br><b>Konsensüs Oranı:</b> %{agreement_ratio}")
                    consensus_text.append(c_row_txt)
                    consensus_hover.append(c_row_hov)

                mean_consensus = round(float(np.mean(consensus_matrix)), 1)
                full_consensus_slots = int((consensus_matrix == 100.0).sum())

                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.metric("Çalıştırılan Çözücü Sayısı", f"{k_solvers} Çözücü")
                with ec2:
                    st.metric("Ortalama Sürü Konsensüsü", f"%{mean_consensus}")
                with ec3:
                    st.metric("Tam Mutabakat Hücreleri (%100)", f"{full_consensus_slots} / {total_slots} Hücre")

                fig_ens = go.Figure(data=go.Heatmap(
                    z=consensus_matrix,
                    x=day_cols,
                    y=worker_rows,
                    text=consensus_text,
                    texttemplate="%{text}",
                    textfont={"size": 11, "color": "white"},
                    hovertext=consensus_hover,
                    hoverinfo="text",
                    colorscale="Blues",
                    zmin=25,
                    zmax=100,
                    colorbar=dict(title="Konsensüs %")
                ))
                fig_ens.update_layout(
                    height=max(450, num_workers * 24 + 100),
                    margin=dict(l=10, r=10, t=25, b=10),
                    xaxis=dict(tickangle=0, side='top', showgrid=False),
                    yaxis=dict(autorange='reversed', showgrid=False)
                )
                st.plotly_chart(fig_ens, width="stretch", key="bench_fig_ensemble")
            else:
                st.info(f"ℹ️ Çoklu konsensüs analizi için en az 3 çözücünün çalıştırılmış olması gerekir (Şu an çalışan: {len(ran_schedules)}). Yukarıdaki '⚡ Tüm Çözücüleri Çalıştır (Benchmark)' butonuna basarak tüm çözücüleri anında devreye alabilirsiniz.")
