"""
  Ana Giriş Dosyası (Master App Entry Point)
  Bu dosya modüler mimari ile yapılandırılmıştır
"""

import streamlit as st

# Modüler Modül İçe Aktarımları
from config import apply_custom_styles
from global_state import get_global_params, render_global_config_tab
from views.tab1_nsp_intro import render_tab1
from views.tab2_industries import render_tab2
from views.tab3_steel_model import render_tab3
from views.tab4_simulation import render_tab4
from views.tab5_csp_backtracking import render_tab5
from views.tab6_ilp_optimization import render_tab6
from views.tab_metaheuristic_guide import render_tab_metaheuristic_guide
from views.tab7_hill_climbing import render_tab7
from views.tab8_simulated_annealing import render_tab8
from views.tab9_genetic_algorithm import render_tab9
from views.tab10_memetic_algorithm import render_tab10
from views.tab11_tabu_search import render_tab11
from views.tab12_vns import render_tab12
from views.tab13_pso import render_tab13
from views.tab14_aco import render_tab14
from views.tab15_cp_sat import render_tab15
from views.tab16_multi_objective import render_tab16
from views.tab17_nsga2 import render_tab17
from views.tab18_epsilon_constraint import render_tab18
from views.tab19_moead import render_tab19
from views.tab_comparison import render_tab_comparison


# ==============================================================================
# 1. STREAMLIT SAYFA YAPILANDIRMASI
# ==============================================================================
st.set_page_config(
    page_title="Hemşire Çizelgeleme Problemi & Sanayi Uygulamaları",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="collapsed"
)
apply_custom_styles()

# Merkezi Global Model Parametreleri (Tüm sekmeler için tek kaynak)
global_params = get_global_params()

# 3. UYGULAMA BAŞLIĞI VE HERO BANNER
st.markdown("""<div class="hero-container">
<div class="hero-title">Hemşire Çizelgeleme Problemi (NSP/NRP) & Sanayi Uygulamaları</div>
<div class="hero-subtitle">
Yöneylem Araştırması (Operations Research) alanının en temel optimizasyon problemlerinden biri olan 
<b>Nurse Scheduling Problem (NSP)</b> tanımı, matematiksel kısıtları, tarihçesi ve ağır sanayi kollarındaki kullanım alanları rehberi.
</div>
</div>""", unsafe_allow_html=True)


# ==============================================================================
# 4. ÇOK KATLI SEKME GEZİNTİSİ & LAZY EVALUATION
# ==============================================================================
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "tab1"

st.markdown("##### 📌 Sayfa & Algoritma Navigasyonu (Sekme Seçin):")

# 1. SATIR: TEORİK KAVRAMLAR VE SANAYİ MODELLERİ (3 KOLON)
c1_1, c1_2, c1_3 = st.columns(3)

with c1_1:
    btn_type = "primary" if st.session_state["active_tab"] == "tab1" else "secondary"
    if st.button("🏥 Sekme 1: NSP Tanımı & Tarihçe", key="btn_t1", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab1"
        st.rerun()

with c1_2:
    btn_type = "primary" if st.session_state["active_tab"] == "tab2" else "secondary"
    if st.button("🏭 Sekme 2: Sanayi Uygulamaları", key="btn_t2", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab2"
        st.rerun()

with c1_3:
    btn_type = "primary" if st.session_state["active_tab"] == "tab3" else "secondary"
    if st.button("🏗️ Sekme 3: Çelik Tesis Kısıtları", key="btn_t3", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab3"
        st.rerun()

# ARA SATIR 1: MERKEZİ MODEL & KADRO YÖNETİMİ (Tek başına satır - Tab 3'ten sonra, Tab 4'ten önce)
btn_type_config = "primary" if st.session_state["active_tab"] == "tab_config" else "secondary"
if st.button("⚙️ Merkezi Model Yapılandırması & Ortak Kadro Yönetimi (Global Parametreler)", key="btn_t_config", width="stretch", type=btn_type_config):
    st.session_state["active_tab"] = "tab_config"
    st.rerun()

# 2. SATIR: DOĞRUDAN VE MATEMATİKSEL ÇÖZÜCÜLER (3 KOLON)
c2_1, c2_2, c2_3 = st.columns(3)

with c2_1:
    btn_type = "primary" if st.session_state["active_tab"] == "tab4" else "secondary"
    if st.button("⚡ Sekme 4: Greedy Simülasyonu", key="btn_t4", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab4"
        st.rerun()

with c2_2:
    btn_type = "primary" if st.session_state["active_tab"] == "tab5" else "secondary"
    if st.button("🔍 Sekme 5: Backtracking (CSP)", key="btn_t5", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab5"
        st.rerun()

with c2_3:
    btn_type = "primary" if st.session_state["active_tab"] == "tab6" else "secondary"
    if st.button("🎯 Sekme 6: ILP / MILP Optimizasyon", key="btn_t6", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab6"
        st.rerun()

# ARA SATIR 2: METASEZGİSEL PROBLEM ADAPTASYON REHBERİ (Tek başına satır - Tab 6 ile Tab 7 arasında)
btn_type_meta_guide = "primary" if st.session_state["active_tab"] == "tab_meta_guide" else "secondary"
if st.button("🧩 Metasezgisel Algoritmalar Rehberi: Problemden Bağımsız Çatılar & Her Probleme Adaptasyon Metodolojisi", key="btn_t_meta_guide", width="stretch", type=btn_type_meta_guide):
    st.session_state["active_tab"] = "tab_meta_guide"
    st.rerun()

# 3. SATIR: METASEZGİSEL OPTİMİZASYON ÇÖZÜCÜLERİ - 1. GRUP (4 KOLON)
c3_1, c3_2, c3_3, c3_4 = st.columns(4)

with c3_1:
    btn_type = "primary" if st.session_state["active_tab"] == "tab7" else "secondary"
    if st.button("🏔️ Sekme 7: Hill Climbing", key="btn_t7", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab7"
        st.rerun()

with c3_2:
    btn_type = "primary" if st.session_state["active_tab"] == "tab8" else "secondary"
    if st.button("♨️ Sekme 8: Simulated Annealing", key="btn_t8", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab8"
        st.rerun()

with c3_3:
    btn_type = "primary" if st.session_state["active_tab"] == "tab9" else "secondary"
    if st.button("🧬 Sekme 9: Genetic Algorithm", key="btn_t9", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab9"
        st.rerun()

with c3_4:
    btn_type = "primary" if st.session_state["active_tab"] == "tab10" else "secondary"
    if st.button("🏆 Sekme 10: Memetik Algoritma", key="btn_t10", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab10"
        st.rerun()

# 4. SATIR: METASEZGİSEL OPTİMİZASYON ÇÖZÜCÜLERİ - 2. GRUP (4 KOLON)
c4_1, c4_2, c4_3, c4_4 = st.columns(4)

with c4_1:
    btn_type = "primary" if st.session_state["active_tab"] == "tab11" else "secondary"
    if st.button("🤫 Sekme 11: Tabu Search", key="btn_t11", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab11"
        st.rerun()

with c4_2:
    btn_type = "primary" if st.session_state["active_tab"] == "tab12" else "secondary"
    if st.button("🔄 Sekme 12: VNS Solver", key="btn_t12", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab12"
        st.rerun()

with c4_3:
    btn_type_t13 = "primary" if st.session_state["active_tab"] == "tab13" else "secondary"
    if st.button("🐝 Sekme 13: Discrete PSO", key="btn_t13", width="stretch", type=btn_type_t13):
        st.session_state["active_tab"] = "tab13"
        st.rerun()

with c4_4:
    btn_type_t14 = "primary" if st.session_state["active_tab"] == "tab14" else "secondary"
    if st.button("🐜 Sekme 14: ACO Solver", key="btn_t14", width="stretch", type=btn_type_t14):
        st.session_state["active_tab"] = "tab14"
        st.rerun()

# 5. SATIR: KISIT PROGRAMLAMA & SAT ÇÖZÜCÜSÜ (Tek başına satır)
btn_type_t15 = "primary" if st.session_state["active_tab"] == "tab15" else "secondary"
if st.button("⚡ Sekme 15: Google CP-SAT (Constraint Programming & SAT Çözücüsü)", key="btn_t15", width="stretch", type=btn_type_t15):
    st.session_state["active_tab"] = "tab15"
    st.rerun()

# 6. SATIR: KARŞILAŞTIRMA & BENCHMARK ANALİZİ (Tek başına satır)
btn_type_comp = "primary" if st.session_state["active_tab"] == "tab_comparison" else "secondary"
if st.button("⚖️ Sekme 4-15 Arası Karşılaştırma & Benchmark Analizi", key="btn_t_comp", width="stretch", type=btn_type_comp):
    st.session_state["active_tab"] = "tab_comparison"
    st.rerun()

# 7. SATIR: ÇOK AMAÇLI OPTİMİZASYON & PARETO ANALİZİ (4 KOLON)
c7_1, c7_2, c7_3, c7_4 = st.columns(4)

with c7_1:
    btn_type_t16 = "primary" if st.session_state["active_tab"] == "tab16" else "secondary"
    if st.button("🎯 Sekme 16: MOO Teorisi & Pareto", key="btn_t16", width="stretch", type=btn_type_t16):
        st.session_state["active_tab"] = "tab16"
        st.rerun()

with c7_2:
    btn_type_t17 = "primary" if st.session_state["active_tab"] == "tab17" else "secondary"
    if st.button("🧬 Sekme 17: NSGA-II Çok Amaçlı (Deb 2002)", key="btn_t17", width="stretch", type=btn_type_t17):
        st.session_state["active_tab"] = "tab17"
        st.rerun()

with c7_3:
    btn_type_t18 = "primary" if st.session_state["active_tab"] == "tab18" else "secondary"
    if st.button("🎯 Sekme 18: ε-Kısıt Kesin Pareto (CP-SAT/MILP)", key="btn_t18", width="stretch", type=btn_type_t18):
        st.session_state["active_tab"] = "tab18"
        st.rerun()

with c7_4:
    btn_type_t19 = "primary" if st.session_state["active_tab"] == "tab19" else "secondary"
    if st.button("🌐 Sekme 19: MOEA/D Ayrıştırma (Zhang 2007)", key="btn_t19", width="stretch", type=btn_type_t19):
        st.session_state["active_tab"] = "tab19"
        st.rerun()

st.divider()

# YALNIZCA AKTİF OLAN SEKMENİN İÇERİĞİ VE HESAPLAMASI ÇALIŞTIRILIR!
curr = st.session_state["active_tab"]

if curr == "tab1":
    render_tab1()
elif curr == "tab2":
    render_tab2()
elif curr == "tab3":
    render_tab3()
elif curr == "tab_config":
    render_global_config_tab()
elif curr == "tab4":
    render_tab4(global_params)
elif curr == "tab5":
    render_tab5(global_params)
elif curr == "tab6":
    render_tab6(global_params)
elif curr == "tab_meta_guide":
    render_tab_metaheuristic_guide()
elif curr == "tab7":
    render_tab7(global_params)
elif curr == "tab8":
    render_tab8(global_params)
elif curr == "tab9":
    render_tab9(global_params)
elif curr == "tab10":
    render_tab10(global_params)
elif curr == "tab11":
    render_tab11(global_params)
elif curr == "tab12":
    render_tab12(global_params)
elif curr == "tab13":
    render_tab13(global_params)
elif curr == "tab14":
    render_tab14(global_params)
elif curr == "tab15":
    render_tab15(global_params)
elif curr == "tab16":
    render_tab16(global_params)
elif curr == "tab17":
    render_tab17(global_params)
elif curr == "tab18":
    render_tab18(global_params)
elif curr == "tab19":
    render_tab19(global_params)
elif curr == "tab_comparison":
    render_tab_comparison(global_params)
