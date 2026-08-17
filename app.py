"""
  Ana Giriş Dosyası (Master App Entry Point)
  Bu dosya modüler mimari ile yapılandırılmıştır
"""

import streamlit as st

# Modüler Modül İçe Aktarımları
from config import apply_custom_styles
from global_state import render_global_sidebar
from views.tab1_nsp_intro import render_tab1
from views.tab2_industries import render_tab2
from views.tab3_steel_model import render_tab3
from views.tab4_simulation import render_tab4
from views.tab5_csp_backtracking import render_tab5
from views.tab6_ilp_optimization import render_tab6
from views.tab7_hill_climbing import render_tab7
from views.tab8_simulated_annealing import render_tab8
from views.tab9_genetic_algorithm import render_tab9
from views.tab10_memetic_algorithm import render_tab10
from views.tab11_tabu_search import render_tab11
from views.tab_comparison import render_tab_comparison


# ==============================================================================
# 1. STREAMLIT SAYFA YAPILANDIRMASI
# ==============================================================================
st.set_page_config(
    page_title="Hemşire Çizelgeleme Problemi & Sanayi Uygulamaları",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_custom_styles()

#(SOL MENÜ)
global_params = render_global_sidebar()

# 3. UYGULAMA BAŞLIĞI VE HERO BANNER
st.markdown("""<div class="hero-container">
<div class="hero-title">Hemşire Çizelgeleme Problemi (NSP/NRP) & Sanayi Uygulamaları</div>
<div class="hero-subtitle">
Yöneylem Araştırması (Operations Research) alanının en temel optimizasyon problemlerinden biri olan 
<b>Nurse Scheduling Problem (NSP)</b> tanımı, matematiksel kısıtları, tarihçesi ve ağır sanayi kollarındaki kullanım alanları rehberi.
</div>
</div>""", unsafe_allow_html=True)


# ==============================================================================
# 4. 2 KATLI (2-TIER GRID) SEKME GEZİNTİSİ & LAZY EVALUATION
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

# 3. SATIR: METASEZGİSEL OPTİMİZASYON ÇÖZÜCÜLERİ - BÖLÜM 1 (3 KOLON)
c3_1, c3_2, c3_3 = st.columns(3)

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

# 4. SATIR: GELİŞMİŞ METASEZGİSEL ÇÖZÜCÜLER - BÖLÜM 2 (2 KOLON)
c4_1, c4_2 = st.columns(2)

with c4_1:
    btn_type = "primary" if st.session_state["active_tab"] == "tab10" else "secondary"
    if st.button("🏆 Sekme 10: Memetik Algoritma", key="btn_t10", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab10"
        st.rerun()

with c4_2:
    btn_type = "primary" if st.session_state["active_tab"] == "tab11" else "secondary"
    if st.button("🤫 Sekme 11: Tabu Search", key="btn_t11", width="stretch", type=btn_type):
        st.session_state["active_tab"] = "tab11"
        st.rerun()

# 5. SATIR / ALT NAVİGASYON: TÜM NAVİGASYONUN ALTINDA TEK BAŞINA DURAN SON SEKME
btn_type_comp = "primary" if st.session_state["active_tab"] == "tab_comparison" else "secondary"
if st.button("⚖️ Son Sekme: Bütüncül Karşılaştırma & Metasezgisel Benchmark Analizi", key="btn_t_comp", width="stretch", type=btn_type_comp):
    st.session_state["active_tab"] = "tab_comparison"
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
elif curr == "tab4":
    render_tab4(global_params)
elif curr == "tab5":
    render_tab5(global_params)
elif curr == "tab6":
    render_tab6(global_params)
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
elif curr == "tab_comparison":
    render_tab_comparison(global_params)
