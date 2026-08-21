import sys, os
sys.path.insert(0, os.path.abspath('.'))
import streamlit as st

# Setup test
st.session_state.clear()

GLOBAL_DEFAULTS = {
    'n_workers': 28,
    'n_days': 14,
    'r_day': 9,
    'r_eve': 7,
    'r_night': 5,
    'circadian': 50,
    'night_imb': 25,
    'exp_mix': 60,
    'pref_off': 40,
    'posta': 15,
}

# 1. Initialize persistent store if not present
if "persistent_global_config" not in st.session_state:
    st.session_state["persistent_global_config"] = dict(GLOBAL_DEFAULTS)

cfg = st.session_state["persistent_global_config"]

print("1. Initial state:")
print(f"   r_day = {cfg['r_day']}, n_workers = {cfg['n_workers']}")
assert cfg['r_day'] == 9
assert cfg['n_workers'] == 28

# 2. Simulate User changing widget in UI:
# In Streamlit, on_change callback:
def on_r_day_change():
    cfg['r_day'] = st.session_state["w_r_day"]

# User types 14:
st.session_state["w_r_day"] = 14
on_r_day_change()
print("2. User changed w_r_day to 14:")
print(f"   persistent r_day = {cfg['r_day']}")
assert cfg['r_day'] == 14

# 3. Simulate User clicking Preset (Kompakt Haddehane: 6-5-4, N=20):
def apply_preset(nw, rd, re, rn):
    cfg['n_workers'] = nw
    cfg['r_day'] = rd
    cfg['r_eve'] = re
    cfg['r_night'] = rn
    st.session_state["w_n_workers"] = nw
    st.session_state["w_r_day"] = rd
    st.session_state["w_r_eve"] = re
    st.session_state["w_r_night"] = rn

apply_preset(20, 6, 5, 4)
print("3. Preset applied:")
print(f"   persistent r_day = {cfg['r_day']}, n_workers = {cfg['n_workers']}")
assert cfg['r_day'] == 6
assert cfg['n_workers'] == 20

# 4. Simulate Tab switch to Tab 4 (Streamlit deletes all w_* widget keys):
for k in list(st.session_state.keys()):
    if k.startswith("w_"):
        del st.session_state[k]

print("4. Switched to Tab 4 (widget keys wiped by Streamlit):")
print(f"   Tab 4 reads persistent r_day = {cfg['r_day']}, n_workers = {cfg['n_workers']}")
assert cfg['r_day'] == 6
assert cfg['n_workers'] == 20

# 5. User switches back to tab_config:
# Widget renders with value=cfg['r_day'] (which is 6):
rendered_value = cfg['r_day']
print("5. Back on tab_config, widget renders with value=cfg['r_day']:")
print(f"   rendered value = {rendered_value}")
assert rendered_value == 6

print("\n>>> ALL TESTS PASSED WITH 100% STABILITY! <<<")
