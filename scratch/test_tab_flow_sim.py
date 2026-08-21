import sys, os
sys.path.insert(0, os.path.abspath('.'))
import streamlit as st
import global_state

print("=== STEP 1: INITIAL RUN (Tab 1 active) ===")
st.session_state.clear()
st.session_state["active_tab"] = "tab1"
p1 = global_state.get_global_params()
print(f"Tab 1 loaded. r_day in persistent cfg: {st.session_state['persistent_global_config']['r_day']}")

print("\n=== STEP 2: USER GOES TO TAB CONFIG AND CHANGES VALUES ===")
st.session_state["active_tab"] = "tab_config"
# User clicks preset Compact (6, 5, 4, N=20)
cfg = st.session_state["persistent_global_config"]
cfg['n_workers'] = 20
cfg['r_day'] = 6
cfg['r_eve'] = 5
cfg['r_night'] = 4
st.session_state["ui_cfg_n"] = 20
st.session_state["ui_cfg_rd"] = 6
st.session_state["ui_cfg_re"] = 5
st.session_state["ui_cfg_rn"] = 4

p2 = global_state.get_global_params()
print(f"After preset: persistent r_day={st.session_state['persistent_global_config']['r_day']}, n_workers={st.session_state['persistent_global_config']['n_workers']}")
assert st.session_state['persistent_global_config']['r_day'] == 6
assert st.session_state['persistent_global_config']['n_workers'] == 20

print("\n=== STEP 3: USER SWITCHES TO TAB 4 (Streamlit deletes unrendered ui_cfg_* keys) ===")
st.session_state["active_tab"] = "tab4"
for k in list(st.session_state.keys()):
    if k.startswith("ui_cfg_"):
        del st.session_state[k]

p3 = global_state.get_global_params()
print(f"On Tab 4: get_global_params returned r_day={p3['r_day']}, n_workers={p3['n_workers']}")
assert p3['r_day'] == 6
assert p3['n_workers'] == 20
print(f"On Tab 4: persistent cfg r_day={st.session_state['persistent_global_config']['r_day']}")
assert st.session_state['persistent_global_config']['r_day'] == 6

print("\n=== STEP 4: END OF TAB 4 RUN (Streamlit again prunes unrendered ui_cfg_* keys) ===")
for k in list(st.session_state.keys()):
    if k.startswith("ui_cfg_"):
        del st.session_state[k]

print("\n=== STEP 5: USER SWITCHES BACK TO TAB CONFIG ===")
st.session_state["active_tab"] = "tab_config"
global_state._sync_state_from_ui()
print(f"Back on tab_config: ui_cfg_rd={st.session_state.get('ui_cfg_rd')}, persistent r_day={st.session_state['persistent_global_config']['r_day']}")
assert st.session_state.get('ui_cfg_rd') == 6
assert st.session_state['persistent_global_config']['r_day'] == 6

p5 = global_state.get_global_params()
print(f"Final check: r_day={p5['r_day']}, n_workers={p5['n_workers']}")
assert p5['r_day'] == 6
assert p5['n_workers'] == 20
print("\n>>> SIMULATION COMPLETE: ALL ASSERTIONS PASSED! <<<")
