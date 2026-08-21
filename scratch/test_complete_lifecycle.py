import sys, os
sys.path.insert(0, os.path.abspath('.'))
import streamlit as st
import global_state

print("=" * 60)
print("TEST 1: FRESH START TEST")
print("=" * 60)
st.session_state.clear()
p_init = global_state.get_global_params()
print(f"Fresh start: n_workers={p_init['n_workers']}, r_day={p_init['r_day']}, r_eve={p_init['r_eve']}, r_night={p_init['r_night']}")
assert p_init['n_workers'] == 28, f"Expected 28, got {p_init['n_workers']}"
assert p_init['r_day'] == 9, f"Expected 9, got {p_init['r_day']}"
assert p_init['r_eve'] == 7, f"Expected 7, got {p_init['r_eve']}"
assert p_init['r_night'] == 5, f"Expected 5, got {p_init['r_night']}"

print("\n" + "=" * 60)
print("TEST 2: USER MANUALLY CHANGES VALUES ON GLOBAL TAB")
print("=" * 60)
# Simulate user typing into number input and sliders
st.session_state["w_r_day"] = 15
st.session_state["persistent_global_config"]['r_day'] = 15
st.session_state["w_workload_imb"] = 35
st.session_state["persistent_global_config"]['workload_imb'] = 35

p_edit = global_state.get_global_params()
print(f"After manual edit: r_day={p_edit['r_day']}, workload_imb={p_edit['weights']['workload_imb']}")
assert p_edit['r_day'] == 15
assert p_edit['weights']['workload_imb'] == 35

print("\n" + "=" * 60)
print("TEST 3: TAB SWITCH TO TAB 4 AND BACK")
print("=" * 60)
# Streamlit clears unrendered widget keys
for k in list(st.session_state.keys()):
    if k.startswith("w_"):
        del st.session_state[k]

p_tab4 = global_state.get_global_params()
print(f"On Tab 4: r_day={p_tab4['r_day']}, workload_imb={p_tab4['weights']['workload_imb']}")
assert p_tab4['r_day'] == 15
assert p_tab4['weights']['workload_imb'] == 35

# Returning to global config tab
global_state._init_persistent_config()
cfg = st.session_state["persistent_global_config"]
assert cfg['r_day'] == 15

print("\n" + "=" * 60)
print("TEST 4: USER CLICKS PRESET BUTTON (Kompakt Haddehane 6-5-4 | N:20)")
print("=" * 60)
# Simulate set_shift_preset(20, 6, 5, 4)
cfg['n_workers'] = 20
cfg['r_day'] = 6
cfg['r_eve'] = 5
cfg['r_night'] = 4
st.session_state["w_n_workers"] = 20
st.session_state["w_r_day"] = 6
st.session_state["w_r_eve"] = 5
st.session_state["w_r_night"] = 4

p_preset = global_state.get_global_params()
print(f"After preset click: n_workers={p_preset['n_workers']}, r_day={p_preset['r_day']}")
assert p_preset['n_workers'] == 20
assert p_preset['r_day'] == 6

print("\n" + "=" * 60)
print("TEST 5: TAB SWITCH TO TAB 6 AND RETURN TO GLOBAL TAB")
print("=" * 60)
for k in list(st.session_state.keys()):
    if k.startswith("w_"):
        del st.session_state[k]

p_tab6 = global_state.get_global_params()
print(f"On Tab 6: n_workers={p_tab6['n_workers']}, r_day={p_tab6['r_day']}")
assert p_tab6['n_workers'] == 20
assert p_tab6['r_day'] == 6

# Return to Global Config tab:
global_state._init_persistent_config()
cfg_return = st.session_state["persistent_global_config"]
print(f"Back on Global Tab: n_workers={cfg_return['n_workers']}, r_day={cfg_return['r_day']}")
assert cfg_return['n_workers'] == 20
assert cfg_return['r_day'] == 6

print("\n" + "=" * 60)
print(">>> ALL 5 LIFECYCLE TESTS PASSED WITH ZERO LOSS OF STATE! <<<")
print("=" * 60)
