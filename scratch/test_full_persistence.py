import sys, os
sys.path.insert(0, os.path.abspath('.'))
import streamlit as st
import global_state


# Clear session state for clean test
for k in list(st.session_state.keys()):
    del st.session_state[k]

# 1. First initial call
params1 = global_state.get_global_params()
print(f"Step 1 - Initial params: r_day={params1['r_day']}, r_eve={params1['r_eve']}, r_night={params1['r_night']}")
assert params1['r_day'] == 9

# 2. User changes values on Global Config tab
st.session_state['ui_cfg_rd'] = 14
st.session_state['ui_cfg_re'] = 10
st.session_state['ui_cfg_rn'] = 8
params2 = global_state.get_global_params()
print(f"Step 2 - After UI update: r_day={params2['r_day']}, r_eve={params2['r_eve']}, r_night={params2['r_night']}")
assert params2['r_day'] == 14
assert params2['r_eve'] == 10
assert params2['r_night'] == 8

# 3. User switches to Tab 4 -> Streamlit garbage collection removes all ui_cfg_* keys
ui_keys = [k for k in list(st.session_state.keys()) if k.startswith('ui_cfg_')]
for k in ui_keys:
    del st.session_state[k]
print(f"Step 3 - Switched to Tab 4, widget keys deleted by Streamlit: {len(ui_keys)} keys deleted")

# 4. Tab 4 gets global params
params_tab4 = global_state.get_global_params()
print(f"Step 4 - On Tab 4, params received: r_day={params_tab4['r_day']}, r_eve={params_tab4['r_eve']}, r_night={params_tab4['r_night']}")
assert params_tab4['r_day'] == 14
assert params_tab4['r_eve'] == 10
assert params_tab4['r_night'] == 8

# 5. User switches BACK to Global Config tab
global_state._sync_state_from_ui()
print(f"Step 5 - Switched back to Global Config: st.session_state['ui_cfg_rd']={st.session_state.get('ui_cfg_rd')}")
assert st.session_state.get('ui_cfg_rd') == 14
assert st.session_state.get('ui_cfg_re') == 10
assert st.session_state.get('ui_cfg_rn') == 8

params_final = global_state.get_global_params()
print(f"Step 6 - Final params verified: r_day={params_final['r_day']}, r_eve={params_final['r_eve']}, r_night={params_final['r_night']}")
assert params_final['r_day'] == 14

print("\n>>> ALL TESTS PASSED! PERSISTENCE ACROSS TAB SWITCHES IS 100% VERIFIED! <<<")
