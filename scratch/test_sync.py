"""
Test script to simulate Streamlit session state and global state transitions
"""

GLOBAL_DEFAULTS = {
    'n_workers': 28,
    'n_days': 7,
    'r_day': 9,
    'r_eve': 7,
    'r_night': 5,
    'penalties': {
        'circadian': 50,
        'night_imb': 25,
        'exp_mix': 60,
        'pref_off': 40,
        'posta': 15
    },
    'kadro_mode': "Düzenli Kadro",
    'rand_seed': 42,
    'live_stream_enabled': True,
    'live_stream_interval': 50
}

session_state = {}

def init_persistent():
    if "persistent_global_config" not in session_state:
        session_state["persistent_global_config"] = {
            'n_workers': GLOBAL_DEFAULTS['n_workers'],
            'n_days': GLOBAL_DEFAULTS['n_days'],
            'r_day': GLOBAL_DEFAULTS['r_day'],
            'r_eve': GLOBAL_DEFAULTS['r_eve'],
            'r_night': GLOBAL_DEFAULTS['r_night'],
            'circadian': GLOBAL_DEFAULTS['penalties']['circadian'],
            'night_imb': GLOBAL_DEFAULTS['penalties']['night_imb'],
            'exp_mix': GLOBAL_DEFAULTS['penalties']['exp_mix'],
            'pref_off': GLOBAL_DEFAULTS['penalties']['pref_off'],
            'posta': GLOBAL_DEFAULTS['penalties']['posta'],
            'kadro_mode': GLOBAL_DEFAULTS['kadro_mode'],
            'rand_seed': GLOBAL_DEFAULTS['rand_seed'],
            'live_stream_enabled': GLOBAL_DEFAULTS['live_stream_enabled'],
            'live_stream_interval': GLOBAL_DEFAULTS['live_stream_interval']
        }

def sync_state():
    init_persistent()
    cfg = session_state["persistent_global_config"]
    mapping = {
        'n_workers': 'ui_cfg_n',
        'n_days': 'ui_cfg_d',
        'r_day': 'ui_cfg_rd',
        'r_eve': 'ui_cfg_re',
        'r_night': 'ui_cfg_rn',
        'circadian': 'ui_cfg_wc',
        'night_imb': 'ui_cfg_wn',
        'exp_mix': 'ui_cfg_we',
        'pref_off': 'ui_cfg_wp',
        'posta': 'ui_cfg_wposta',
        'kadro_mode': 'ui_cfg_km',
        'rand_seed': 'ui_cfg_seed',
        'live_stream_enabled': 'ui_cfg_live_enabled',
        'live_stream_interval': 'ui_cfg_live_interval'
    }
    for cfg_key, ui_key in mapping.items():
        if ui_key in session_state:
            cfg[cfg_key] = session_state[ui_key]
        else:
            session_state[ui_key] = cfg[cfg_key]

# 1. First run on tab_config
sync_state()
print("Run 1 (initial): r_day =", session_state["persistent_global_config"]['r_day'])

# 2. User changes r_day to 12
session_state['ui_cfg_rd'] = 12
sync_state()
print("User changes r_day to 12: persistent r_day =", session_state["persistent_global_config"]['r_day'])

# 3. User switches to Tab 4 (Streamlit widget GC runs, deletes all ui_cfg_* keys)
keys_to_delete = [k for k in list(session_state.keys()) if k.startswith('ui_cfg_')]
for k in keys_to_delete:
    del session_state[k]

print("On Tab 4: ui_cfg keys deleted. Persistent r_day =", session_state["persistent_global_config"]['r_day'])

# 4. User switches back to Tab Config
sync_state()
print("Switched back to tab_config: ui_cfg_rd restored to:", session_state['ui_cfg_rd'])
print("Persistent r_day:", session_state["persistent_global_config"]['r_day'])
assert session_state["persistent_global_config"]['r_day'] == 12
assert session_state['ui_cfg_rd'] == 12
print("TEST PASSED!")
