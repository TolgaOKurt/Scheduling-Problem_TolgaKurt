import sys, os
sys.path.insert(0, os.path.abspath('.'))
import streamlit as st

# Test Streamlit's widget instantiation logic with session state
st.session_state.clear()

# 1. Init session state before widget
st.session_state["my_num"] = 15

# 2. Render widget with key only (no value)
# In Streamlit, when key is in session_state, it takes the value from session_state!
val = st.session_state["my_num"]
print("Initial value from session_state:", val)
assert val == 15

# 3. Simulate preset change
st.session_state["my_num"] = 6
print("After preset changed session_state:", st.session_state["my_num"])
assert st.session_state["my_num"] == 6

# 4. Simulate tab switch: key removed by Streamlit GC
del st.session_state["my_num"]

# 5. Restore before rendering tab again from persistent dict
persistent_store = {"my_num": 6}
if "my_num" not in st.session_state:
    st.session_state["my_num"] = persistent_store["my_num"]

print("After tab switch restore:", st.session_state["my_num"])
assert st.session_state["my_num"] == 6
print("SUCCESS!")
