import streamlit as st
# import serial # We don't need serial for the dry run
import json
import time
import pandas as pd
import numpy as np
from collections import deque

# Import your own mood detection logic
import mood_detector as md 

# --- Constants ---
# SERIAL_PORT = "COM3"  # No serial port needed
# BAUD_RATE = 115200 

CALIBRATION_SECONDS = 180 
SAMPLING_RATE_HZ = 25     # Approx 25Hz.
WINDOW_SECONDS = 8
HOP_SECONDS = 2           

# Calculate buffer sizes
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_SECONDS
HOP_SAMPLES = SAMPLING_RATE_HZ * HOP_SECONDS
MAX_BUFFER_SIZE = SAMPLING_RATE_HZ * 60 # Store 1 min of data

# --- Helper Function to Connect (REMOVED) ---
# We don't need to connect to a real device
# @st.cache_resource
# def get_serial_connection():
#     ...

# --- NEW: Fake Data Generator (BUG FIXED) ---
def generate_fake_data(state="CALM"):
    """Generates a single fake JSON data packet."""
    
    # Base values (This is what calibration will learn)
    hr_base = 70
    ibi_base = 850
    temp_base = 28.0
    ppg_base = 30000
    
    # Simulate different states
    if state == "CALIBRATING":
        hr = hr_base + np.random.randn() * 2
        ibi = ibi_base + np.random.randn() * 10 # Normal HRV
        ppg = ppg_base + np.sin(time.time() * hr / 60 * 2 * np.pi) * 1000 + np.random.randn() * 50
        acc_x = np.random.randn() * 0.05
        acc_y = np.random.randn() * 0.05
        acc_z = 9.8 + np.random.randn() * 0.05
        gyro_x = np.random.randn() * 0.1
        temp = temp_base + np.random.randn() * 0.1

    elif state == "CALM":
        # Data must be *different* from baseline to trigger rules
        hr = hr_base - 5 + np.random.randn() * 2 # Lower HR (HR_delta ~ -5)
        ibi = ibi_base + 50 + np.random.randn() * 20 # Higher IBI, Higher HRV (HRV_ratio > 1.15)
        ppg = ppg_base + np.sin(time.time() * hr / 60 * 2 * np.pi) * 1100 + np.random.randn() * 50 # Higher amplitude (PAV_ratio > 1.05)
        acc_x = np.random.randn() * 0.02 # Very still
        acc_y = np.random.randn() * 0.02
        acc_z = 9.8 + np.random.randn() * 0.02
        gyro_x = np.random.randn() * 0.05
        temp = temp_base + np.random.randn() * 0.1

    elif state == "STRESS":
        # Data must be *different* from baseline
        hr = hr_base + 15 + np.random.randn() * 3 # Higher HR (HR_delta ~ +15)
        ibi = ibi_base - 150 + np.random.randn() * 5 # Lower IBI, Lower HRV (HRV_ratio < 0.75)
        ppg = ppg_base + np.sin(time.time() * hr / 60 * 2 * np.pi) * 800 + np.random.randn() * 50 # Lower amplitude (PAV_ratio < 0.90)
        acc_x = np.random.randn() * 0.1 # A little fidgety
        acc_y = np.random.randn() * 0.1
        acc_z = 9.8 + np.random.randn() * 0.1
        gyro_x = np.random.randn() * 0.5 # More fidgety
        temp = temp_base - 0.2 + np.random.randn() * 0.1 # Lower temp (TEMP_delta < -0.15)

    else: # ACTIVE / UNKNOWN
        hr = hr_base + 30 + np.random.randn() * 5
        ibi = ibi_base - 300 + np.random.randn() * 15
        ppg = ppg_base + np.sin(time.time() * hr / 60 * 2 * np.pi) * 500 + np.random.randn() * 50
        acc_x = np.random.randn() * 1.5 # Walking
        acc_y = np.random.randn() * 2.0
        acc_z = 9.8 + np.random.randn() * 1.5
        gyro_x = np.random.randn() * 2.0
        temp = temp_base + 0.5 + np.random.randn() * 0.1
        
    data = {
        "timestamp": int(time.time() * 1000),
        "hr": hr,
        "spo2": 98.0 + np.random.randn() * 0.5,
        "ibi": ibi,
        "ppg": ppg,
        "acc": {
            "x": acc_x,
            "y": acc_y,
            "z": acc_z
        },
        "gyro": {
            "x": gyro_x,
            "y": np.random.randn() * 0.1,
            "z": np.random.randn() * 0.1
        },
        "temp": temp,
        "pressure": 1013.25 + np.random.randn() * 0.5
    }
    return data

# --- Initialize Session State (Same as before) ---
if "state" not in st.session_state:
    st.session_state.state = "INITIAL" 
    st.session_state.baseline = None
    st.session_state.current_mood = "UNKNOWN"
    st.session_state.cal_start_time = 0
    st.session_state.cal_data = {
        "hr": [], "ibi": [], "ppg": [], "acc": [], "temp": [], "gyro": []
    }
    st.session_state.buffers = {
        "hr": deque(maxlen=MAX_BUFFER_SIZE), "ppg": deque(maxlen=MAX_BUFFER_SIZE),
        "acc_x": deque(maxlen=MAX_BUFFER_SIZE), "acc_y": deque(maxlen=MAX_BUFFER_SIZE),
        "acc_z": deque(maxlen=MAX_BUFFER_SIZE), "temp": deque(maxlen=MAX_BUFFER_SIZE)
    }
    st.session_state.window_buffer = {
        "hr": [], "ibi": [], "ppg": [], "acc": [], "temp": [], "gyro": []
    }

# --- Streamlit UI ---
st.set_page_config(page_title="Live Mood Detector", layout="wide")
st.title("🧠 Wearable Mood Detector (DRY RUN SIMULATOR)") 

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Status")
    status_placeholder = st.empty()
    st.header("Current Mood")
    mood_placeholder = st.empty()
    st.header("Raw Data (Simulated)")
    data_placeholder = st.empty()
    
    if st.session_state.state == "INITIAL":
        if st.button("Start 3-Minute Calibration"):
            st.session_state.state = "CALIBRATING"
            st.session_state.cal_start_time = time.time()
            st.session_state.cal_data = {
                "hr": [], "ibi": [], "ppg": [], "acc": [], "temp": [], "gyro": []
            }
            st.rerun()

    # --- NEW: Radio button to test different moods ---
    if st.session_state.state == "RUNNING":
        st.session_state.sim_state = st.radio(
            "Simulate Mood:",
            ("CALM", "STRESS", "ACTIVE / UNKNOWN"),
            key="sim_state_radio"
        )
    
with col2:
    st.header("Live Sensor Data (Simulated)")
    hr_chart = st.empty()
    acc_chart = st.empty()
    temp_chart = st.empty()

# --- Main Logic Loop (CHANGED for Simulator) ---
try:
    # We don't need to connect, just loop
    # ser = get_serial_connection() 
    # if not ser:
    #     st.stop()

    while True:
        # 1. Read and Parse Data (CHANGED)
        try:
            # We generate data instead of reading it
            if st.session_state.state == "CALIBRATING":
                data = generate_fake_data(state="CALIBRATING") # Use special calibration state
            elif st.session_state.state == "RUNNING":
                data = generate_fake_data(state=st.session_state.sim_state)
            else:
                data = generate_fake_data(state="CALM") # Default
            
            # Simulate the 25Hz sample rate
            time.sleep(1.0 / SAMPLING_RATE_HZ) # sleep for 40ms

        except Exception as e:
            st.warning(f"Data generation error: {e}")
            continue

        # --- From here on, the logic is IDENTICAL ---

        # 2. Append data to visualization buffers
        buffers = st.session_state.buffers
        buffers["hr"].append(data.get("hr", np.nan))
        buffers["ppg"].append(data.get("ppg", np.nan))
        buffers["temp"].append(data.get("temp", np.nan))
        
        acc = data.get("acc", {"x":np.nan, "y":np.nan, "z":np.nan})
        gyro = data.get("gyro", {"x":np.nan, "y":np.nan, "z":np.nan})
        buffers["acc_x"].append(acc.get("x"))
        buffers["acc_y"].append(acc.get("y"))
        buffers["acc_z"].append(acc.get("z"))
        
        acc_sample = [acc.get("x"), acc.get("y"), acc.get("z")]
        gyro_sample = [gyro.get("x"), gyro.get("y"), gyro.get("z")]

        # 3. State Machine
        state = st.session_state.state

        if state == "CALIBRATING":
            elapsed = time.time() - st.session_state.cal_start_time
            progress = min(1.0, elapsed / CALIBRATION_SECONDS)
            status_placeholder.info(f"CALIBRATING... Simulating calm state.\n{int(elapsed)} / {CALIBRATION_SECONDS}s")
            st.progress(progress)
            
            cal = st.session_state.cal_data
            cal["hr"].append(data.get("hr", np.nan))
            cal["ibi"].append(data.get("ibi", np.nan))
            cal["ppg"].append(data.get("ppg", np.nan))
            cal["acc"].append(acc_sample)
            cal["temp"].append(data.get("temp", np.nan))
            cal["gyro"].append(gyro_sample)

            if elapsed >= CALIBRATION_SECONDS:
                status_placeholder.info("Calculating baseline...")
                st.session_state.baseline = md.compute_baselines(
                    hr_series=cal["hr"], ibi_series=cal["ibi"], ppg_series=cal["ppg"],
                    acc_series=np.array(cal["acc"]), temp_series=cal["temp"], gyro_series=np.array(cal["gyro"])
                )
                st.session_state.state = "RUNNING"
                st.session_state.sim_state = "CALM" # Default to calm
                status_placeholder.success("Calibration complete! Now running.")
                st.balloons()
                st.rerun() # Rerun to show the radio buttons

        elif state == "RUNNING":
            status_placeholder.success(f"RUNNING (Simulating: {st.session_state.sim_state})")
            
            win = st.session_state.window_buffer
            win["hr"].append(data.get("hr", np.nan))
            win["ibi"].append(data.get("ibi", np.nan))
            win["ppg"].append(data.get("ppg", np.nan))
            win["acc"].append(acc_sample)
            win["temp"].append(data.get("temp", np.nan))
            win["gyro"].append(gyro_sample)

            if len(win["hr"]) >= WINDOW_SAMPLES:
                window_data = {
                    "hr": np.array(win["hr"][-WINDOW_SAMPLES:]),
                    "ibi": np.array(win["ibi"][-WINDOW_SAMPLES:]),
                    "ppg": np.array(win["ppg"][-WINDOW_SAMPLES:]),
                    "acc": np.array(win["acc"][-WINDOW_SAMPLES:]),
                    "temp": np.array(win["temp"][-WINDOW_SAMPLES:]),
                    "gyro": np.array(win["gyro"][-WINDOW_SAMPLES:])
                }

                result = md.process_window(
                    window_data=window_data,
                    baseline=st.session_state.baseline
                )
                
                st.session_state.current_mood = result.get("classification", "ERROR")
                
                for k in win:
                    win[k] = win[k][HOP_SAMPLES:]
            
        # 4. Update UI Elements (inside the loop)
        mood_map = {
            "STRESS": "😥", "CALM": "🧘", "AGITATED": "😠",
            "NEUTRAL": "🙂", "UNKNOWN": "❓", "ACTIVE / UNKNOWN": "🏃"
        }
        mood = st.session_state.current_mood
        mood_placeholder.markdown(f"## **{mood}** {mood_map.get(mood, '❓')}")
        
        # --- Create DataFrames for Charting (LABELS ADDED) ---
        # Create an index for the x-axis
        sample_index = range(len(buffers["hr"]))
        
        hr_df = pd.DataFrame({"Heart Rate (BPM)": list(buffers["hr"])}, index=sample_index)
        hr_df.index.name = "Sample Index" # This will be the x-axis title
        
        acc_df = pd.DataFrame({
            "Acc_X": list(buffers["acc_x"]), # These become the legend
            "Acc_Y": list(buffers["acc_y"]), 
            "Acc_Z": list(buffers["acc_z"])
        }, index=sample_index)
        acc_df.index.name = "Sample Index" # x-axis title
        
        temp_df = pd.DataFrame({"Env Temp (C)": list(buffers["temp"])}, index=sample_index)
        temp_df.index.name = "Sample Index" # x-axis title

        # --- Draw Charts with Titles (LABELS ADDED) ---
        hr_chart.line_chart(hr_df) # Title is inferred from y-axis label
        acc_chart.line_chart(acc_df) # Title inferred
        temp_chart.line_chart(temp_df) # Title inferred
        
        data_placeholder.json(data)

except Exception as e:
    st.error(f"A runtime error occurred: {e}")