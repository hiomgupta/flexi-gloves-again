import streamlit as st
import serial
import json
import time
import pandas as pd
import numpy as np
from collections import deque

# Import your *EDITED* mood detection logic
import mood_detector as md 

# --- Constants ---
# !!! CHANGE THIS TO YOUR ESP32's PORT !!!
SERIAL_PORT = "COM3" 
BAUD_RATE = 115200      # MUST MATCH THE ESP32 SCRIPT

# --- NEW TIMING CONSTANTS ---
# Your new script sends data 1 time per second (1Hz)
SAMPLING_RATE_HZ = 1
CALIBRATION_SECONDS = 180 # 3 minutes = 180 samples

# We need a longer window for 1Hz data
WINDOW_SECONDS = 30 # Use 30 seconds (30 samples)
HOP_SECONDS = 5     # Calculate mood every 5 seconds (5 samples)

# Calculate buffer sizes
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_SECONDS # 30 samples
HOP_SAMPLES = SAMPLING_RATE_HZ * HOP_SECONDS   # 5 samples
MAX_BUFFER_SIZE = SAMPLING_RATE_HZ * 120 # Store 2 min of data

# --- Helper Function to Connect (Unchanged) ---
@st.cache_resource
def get_serial_connection():
    """Caches the serial connection."""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        # --- NEW: Add logic to clear buffer and skip CSV header ---
        ser.flushInput()
        # Wait for the ESP32 to send its header line
        time.sleep(1.5) # Wait for the first data line
        header_line = ser.readline().decode('utf-8').strip()
        st.info(f"Skipped Header: {header_line}")
        return ser
    except serial.SerialException as e:
        st.error(f"Failed to connect to Serial port {SERIAL_PORT}: {e}")
        st.info("Check if the port is correct and no other program (like Arduino Serial Monitor) is using it.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

# --- Initialize Session State (Simplified) ---
if "state" not in st.session_state:
    st.session_state.state = "INITIAL" 
    st.session_state.baseline = None
    st.session_state.current_mood = "UNKNOWN"
    st.session_state.cal_start_time = 0
    # Simplified cal_data: no ibi, ppg, or gyro
    st.session_state.cal_data = {
        "hr": [], "acc": [], "temp": []
    }
    # Simplified buffers: no ppg
    st.session_state.buffers = {
        "hr": deque(maxlen=MAX_BUFFER_SIZE),
        "acc_x": deque(maxlen=MAX_BUFFER_SIZE), "acc_y": deque(maxlen=MAX_BUFFER_SIZE),
        "acc_z": deque(maxlen=MAX_BUFFER_SIZE), "temp": deque(maxlen=MAX_BUFFER_SIZE)
    }
    # Simplified window_buffer
    st.session_state.window_buffer = {
        "hr": [], "ibi": [], "ppg": [], "acc": [], "temp": [], "gyro": []
    }

# --- Streamlit UI (Unchanged) ---
st.set_page_config(page_title="Live Mood Detector", layout="wide")
st.title("🧠 Wearable Mood Detector (USB - CSV)")

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Status")
    status_placeholder = st.empty()
    st.header("Current Mood")
    mood_placeholder = st.empty()
    st.header("Raw Data (CSV)")
    data_placeholder = st.empty()
    
    if st.session_state.state == "INITIAL":
        if st.button("Start 3-Minute Calibration"):
            st.session_state.state = "CALIBRATING"
            st.session_state.cal_start_time = time.time()
            st.session_state.cal_data = {
                "hr": [], "acc": [], "temp": [] # Reset simplified data
            }
            st.rerun()

with col2:
    st.header("Live Sensor Data (1Hz)")
    hr_chart = st.empty()
    acc_chart = st.empty()
    temp_chart = st.empty()

# --- Main Logic Loop (HEAVILY EDITED for CSV) ---
try:
    ser = get_serial_connection()
    if not ser:
        st.stop()

    while True:
        # 1. Read and Parse CSV Data
        data_dict = {} # To show in "Raw Data"
        try:
            if not ser.is_open:
                st.error("Serial port disconnected. Please reconnect.")
                st.stop()
            
            if ser.in_waiting > 0:
                line_raw = ser.readline()
                line = line_raw.decode('utf-8').strip()
                
                if not line:
                    continue
                
                # --- NEW CSV PARSING LOGIC ---
                # Expected: Time(ms),HR,SpO2,Temp(C),Pressure(hPa),AccelX,AccelY,AccelZ
                parts = line.split(',')
                if len(parts) != 8:
                    st.warning(f"Bad CSV line, skipping: {line}")
                    continue
                
                # Convert parts to numbers
                hr_val = float(parts[1])
                spo2_val = float(parts[2])
                temp_val = float(parts[3])
                acc_x = float(parts[5])
                acc_y = float(parts[6])
                acc_z = float(parts[7])
                
                # Store for "Raw Data" display
                data_dict = {
                    "Time(ms)": float(parts[0]), "HR": hr_val, "SpO2": spo2_val,
                    "Temp(C)": temp_val, "Pressure(hPa)": float(parts[4]),
                    "AccelX": acc_x, "AccelY": acc_y, "AccelZ": acc_z
                }
                
                # Pack for mood detector (it expects 3D array for acc)
                acc_sample = [acc_x, acc_y, acc_z]
                
            else:
                # No data, just sleep briefly. Data only comes 1x/sec
                time.sleep(0.1)
                continue

        except serial.SerialException as e:
            st.error(f"Serial Error: {e}. Please check connection.")
            st.stop()
        except (ValueError, TypeError) as e:
            st.warning(f"Data conversion error, skipping line: {line} ({e})")
            continue
        except Exception as e:
            st.warning(f"Data read error: {e}")
            continue

        # 2. Append data to visualization buffers
        buffers = st.session_state.buffers
        buffers["hr"].append(hr_val)
        buffers["temp"].append(temp_val)
        buffers["acc_x"].append(acc_x)
        buffers["acc_y"].append(acc_y)
        buffers["acc_z"].append(acc_z)

        # 3. State Machine
        state = st.session_state.state

        if state == "CALIBRATING":
            elapsed = time.time() - st.session_state.cal_start_time
            progress = min(1.0, elapsed / CALIBRATION_SECONDS)
            status_placeholder.info(f"CALIBRATING... Stay calm and still.\n{int(elapsed)} / {CALIBRATION_SECONDS}s")
            st.progress(progress)
            
            # Append simplified data
            cal = st.session_state.cal_data
            cal["hr"].append(hr_val)
            cal["temp"].append(temp_val)
            cal["acc"].append(acc_sample)

            if elapsed >= CALIBRATION_SECONDS:
                status_placeholder.info("Calculating baseline...")
                # Call simplified baseline function
                st.session_state.baseline = md.compute_baselines(
                    hr_series=cal["hr"],
                    ibi_series=[], # Pass empty list
                    ppg_series=[], # Pass empty list
                    acc_series=np.array(cal["acc"]),
                    temp_series=cal["temp"],
                    gyro_series=None # Pass None
                )
                st.session_state.state = "RUNNING"
                status_placeholder.success("Calibration complete! Now running.")
                st.balloons()

        elif state == "RUNNING":
            status_placeholder.success("RUNNING")
            
            # Feed simplified data to window buffer
            win = st.session_state.window_buffer
            win["hr"].append(hr_val)
            win["acc"].append(acc_sample)
            win["temp"].append(temp_val)
            # ibi, ppg, gyro lists remain empty

            if len(win["hr"]) >= WINDOW_SAMPLES: # Check against 30 samples
                # Get 30 samples
                window_data = {
                    "hr": np.array(win["hr"][-WINDOW_SAMPLES:]),
                    "ibi": np.array([]), # Pass empty array
                    "ppg": np.array([]), # Pass empty array
                    "acc": np.array(win["acc"][-WINDOW_SAMPLES:]),
                    "temp": np.array(win["temp"][-WINDOW_SAMPLES:]),
                    "gyro": None
                }

                result = md.process_window(
                    window_data=window_data,
                    baseline=st.session_state.baseline
                )
                
                st.session_state.current_mood = result.get("classification", "ERROR")
                
                # Hop (remove oldest 5 samples)
                for k in ["hr", "acc", "temp"]:
                    win[k] = win[k][HOP_SAMPLES:]
            
        # 4. Update UI Elements (inside the loop)
        mood_map = {
            "STRESS": "😥", "CALM": "🧘", "AGITATED": "😠",
            "NEUTRAL": "🙂", "UNKNOWN": "❓", "ACTIVE / UNKNOWN": "🏃"
        }
        mood = st.session_state.current_mood
        mood_placeholder.markdown(f"## **{mood}** {mood_map.get(mood, '❓')}")
        
        # --- Create DataFrames for Charting (LABELS ADDED) ---
        sample_index = range(len(buffers["hr"]))
        
        hr_df = pd.DataFrame({"Heart Rate (BPM)": list(buffers["hr"])}, index=sample_index)
        hr_df.index.name = "Sample (1 per second)"
        
        acc_df = pd.DataFrame({
            "Acc_X": list(buffers["acc_x"]), "Acc_Y": list(buffers["acc_y"]), "Acc_Z": list(buffers["acc_z"])
        }, index=sample_index)
        acc_df.index.name = "Sample (1 per second)"
        
        temp_df = pd.DataFrame({"Env Temp (C)": list(buffers["temp"])}, index=sample_index)
        temp_df.index.name = "Sample (1 per second)"

        # --- Draw Charts with Titles (LABELS ADDED) ---
        hr_chart.line_chart(hr_df)
        acc_chart.line_chart(acc_df)
        temp_chart.line_chart(temp_df)
        
        if data_dict:
            data_placeholder.json(data_dict)

except Exception as e:
    st.error(f"A runtime error occurred: {e}")