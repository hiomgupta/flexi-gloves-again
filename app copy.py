import streamlit as st
import socket
import json
import time
import pandas as pd
import numpy as np
from collections import deque

# Import your own mood detection logic
import mood_detector as md 

# --- Constants ---
UDP_IP = "0.0.0.0"      # Listen on all available network interfaces
UDP_PORT = 12345        # MUST MATCH THE ESP32 SCRIPT
CALIBRATION_SECONDS = 180 
SAMPLING_RATE_HZ = 25     
WINDOW_SECONDS = 8
HOP_SECONDS = 2           

# Calculate buffer sizes
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_SECONDS
HOP_SAMPLES = SAMPLING_RATE_HZ * HOP_SECONDS
MAX_BUFFER_SIZE = SAMPLING_RATE_HZ * 60 # Store 1 min of data

# --- Helper Function to Connect ---
@st.cache_resource
def get_udp_socket():
    """Caches the UDP socket to prevent re-opening."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, UDP_PORT))
        # *** THIS IS THE MOST IMPORTANT LINE FOR STREAMLIT ***
        # It prevents the app from freezing while waiting for data.
        sock.setblocking(False) 
        return sock
    except OSError as e:
        if e.errno == 10048: # WinError 10048: Address already in use
             st.error(f"PORT {UDP_PORT} is already in use. Please close other apps or restart the kernel.")
        else:
            st.error(f"Failed to bind to port {UDP_PORT}: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

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

# --- Streamlit UI (Same as before) ---
st.set_page_config(page_title="Live Mood Detector", layout="wide")
st.title("🧠 Wearable Mood Detector (Wi-Fi)")

col1, col2 = st.columns([1, 2])
with col1:
    st.header("Status")
    status_placeholder = st.empty()
    st.header("Current Mood")
    mood_placeholder = st.empty()
    st.header("Raw Data")
    data_placeholder = st.empty()
    
    if st.session_state.state == "INITIAL":
        if st.button("Start 3-Minute Calibration"):
            st.session_state.state = "CALIBRATING"
            st.session_state.cal_start_time = time.time()
            st.session_state.cal_data = {
                "hr": [], "ibi": [], "ppg": [], "acc": [], "temp": [], "gyro": []
            }
            st.rerun()

with col2:
    st.header("Live Sensor Data")
    hr_chart = st.empty()
    acc_chart = st.empty()
    temp_chart = st.empty()

# --- Main Logic Loop (Updated for UDP) ---
try:
    sock = get_udp_socket()
    if not sock:
        st.stop()

    while True:
        # 1. Read and Parse Data
        try:
            # Try to receive data. 1024 is the buffer size.
            data_raw, addr = sock.recvfrom(1024)
            line = data_raw.decode('utf-8').strip()
            data = json.loads(line)

        except socket.error as e:
            # This error (10035 or 11) is *expected*. 
            # It just means "no data waiting right now".
            # We just ignore it and let Streamlit loop.
            err_num = e.errno
            if err_num == 10035 or err_num == 11: # WinError 10035 or errno.EWOULDBLOCK
                time.sleep(0.01) # Sleep briefly to prevent high CPU use
                continue
            else:
                st.warning(f"Socket Error: {e}")
                continue
        except json.JSONDecodeError:
            # Incomplete or corrupted packet, skip it
            continue
        except Exception as e:
            st.warning(f"Data read error: {e}")
            continue

        # --- From here on, the logic is IDENTICAL to the Serial version ---

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
            status_placeholder.info(f"CALIBRATING... Stay calm and still.\n{int(elapsed)} / {CALIBRATION_SECONDS}s")
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
                status_placeholder.success("Calibration complete! Now running.")
                st.balloons()

        elif state == "RUNNING":
            status_placeholder.success("RUNNING")
            
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
        
        hr_df = pd.DataFrame({"Heart Rate (BPM)": buffers["hr"]})
        acc_df = pd.DataFrame({
            "Acc_X": buffers["acc_x"], "Acc_Y": buffers["acc_y"], "Acc_Z": buffers["acc_z"]
        })
        temp_df = pd.DataFrame({"Env Temp (C)": buffers["temp"]})

        hr_chart.line_chart(hr_df)
        acc_chart.line_chart(acc_df)
        temp_chart.line_chart(temp_df)
        data_placeholder.json(data)

except Exception as e:
    st.error(f"A runtime error occurred: {e}")