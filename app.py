import cv2
import time
import numpy as np
import streamlit as st
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
matplotlib.use('Agg')

# Streamlit Page Config
st.set_page_config(page_title="Lizard AI - Focus Tracker", layout="centered")
st.markdown("""
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .main {
        background-color: #1e1e1e;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0px 4px 12px rgba(255, 255, 255, 0.1);
    }
    .title {
        color: #00ffcc;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .subtitle {
        color: #aaaaaa;
        font-size: 1.25rem;
    }
    .metric {
        background: #2a2a2a;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        color: #ffffff;
    }
    .stButton > button {
        width: 100%;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Add Logo from Local File
logo_path = "Artboard_1-100-removebg-preview.png"  # Replace with your local image file name
logo = Image.open(logo_path)
st.image(logo, width=200)

# UI Title
st.markdown("<h1 class='title'>Lizard - Focus Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Set up your focus session to stay on track and boost productivity</p>", unsafe_allow_html=True)

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

# Eye Aspect Ratio

def eye_aspect_ratio(eye):
    A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
    B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
    C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
    return (A + B) / (2.0 * C)

# Custom Time Setter
st.markdown("### ⏱️ Set Focus Time")
if 'session_minutes' not in st.session_state:
    st.session_state.session_minutes = 10
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("➖"):
        if st.session_state.session_minutes > 1:
            st.session_state.session_minutes -= 1
with col3:
    if st.button("➕"):
        if st.session_state.session_minutes < 60:
            st.session_state.session_minutes += 1
with col2:
    st.markdown(f"<h3 style='text-align:center; color:#00ffcc;'>{st.session_state.session_minutes} Minutes</h3>", unsafe_allow_html=True)

# Start / Stop Buttons Side by Side
colA, colB = st.columns(2)
with colA:
    start_btn = st.button("✅ Start Session", key="start", use_container_width=True)
with colB:
    stop_btn = st.button("🛑 Stop Session", key="stop", use_container_width=True)

# Initialize variables
focused_time = 0
focus_start_time = None
focus_times = []
timestamps = []
last_check_time = time.time()

# Detection
if start_btn:
    st.session_state["running"] = True
    focus_start_time = time.time()
    focus_times.clear()
    timestamps.clear()
    focused_time = 0

if stop_btn:
    st.session_state["running"] = False

if st.session_state.get("running", False):
    frame_spot = st.empty()
    time_spot = st.empty()
    metric_spot = st.empty()

    while time.time() - focus_start_time < st.session_state.session_minutes * 60:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        status = "Not Focused"
        now = time.time()

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                left_eye = [landmarks.landmark[i] for i in range(33, 39)]
                right_eye = [landmarks.landmark[i] for i in range(39, 45)]
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                if ear > 0.2:
                    status = "Focused"
                    focused_time += now - last_check_time

                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        last_check_time = now

        # Overlay label
        label = f"{status}"
        color = (0, 255, 0) if status == "Focused" else (0, 0, 255)
        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        timestamps.append(now - focus_start_time)
        focus_times.append(focused_time)

        frame_spot.image(frame, channels="BGR")
        remaining = int(st.session_state.session_minutes * 60 - (now - focus_start_time))
        time_spot.markdown(f"### ⏳ Remaining Time: {remaining // 60}m {remaining % 60}s")

        if stop_btn:
            st.session_state["running"] = False
            break

    # Final stats
    cap.release()
    cv2.destroyAllWindows()

    st.markdown("<div class='metric'>✅ Session Completed!</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'>🧠 Total Focused Time: {int(focused_time)} seconds</div>", unsafe_allow_html=True)

    # Focus %
    total = time.time() - focus_start_time
    percent = (focused_time / total) * 100 if total > 0 else 0
    st.markdown(f"<div class='metric'>📊 Focus Percentage: {percent:.2f}%</div>", unsafe_allow_html=True)

    # Plot
    fig, ax = plt.subplots(facecolor = '#1e1e1e')
    ax.plot(timestamps, focus_times, color='#00ffcc', linewidth=2)
    ax.set_facecolor('#1e1e1e')
    ax.set_title("Focus Over Time", fontsize=14, color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Focused Time (s)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='#333')
    st.pyplot(fig)
