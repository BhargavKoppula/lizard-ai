import cv2
import time
import numpy as np
import streamlit as st
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import random
matplotlib.use('Agg')

# Streamlit Page Config
st.set_page_config(page_title="Lizard - Focus Tracker", layout="wide")
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

# Layout with Sidebar
with st.sidebar:
    logo_path = "logo_cut-removebg-preview.png"
    logo = Image.open(logo_path)
    st.image(logo, width=200)

    st.markdown("### ⏱️ Set Focus Time")
    if 'session_minutes' not in st.session_state:
        st.session_state.session_minutes = 10

    st.markdown("#### 🔘 Quick Set Time")
    if st.button("2 Min"):
        st.session_state.session_minutes = 2
    if st.button("5 Min"):
        st.session_state.session_minutes = 5
    if st.button("15 Min"):
        st.session_state.session_minutes = 15

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

    start_btn = st.button("✅ Start Session", key="start")
    # pause_btn = st.button("⏸️ Pause Session", key="pause")
    # resume_btn = st.button("▶️ Resume Session", key="resume")
    stop_btn = st.button("🛑 Stop Session", key="stop")

    sidebar_timer = st.empty()

logo_large = Image.open("logo_cut-removebg-preview.png")
st.image(logo_large, width=600)
st.markdown("<h1 class='title'>Focus Tracker</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Set up your focus session to stay on track and boost productivity</p>", unsafe_allow_html=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

if 'running' not in st.session_state:
    st.session_state['running'] = False
# if 'paused' not in st.session_state:
#     st.session_state['paused'] = False

if start_btn:
    st.session_state['running'] = True
    # st.session_state['paused'] = False
    st.session_state['focus_start_time'] = time.time()
    st.session_state['focused_time'] = 0
    st.session_state['focus_times'] = []
    st.session_state['timestamps'] = []
    st.session_state['last_check_time'] = st.session_state['focus_start_time']

# elif pause_btn:
#     st.session_state['paused'] = True
# elif resume_btn:
#     st.session_state['paused'] = False
elif stop_btn:
    st.session_state['running'] = False

if st.session_state['running']:
    cap = cv2.VideoCapture(0)
    frame_spot = st.empty()
    time_spot = st.empty()
    metric_spot = st.empty()

    while time.time() - st.session_state['focus_start_time'] < st.session_state.session_minutes * 60:
        # if st.session_state['paused']:
        #     time.sleep(1)
        #     continue

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
                ear = (np.linalg.norm(np.array([left_eye[1].x, left_eye[1].y]) - np.array([left_eye[5].x, left_eye[5].y])) +
                       np.linalg.norm(np.array([left_eye[2].x, left_eye[2].y]) - np.array([left_eye[4].x, left_eye[4].y])) +
                       np.linalg.norm(np.array([right_eye[1].x, right_eye[1].y]) - np.array([right_eye[5].x, right_eye[5].y])) +
                       np.linalg.norm(np.array([right_eye[2].x, right_eye[2].y]) - np.array([right_eye[4].x, right_eye[4].y]))
                       ) / (2.0 * (np.linalg.norm(np.array([left_eye[0].x, left_eye[0].y]) - np.array([left_eye[3].x, left_eye[3].y])) +
                               np.linalg.norm(np.array([right_eye[0].x, right_eye[0].y]) - np.array([right_eye[3].x, right_eye[3].y]))))

                if ear > 0.2:
                    status = "Focused"
                    st.session_state['focused_time'] += now - st.session_state['last_check_time']

                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        st.session_state['last_check_time'] = now

        label = f"{status}"
        color = (0, 255, 0) if status == "Focused" else (0, 0, 255)
        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        st.session_state['timestamps'].append(now - st.session_state['focus_start_time'])
        st.session_state['focus_times'].append(st.session_state['focused_time'])

        frame_spot.image(frame, channels="BGR")
        remaining = int(st.session_state.session_minutes * 60 - (now - st.session_state['focus_start_time']))
        time_spot.markdown(f"### ⏳ Remaining Time: {remaining // 60}m {remaining % 60}s")
        sidebar_timer.markdown(f"#### ⏳ Time Left: {remaining // 60}m {remaining % 60}s")

        if stop_btn:
            st.session_state['running'] = False
            break

    cap.release()

    st.markdown("<div class='metric'>✅ Session Completed!</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'>🧠 Total Focused Time: {int(st.session_state['focused_time'])} seconds</div>", unsafe_allow_html=True)

    total = time.time() - st.session_state['focus_start_time']
    percent = (st.session_state['focused_time'] / total) * 100 if total > 0 else 0
    st.markdown(f"<div class='metric'>📊 Focus Percentage: {percent:.2f}%</div>", unsafe_allow_html=True)
    if percent >= 80:
        message = random.choice([
            "🌟 Amazing! You maintained strong focus throughout. Keep this up!",
            "🚀 You're in the zone! That was a great session.",
            "🧠 Excellent control and dedication! You're building deep work muscles."
        ])
        st.success(message)
    elif percent >= 50:
        message = random.choice([
            "👍 Good effort! A little more consistency and you'll nail it.",
            "👀 You were on and off—try to remove distractions for next time.",
            "🔄 Nice session. Next time, aim to stay locked in a bit longer."
        ])
        st.info(message)
    else:
        message = random.choice([
            "😕 Tough session? It's okay—next one will be better!",
            "😴 Focus was slipping. Try changing your environment or taking breaks.",
            "📵 Consider turning off distractions and try again!"
        ])
        st.warning(message)

    #ploting metrics

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e1e1e')
    ax.plot(st.session_state['timestamps'], st.session_state['focus_times'], color='#00ffcc', linewidth=2)
    ax.set_facecolor('#1e1e1e')
    ax.set_title("Focus Over Time", fontsize=14, color='white')
    ax.set_xlabel("Time (s)", color='white')
    ax.set_ylabel("Focused Time (s)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='#333')
    st.pyplot(fig)

