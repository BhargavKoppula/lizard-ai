import time
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
import random
import cv2

# ---- Page Setup ----
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

# ---- Sidebar Layout ----
with st.sidebar:
    logo = Image.open("logo_cut-removebg-preview.png")
    st.image(logo, width=200)
    st.markdown("### ⏱️ Set Focus Time")

    if 'session_minutes' not in st.session_state:
        st.session_state.session_minutes = 10

    if st.button("2 Min"): st.session_state.session_minutes = 2
    if st.button("5 Min"): st.session_state.session_minutes = 5
    if st.button("15 Min"): st.session_state.session_minutes = 15

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("➖") and st.session_state.session_minutes > 1:
            st.session_state.session_minutes -= 1
    with col3:
        if st.button("➕") and st.session_state.session_minutes < 60:
            st.session_state.session_minutes += 1
    with col2:
        st.markdown(f"<h3 style='text-align:center; color:#00ffcc;'>{st.session_state.session_minutes} Minutes</h3>", unsafe_allow_html=True)

    start_btn = st.button("✅ Start Session")
    stop_btn = st.button("🛑 Stop Session")
    sidebar_timer = st.empty()

# ---- Main Content ----
st.image("logo_cut-removebg-preview.png", width=600)
st.markdown("<h1 class='title'>Focus Tracker</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Set up your focus session to stay on track and boost productivity</p>", unsafe_allow_html=True)

# ---- Session State Initialization ----
if 'running' not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True
    st.session_state.focus_start_time = time.time()
    st.session_state.focused_time = 0
    st.session_state.timestamps = []
    st.session_state.focus_times = []
    st.session_state.last_check_time = time.time()

elif stop_btn:
    st.session_state.running = False

# ---- Video Processing ----
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        self.drawing = mp.solutions.drawing_utils

    def transform(self, frame):
        img = frame.to_rgb()
        results = self.face_mesh.process(img)
        status = "Not Focused"
        now = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = [landmarks[i] for i in range(33, 39)]
            right_eye = [landmarks[i] for i in range(39, 45)]
            ear = (
                np.linalg.norm(np.array([left_eye[1].x, left_eye[1].y]) - np.array([left_eye[5].x, left_eye[5].y])) +
                np.linalg.norm(np.array([left_eye[2].x, left_eye[2].y]) - np.array([left_eye[4].x, left_eye[4].y])) +
                np.linalg.norm(np.array([right_eye[1].x, right_eye[1].y]) - np.array([right_eye[5].x, right_eye[5].y])) +
                np.linalg.norm(np.array([right_eye[2].x, right_eye[2].y]) - np.array([right_eye[4].x, right_eye[4].y]))
            ) / (2.0 * (
                np.linalg.norm(np.array([left_eye[0].x, left_eye[0].y]) - np.array([left_eye[3].x, left_eye[3].y])) +
                np.linalg.norm(np.array([right_eye[0].x, right_eye[0].y]) - np.array([right_eye[3].x, right_eye[3].y]))
            ))

            if ear > 0.2:
                status = "Focused"
                st.session_state.focused_time += now - st.session_state.last_check_time

        st.session_state.last_check_time = now
        st.session_state.timestamps.append(now - st.session_state.focus_start_time)
        st.session_state.focus_times.append(st.session_state.focused_time)

        img_bgr = frame.to_ndarray(format="bgr24")
        cv2.putText(img_bgr, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if status == "Focused" else (0, 0, 255), 2)
        return img_bgr

# ---- Main Logic ----
if st.session_state.running:
    webrtc_streamer(key="stream", video_transformer_factory=VideoTransformer)
    timer_area = st.empty()
    result_area = st.empty()

    while time.time() - st.session_state.focus_start_time < st.session_state.session_minutes * 60:
        remaining = int(st.session_state.session_minutes * 60 - (time.time() - st.session_state.focus_start_time))
        mins, secs = divmod(remaining, 60)
        timer_area.markdown(f"### ⏳ Remaining Time: {mins}m {secs}s")
        sidebar_timer.markdown(f"#### ⏳ Time Left: {mins}m {secs}s")
        time.sleep(1)
        if not st.session_state.running:
            break

    st.session_state.running = False
    total_time = time.time() - st.session_state.focus_start_time
    focus_percent = (st.session_state.focused_time / total_time) * 100 if total_time > 0 else 0

    st.markdown("<div class='metric'>✅ Session Completed!</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'>🧠 Total Focused Time: {int(st.session_state.focused_time)} seconds</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'>📊 Focus Percentage: {focus_percent:.2f}%</div>", unsafe_allow_html=True)

    # Feedback messages
    if focus_percent >= 80:
        st.success(random.choice([
            "🌟 Amazing! You maintained strong focus throughout.",
            "🚀 You're in the zone! Great session.",
            "🧠 Excellent control and dedication!"
        ]))
    elif focus_percent >= 50:
        st.info(random.choice([
            "👍 Good effort! A bit more consistency and you're there.",
            "👀 On and off—try removing distractions.",
            "🔄 Nice! Aim to stay locked in a bit longer."
        ]))
    else:
        st.warning(random.choice([
            "😕 Tough session? Next one will be better!",
            "😴 Focus slipping? Try breaks or environment changes.",
            "📵 Turn off distractions and try again!"
        ]))
        # st.warning(message)
