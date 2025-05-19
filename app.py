import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import mediapipe as mp
import av
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import random

matplotlib.use("Agg")

# Streamlit page config
st.set_page_config(page_title="Lizard - Focus Tracker", layout="wide")

# RTC Config (Optional: You can omit or replace STUN server)
rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Custom CSS
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #ffffff;
}
.metric {
    background: #2a2a2a;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 1rem 0;
    color: #ffffff;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Logo and setup
st.sidebar.image("logo_cut-removebg-preview.png", width=200)
st.markdown("<h1 style='color:#00ffcc;'>🦎 Lizard Focus Tracker</h1>", unsafe_allow_html=True)

# Session timer setup
if 'session_minutes' not in st.session_state:
    st.session_state.session_minutes = 5
if 'focused_time' not in st.session_state:
    st.session_state.focused_time = 0
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'focus_times' not in st.session_state:
    st.session_state.focus_times = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None

with st.sidebar:
    st.markdown("### ⏱️ Focus Time (minutes)")
    st.session_state.session_minutes = st.slider("Set time", 1, 60, st.session_state.session_minutes)

    if st.button("✅ Start Session"):
        st.session_state.start_time = time.time()
        st.session_state.focused_time = 0
        st.session_state.timestamps = []
        st.session_state.focus_times = []

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# VideoProcessor Class
class FocusProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_check_time = time.time()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        now = time.time()
        status = "Not Focused"

        if result.multi_face_landmarks:
            for landmarks in result.multi_face_landmarks:
                left_eye = [landmarks.landmark[i] for i in range(33, 39)]
                right_eye = [landmarks.landmark[i] for i in range(39, 45)]
                ear = (np.linalg.norm(np.array([left_eye[1].x, left_eye[1].y]) - np.array([left_eye[5].x, left_eye[5].y])) +
                       np.linalg.norm(np.array([left_eye[2].x, left_eye[2].y]) - np.array([left_eye[4].x, left_eye[4].y])) +
                       np.linalg.norm(np.array([right_eye[1].x, right_eye[1].y]) - np.array([right_eye[5].x, right_eye[5].y])) +
                       np.linalg.norm(np.array([right_eye[2].x, right_eye[2].y]) - np.array([right_eye[4].x, right_eye[4].y]))) / \
                      (2.0 * (np.linalg.norm(np.array([left_eye[0].x, left_eye[0].y]) - np.array([left_eye[3].x, left_eye[3].y])) +
                              np.linalg.norm(np.array([right_eye[0].x, right_eye[0].y]) - np.array([right_eye[3].x, right_eye[3].y]))))

                if ear > 0.2:
                    status = "Focused"
                    st.session_state.focused_time += now - self.last_check_time

                mp.solutions.drawing_utils.draw_landmarks(img, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        # Session metrics
        if st.session_state.start_time:
            st.session_state.timestamps.append(now - st.session_state.start_time)
            st.session_state.focus_times.append(st.session_state.focused_time)

        self.last_check_time = now
        color = (0, 255, 0) if status == "Focused" else (0, 0, 255)
        cv2.putText(img, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Launch Webcam Stream
if st.session_state.start_time:
    webrtc_streamer(
        key="focus-stream",
        video_processor_factory=FocusProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Remaining time display
    elapsed = time.time() - st.session_state.start_time
    remaining = int(st.session_state.session_minutes * 60 - elapsed)
    if remaining > 0:
        st.markdown(f"<div class='metric'>⏳ Remaining Time: {remaining // 60}m {remaining % 60}s</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='metric'>✅ Session Complete!</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'>🧠 Focused Time: {int(st.session_state.focused_time)}s</div>", unsafe_allow_html=True)
        percent = (st.session_state.focused_time / (st.session_state.session_minutes * 60)) * 100
        st.markdown(f"<div class='metric'>📊 Focus %: {percent:.2f}%</div>", unsafe_allow_html=True)

        if percent >= 80:
            st.success("🌟 Amazing! Strong focus throughout.")
        elif percent >= 50:
            st.info("👍 Good effort! Try removing more distractions next time.")
        else:
            st.warning("😴 Tough session? Try again after a break.")

        # Plot
        fig, ax = plt.subplots(facecolor='#1e1e1e')
        ax.plot(st.session_state['timestamps'], st.session_state['focus_times'], color='#00ffcc')
        ax.set_facecolor('#1e1e1e')
        ax.set_xlabel("Time (s)", color='white')
        ax.set_ylabel("Focused Time (s)", color='white')
        ax.set_title("Focus Over Time", color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
