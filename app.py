import av
import cv2
import time
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Streamlit Page Config
st.set_page_config(page_title="Lizard - Focus Tracker", layout="wide")
st.title("🦎 Lizard AI - Focus Tracker")
st.markdown("Track your focus level using webcam and MediaPipe FaceMesh")

mp_face_mesh = mp.solutions.face_mesh

# Session variables
if 'focused_time' not in st.session_state:
    st.session_state.focused_time = 0
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []
if 'focus_scores' not in st.session_state:
    st.session_state.focus_scores = []

# Face detection transformer
class FocusTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.last_check_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        status = "Not Focused"
        now = time.time()
        focused = 0

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                status = "Focused"
                focused = 1

        # Update metrics
        st.session_state.timestamps.append(now - st.session_state.start_time)
        st.session_state.focus_scores.append(focused)
        if focused:
            st.session_state.focused_time += now - self.last_check_time

        self.last_check_time = now
        label = f"{status}"
        color = (0, 255, 0) if status == "Focused" else (0, 0, 255)
        cv2.putText(img, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# Start session
minutes = st.slider("🕒 Set Focus Duration (minutes):", 1, 60, 10)
duration = minutes * 60

if st.button("Start Focus Session"):
    st.session_state.start_time = time.time()
    st.session_state.focused_time = 0
    st.session_state.timestamps = []
    st.session_state.focus_scores = []

    webrtc_streamer(key="focus", video_transformer_factory=FocusTransformer)

    st.success("Focus session in progress... Keep your eyes on the screen! 🧠")

if st.session_state.focus_scores:
    total = time.time() - st.session_state.start_time
    percent = (st.session_state.focused_time / total) * 100 if total > 0 else 0
    st.markdown(f"### 🧠 Total Focused Time: `{int(st.session_state.focused_time)}` seconds")
    st.markdown(f"### 📊 Focus Percentage: `{percent:.2f}%`")

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    fig, ax = plt.subplots()
    ax.plot(st.session_state.timestamps, st.session_state.focus_scores, color='cyan')
    ax.set_title("Focus Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Focus (1 = focused, 0 = not)")
    ax.grid(True)
    st.pyplot(fig)

    if percent >= 80:
        st.success("🌟 Amazing! You maintained strong focus throughout. Keep this up!")
    elif percent >= 50:
        st.info("👍 Good effort! A little more consistency and you'll nail it.")
    else:
        st.warning("😕 Tough session? It's okay—next one will be better!")
