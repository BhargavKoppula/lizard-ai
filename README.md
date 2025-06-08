# lizard-ai
# 🦎 Lizard AI - Focus Tracker

A webcam-based real-time focus tracking tool to help users stay productive by analyzing facial cues using AI.

## 🚀 Features

- 🎥 Live webcam feed
- 👁️ Face & eye tracking with MediaPipe
- ⏱️ Session timer and duration selection
- 🧠 Real-time focus status ("Focused" or "Not Focused")
- 📊 Focus percentage & analytics after session
- 📈 Focus-over-time visualization
- 🎯 Motivational feedback based on performance

---

## 🛠️ Technologies Used

| Tech | Purpose |
|------|---------|
| `Python` | Main programming language |
| `Streamlit` | Web app frontend |
| `OpenCV` | Image processing |
| `MediaPipe` | Face and eye landmark detection |
| `NumPy` | Array manipulation |
| `Matplotlib` | Graph plotting |
| `PIL` | Image loading |
| `random` | Motivational message generation |

---

## 📷 How It Works

1. User sets a session time (e.g., 5 min, 15 min).
2. App activates the webcam and starts tracking face & eyes.
3. Focus is determined by Eye Aspect Ratio (EAR) and presence of facial landmarks.
4. Session runs in real-time, logging focus metrics.
5. At the end, a full analysis is displayed:
   - Total focused time
   - Focus percentage
   - Feedback messages
   - A line graph of focus over time

---

## 🧪 Setup Instructions

> ⚠️ Works locally — requires webcam access.

### 1. Clone the repo
```bash
git clone https://github.com/BhargavKoppula/Lizard-AI.git
cd Lizard-AI
```

## install requirements
```pip install -r requirements.txt```

## run the app
```streamlit run app.py```

## Folder Structure
Lizard-AI/
├── app.py                  # Main app logic
├── logo_cut.png            # Logo for branding
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation


## requirements.txt
streamlit
opencv-python
mediapipe
numpy
matplotlib
Pillow

## 🛠️ Features

- 👁️ **Webcam-Based Detection** – Uses face & eye landmarks to assess attention
- ⏱️ **Custom Focus Timer** – Set sessions from 2 to 60 minutes
- 🧠 **Focus Status Tracking** – Real-time "Focused / Not Focused" display
- 📊 **Summary Analytics** – Focus percentage, total focused time, and performance messages
- 📈 **Graph Output** – Line graph showing focus level progression
- ⚡ **No Manual Input Needed** – Just click and go!

@📌 Upcoming Features

🌐 Web/Cloud Deployment – So users can use it anywhere via a browser
🔌 Chrome Extension – Track focus while using other tools like YouTube, ChatGPT, VS Code
📤 Session export & reporting
🔐 Optional login for history tracking

## 🎥 Demo

> You can also watch it on [LinkedIn](https://www.linkedin.com/posts/YOUR_LINKEDIN_VIDEO)  
> *(Includes subtitles to showcase session flow)*


## 👨‍💻 Author
# Bhargav Koppula
[LinkedIn](https://www.linkedin.com/in/bhargav-koppula)

## 📢 License
This project is open-source under the MIT License.

## 🙌 Contribute
Have ideas? Found a bug? Want to collaborate?
Feel free to open an issue or pull request!

## 🙏 Acknowledgements

- 📦[MediaPipe](https://github.com/google/mediapipe) – Real-time face and eye landmark detection by Google
- 📦[OpenCV](https://opencv.org/) – Image and video processing library
- 📦[Streamlit](https://streamlit.io/) – Web app framework for ML tools and dashboards
- 📦[Matplotlib](https://matplotlib.org/) – Graph plotting and visualization
- 📦[NumPy](https://numpy.org/) – Efficient numerical and array computations




