import os
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from deepface import DeepFace
import numpy as np
import shutil
import tempfile

# Setup
st.set_page_config(page_title="Live Facial Recognition", layout="centered")
st.title("🧠 Real-Time Face Recognition with DeepFace")
st.markdown("Upload a reference image, then start webcam detection.")

# Directories
DB_DIR = "reference_faces"
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# Upload image
uploaded_file = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save reference image
    ref_path = os.path.join(DB_DIR, "user.jpg")
    with open(ref_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.image(ref_path, caption="Reference Image", width=300)

# Initialize detection state
match_found = st.empty()

# Streamlit WebRTC for webcam
class FaceMatchTransformer(VideoTransformerBase):
    def __init__(self):
        self.match_displayed = False
        self.ref_img_path = os.path.join(DB_DIR, "user.jpg")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            # Temporarily save webcam frame
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                cv2.imwrite(tmp_file.name, img)

                result = DeepFace.verify(
                    img1_path=self.ref_img_path,
                    img2_path=tmp_file.name,
                    enforce_detection=False,
                    detector_backend="opencv",
                    model_name="Facenet512"
                )

                if result["verified"] and not self.match_displayed:
                    match_found.success("✅ Face Match Found!")
                    self.match_displayed = True
                elif not result["verified"]:
                    match_found.warning("❌ Face Not Matching")
        except Exception as e:
            pass

        return img

if uploaded_file is not None:
    st.markdown("### 🔍 Press the button below to start webcam recognition")
    webrtc_streamer(
        key="face-rec",
        video_transformer_factory=FaceMatchTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )
