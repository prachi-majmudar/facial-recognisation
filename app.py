import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image
import tempfile
import os
import pandas as pd
from datetime import datetime

# Enhance image for better detection
def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

# App title and instructions
st.set_page_config(page_title="Face Verification App", layout="centered")
st.title("🔒 Live Facial Recognition")
st.markdown("Upload a reference face image and verify your identity via webcam using `DeepFace` + `ArcFace`. All verifications are logged.\n\nMake sure your face is clearly visible and well-lit.")

# Upload reference image
uploaded_ref = st.file_uploader("📁 Upload Reference Image", type=["jpg", "jpeg", "png"])

if uploaded_ref:
    ref_img = Image.open(uploaded_ref).convert("RGB")
    st.image(ref_img, caption="🧾 Reference Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as ref_file:
        ref_img.save(ref_file.name)
        ref_path = ref_file.name

    # Extract face from reference image
    ref_faces = DeepFace.extract_faces(ref_path, detector_backend='retinaface')
    if ref_faces:
        ref_crop = ref_faces[0]["face"]
        st.image(ref_crop, caption="🖼️ Cropped Reference Face")
    else:
        st.warning("❌ No face detected in reference image.")

    st.success("✅ Reference image loaded! Now start webcam and press Capture.")

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.capture_now = False
            self.captured_frame = None

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            if self.capture_now:
                self.captured_frame = img.copy()
                self.capture_now = False
            return img

    ctx = webrtc_streamer(key="live-stream", video_processor_factory=VideoProcessor)

    if ctx.video_processor:
        if st.button("📸 Capture & Compare"):
            ctx.video_processor.capture_now = True

        if ctx.video_processor.captured_frame is not None:
            live_img = ctx.video_processor.captured_frame
            st.image(live_img, caption="📷 Captured Live Image", use_column_width=True)

            enhanced = enhance_image(live_img)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as live_temp:
                cv2.imwrite(live_temp.name, enhanced)
                live_path = live_temp.name

            try:
                st.info("🔍 Verifying...")
                result = DeepFace.verify(
                    img1_path=ref_path,
                    img2_path=live_path,
                    model_name="ArcFace",
                    detector_backend="retinaface",
                    enforce_detection=True
                )

                score = 1 - result["distance"]
                st.metric(label="Similarity Score", value=f"{score:.4f}")

                # Extract faces to show side-by-side comparison
                live_faces = DeepFace.extract_faces(live_path, detector_backend='retinaface')
                if ref_faces and live_faces:
                    st.image([ref_crop, live_faces[0]["face"]], caption=["Reference Face", "Live Face"])

                if result["verified"]:
                    st.success("🎉 Face Matched!")
                    st.balloons()
                else:
                    st.warning("❌ Face Not Matched.")

                # Log the result
                log_path = "match_log.csv"
                log_df = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame(columns=["Timestamp", "Result", "Similarity"])

                new_row = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Result": "Matched" if result["verified"] else "Not Matched",
                    "Similarity": round(score, 4)
                }

                log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
                log_df.to_csv(log_path, index=False)
                st.info("📝 Result logged successfully.")

            except Exception as e:
                st.error(f"🚫 Verification failed: {e}")
