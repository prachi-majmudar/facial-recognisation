import streamlit as st
import face_recognition
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Live Facial Recognition App")
st.markdown("Upload a reference image and press the button to start recognition.")

# Upload reference image
reference_image_file = st.file_uploader("Upload Reference Face", type=["jpg", "jpeg", "png"])
start_recognition = st.button("Start Live Face Recognition")

# Load and encode reference image
if reference_image_file is not None:
    reference_image = face_recognition.load_image_file(reference_image_file)
    reference_face_encodings = face_recognition.face_encodings(reference_image)
    if reference_face_encodings:
        reference_encoding = reference_face_encodings[0]
        st.success("Reference face encoded successfully!")
    else:
        st.error("No face detected in the reference image. Try another one.")
        reference_encoding = None
else:
    reference_encoding = None

# Stream video from webcam
if start_recognition and reference_encoding is not None:
    class FaceMatch(VideoTransformerBase):
        def transform(self, frame):
            frame_rgb = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(frame_rgb)
            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                match = face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=0.5)[0]
                label = "MATCH" if match else "No Match"
                color = (0, 255, 0) if match else (0, 0, 255)

                cv2.rectangle(frame_rgb, (left, top), (right, bottom), color, 2)
                cv2.putText(frame_rgb, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    webrtc_streamer(key="facematch", video_transformer_factory=FaceMatch)
