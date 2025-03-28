import streamlit as st
import json
import os
import uuid
import cv2
import numpy as np
from PIL import Image
import firebase_admin
from firebase_admin import credentials, db, storage
import tempfile
from ultralytics import YOLO

import streamlit as st
import firebase_admin
from firebase_admin import credentials, storage

# Load Firebase credentials from Streamlit secrets
firebase_config = st.secrets["firebase"]

# Convert Streamlit secrets to a Firebase credentials object
cred = credentials.Certificate({
    "type": firebase_config["type"],
    "project_id": firebase_config["project_id"],
    "private_key_id": firebase_config["private_key_id"],
    "private_key": firebase_config["private_key"].replace('\\n', '\n'),  # Fix newlines
    "client_email": firebase_config["client_email"],
    "client_id": firebase_config["client_id"],
    "auth_uri": firebase_config["auth_uri"],
    "token_uri": firebase_config["token_uri"],
    "auth_provider_x509_cert_url": firebase_config["auth_provider_x509_cert_url"],
    "client_x509_cert_url": firebase_config["client_x509_cert_url"],
    "universe_domain": firebase_config["universe_domain"],
})

# Initialize Firebase app
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "storageBucket": f"{firebase_config['project_id']}.appspot.com"
    })

# ✅ Cache the YOLO model for performance
@st.cache_resource
def load_model():
    return YOLO("best1.pt")

model = load_model()

# ✅ Streamlit UI
st.title("Skin Cancer Detection with YOLOv8")
st.write("Upload an image to detect skin cancer using AI.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

detected = False
detection_message = "Cancer Not Detected (Probably Normal Skin)"
ml_accuracy = 0.0
image_url = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to OpenCV format
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Perform YOLO Inference
    results = model(img_array)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            label = result.names[int(box.cls[0])]

            # Draw bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_array, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected = True
            detection_message = f"Detected: {label}"
            ml_accuracy = confidence * 100  

    # Convert processed image to RGB for display
    result_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    st.image(result_image, caption="Detection Results", use_column_width=True)

    # ✅ Save image temporarily
    temp_image_path = f"/tmp/{uuid.uuid4()}.jpg"
    cv2.imwrite(temp_image_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    # ✅ Upload to Firebase Storage
    bucket = storage.bucket()
    unique_filename = f"detected_images/{uuid.uuid4()}.jpg"
    blob = bucket.blob(unique_filename)
    blob.upload_from_filename(temp_image_path)
    blob.make_public()  
    image_url = blob.public_url  

    # ✅ Remove temporary file
    os.remove(temp_image_path)

# ✅ Patient Information Form
st.subheader("Patient Information")
with st.form("patient_form"):
    name = st.text_input("Full Name")
    address = st.text_area("Address")
    phone = st.text_input("Phone Number")
    email = st.text_input("Email ID")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    submit_details = st.form_submit_button("Submit & Send to Doctor")

# ✅ Save Patient Data to Firebase
if submit_details:
    patient_data = {
        "name": name,
        "age": age,
        "gender": gender,
        "phone": phone,
        "email": email,
        "address": address,
        "detection_result": detection_message,
        "model_accuracy": ml_accuracy,
        "detection_image": image_url  
    }

    ref = db.reference("patients/SkinCancer_DoctorApp")
    new_patient_ref = ref.push(patient_data)

    st.success(f"✅ Data sent to Firebase successfully!\n\n"
               f"**Patient Details:**\n"
               f"- Name: {name}\n"
               f"- Age: {age}\n"
               f"- Gender: {gender}\n"
               f"- Phone: {phone}\n"
               f"- Email: {email}\n"
               f"- Address: {address}\n\n"
               f"**AI Model Detection:** {detection_message}\n"
               f"**Model Accuracy:** {ml_accuracy:.2f}%\n\n"
               f"📸 **Detection Image:** [View Image]({image_url})\n\n"
               f"📌 *Waiting for doctor's diagnosis...*")
