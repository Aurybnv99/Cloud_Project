import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import tensorflow as tf
import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from send_emotion_data import send_data
from datetime import datetime


#----------------------------MQTT Configuration-----------------------------
import requests
import json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

client = AWSIoTMQTTClient("984158813029")
client.configureEndpoint("a1uw6gpf0g102v-ats.iot.eu-north-1.amazonaws.com", 8883)
client.configureCredentials(
    "/home/aureliobenvenuto/Certificates/AmazonRootCA1.pem",
    "/home/aureliobenvenuto/Certificates/7bd2248c39a1c17a701ebdc33b618cc7cc166fdc02d45a558e2085f2db669d51-private.pem.key",
    "/home/aureliobenvenuto/Certificates/7bd2248c39a1c17a701ebdc33b618cc7cc166fdc02d45a558e2085f2db669d51-certificate.pem.crt"
    )
client.connect()

#----------------------------------------------------------------------------

# --- Configuration section: model and parameters ---
MODEL_PATH = 'Model/emotionModel.h5'
HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_INPUT_SHAPE = (48, 48, 1)
RESIZE_INPUT_FRAME = True # Set to False if you don't want to resize the input feed
FRAME_WIDTH = 480 # Width to resize to if RESIZE_INPUT_FRAME is True

# --- TensorFlow CPU optimization ---
try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
except Exception:
    pass

# --- Load the emotion recognition model and face detector ---
try:
    model = load_model(MODEL_PATH)
    print(f"Emotion detection model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit()

face_classifier = cv2.CascadeClassifier(HAARCASCADE_PATH)
if face_classifier.empty():
    print(f"Error loading Haar Cascade from {HAARCASCADE_PATH}")
    exit()

# --- Initialize webcam video capture ---
cap = cv2.VideoCapture(0) # Use 0 for default camera
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

print("Starting video stream and emotion detection...")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Press 'Q' To Quit.~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

update_interval = 15  # Update detection every 15 frames (~0.5 seconds)
frame_counter = 0
last_face_locations = None
last_preds = None
current_emotion = None
emotion_start_time = None

# --- Main processing loop: read frames, detect faces, predict emotions ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_display = frame.copy()
    # --- Resize frame if needed ---
    scale = FRAME_WIDTH / frame.shape[1] if RESIZE_INPUT_FRAME else 1.0
    frame_processed = cv2.resize(
        frame, (FRAME_WIDTH, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA
    ) if RESIZE_INPUT_FRAME else frame

    # --- Convert frame to grayscale and detect faces ---
    gray = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    face_rois = []
    face_locations = []
    # --- Extract face regions and preprocess for model ---
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue
        roi = img_to_array(
            cv2.resize(roi_gray, (MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1]), interpolation=cv2.INTER_AREA)
        ).astype('float') / 255.0
        if MODEL_INPUT_SHAPE[2] == 1 and len(roi.shape) < 3:
            roi = np.expand_dims(roi, axis=-1)
        face_rois.append(roi)
        face_locations.append((x, y, w, h))

    # --- HANDLE FACE DISAPPEARANCE ---
    if len(faces) == 0:
        if current_emotion is not None and emotion_start_time is not None:
            artwork = str(random.randint(1, 10))
            emotion_name = last_emotion_name if 'last_emotion_name' in locals() else "neutral"
            if frame_counter % update_interval == 0:
                timestamp = emotion_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                send_data(client, artwork, emotion_name, timestamp, 0)
            current_emotion = None
            emotion_start_time = None
        last_face_locations = None
        last_preds = None

    # --- Predict emotions every N frames ---
    if face_rois and frame_counter % update_interval == 0:
        preds = model.predict(np.array(face_rois, dtype='float32'), verbose=0)
        last_face_locations = face_locations
        last_preds = preds

    # --- Draw results and handle emotion state ---
    if last_preds is not None and last_face_locations is not None:
        for i, (x, y, w, h) in enumerate(last_face_locations):
            li = last_preds[i].argmax()
            lab = EMOTION_LABELS[li]
            confidence = float(last_preds[i][li])
            confidence_int = int(confidence * 100)
            emotion_name = lab.lower()
            last_emotion_name = emotion_name
            now = datetime.now()
            if current_emotion != emotion_name:
                if current_emotion is not None and emotion_start_time is not None:
                    artwork = str(random.randint(1, 10))
                    if frame_counter % update_interval == 0:
                        timestamp = emotion_start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        send_data(client, artwork, last_emotion_name, timestamp, confidence_int)
                current_emotion = emotion_name
                emotion_start_time = now
            cv2.rectangle(
                frame_display, (int(x / scale), int(y / scale)), (int((x + w) / scale), int((y + h) / scale)), (255, 165, 0), 2
            )
            cv2.putText(
                frame_display, f"{lab}: {confidence:.2f}", (int(x / scale), int(y / scale) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2
            )

    # --- Show the frame with results ---
    cv2.imshow('Emotion Recognition - Press Q to Quit', frame_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_counter += 1

# --- Cleanup resources ---
client.disconnect()
cap.release()
cv2.destroyAllWindows()