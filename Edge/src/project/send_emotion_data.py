import requests
import json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# url = "http://localhost:1880/receive-emotion"
topic = "rpi5/emotion/data"

def send_data(client, artwork_id, emotion_id, detected_at, confidence):
    data = {
        "artwork_id": artwork_id,
        "emotion_id": emotion_id,
        "detected_at": detected_at,
        "confidence": confidence
    }
    print(data)
    try:
        # Send to AWS IoT Core
        client.publish(topic, json.dumps(data), 0)

    except Exception as e:
        print("Error sending data:", e)
