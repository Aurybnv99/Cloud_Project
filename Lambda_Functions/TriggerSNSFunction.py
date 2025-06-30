import boto3
import os
import json
from datetime import datetime, timedelta

# SNS config
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', 'arn:aws:sns:eu-north-1:984158813029:EmotionAlerts')
sns_client = boto3.client('sns')

# DynamoDB config
dynamodb = boto3.resource('dynamodb')
emotion_table = dynamodb.Table('EmotionDetectionDB')

NEGATIVE_EMOTIONS = ['disgust', 'angry', 'sad', 'fear']

def get_today_time_range():
    now = datetime.now()
    start = datetime(now.year, now.month, now.day)
    end = now
    return start, end  # datetime objects

def query_events(start_dt, end_dt):
    response = emotion_table.scan()
    items = response.get('Items', [])
    filtered = []
    for e in items:
        try:
            ts = datetime.strptime(e['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
            if start_dt <= ts <= end_dt:
                filtered.append(e)
        except Exception:
            continue
    return filtered

def hour_stats(events):
    hour_count = {}
    for e in events:
        dt = datetime.strptime(e['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        hour = dt.hour
        hour_count[hour] = hour_count.get(hour, 0) + 1
    if not hour_count:
        return None, 0
    busiest_hour = max(hour_count, key=hour_count.get)
    return busiest_hour, hour_count[busiest_hour]

def confidence_per_hour_per_artwork(events):
    stats = {}
    for e in events:
        artwork = e['artwork_id']
        dt = datetime.strptime(e['timestamp'], "%Y-%m-%d %H:%M:%S.%f")
        hour = dt.hour
        key = (artwork, hour)
        conf = float(e['confidence']) / 100.0
        stats.setdefault(key, []).append(conf)
    result = {}
    for key, values in stats.items():
        result[key] = sum(values) / len(values)
    return result

def negative_emotions_report(events):
    report = {}
    for e in events:
        emotion = e['emotion_id']
        if emotion in NEGATIVE_EMOTIONS:
            artwork = e['artwork_id']
            report.setdefault(artwork, {})
            report[artwork][emotion] = report[artwork].get(emotion, 0) + 1
    return report

def send_email(subject, message):
    sns_client.publish(
        TopicArn=SNS_TOPIC_ARN,
        Subject=subject,
        Message=message
    )

def lambda_handler(event, context):
    try:
        start_dt, end_dt = get_today_time_range()
        events = query_events(start_dt, end_dt)

        hour, count = hour_stats(events)
        msg1 = f"Most crowded hour today: {hour}:00 with {count} detections." if hour is not None else "No data for today."
        print(msg1) #DEBUG
        send_email("Recap most crowded hour", msg1)

        conf_stats = confidence_per_hour_per_artwork(events)
        msg2 = "Average confidence per hour per artwork (today):\n"
        for (artwork, hour), avg in conf_stats.items():
            msg2 += f"- {artwork} at {hour}:00: {avg:.2f}\n"
        print(msg2) #DEBUG
        send_email("Average confidence per hour per artwork", msg2)

        neg_report = negative_emotions_report(events)
        msg3 = "Negative emotions detected (today):\n"
        for artwork, emotions in neg_report.items():
            msg3 += f"- {artwork}:\n"
            for emotion, count in emotions.items():
                msg3 += f"    {emotion}: {count}\n"
        if not neg_report:
            msg3 += "No negative emotions detected."
        print(msg3) #DEBUG
        send_email("Negative emotions report", msg3)

    except Exception as e:
        send_email("Error in daily report", f"Error: {str(e)}")