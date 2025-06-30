import boto3
from datetime import datetime
from collections import Counter

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('EmotionDetectionDB')
    response = table.scan()
    items = response['Items']
    cloudwatch = boto3.client('cloudwatch')

    # Get today's date string
    today_str = datetime.now().strftime('%Y-%m-%d')
    hour_counter = Counter()

    for item in items:
        detected_at = item['timestamp']
        # Parse timestamp
        if isinstance(detected_at, str):
            try:
                dt = datetime.fromisoformat(detected_at)
            except Exception:
                try:
                    dt = datetime.strptime(detected_at, "%Y-%m-%d %H:%M:%S.%f")
                except Exception:
                    dt = datetime.strptime(detected_at, "%Y-%m-%d %H:%M:%S")
        else:
            dt = datetime.fromtimestamp(int(detected_at))
        date_str = dt.strftime('%Y-%m-%d')
        hour = dt.hour
        if date_str == today_str:
            hour_counter[hour] += 1

    # Send the busiest hour for today
    if hour_counter:
        max_hour = max(hour_counter, key=hour_counter.get)
        max_count = hour_counter[max_hour]
        dt = datetime.strptime(f"{today_str} {max_hour}", "%Y-%m-%d %H")
        day_of_week = dt.strftime('%A')
        print(f"StatsBusyHour - Day: {day_of_week}, Date: {today_str}, Hour: {max_hour}, Count: {max_count}") #DEBUG
        cloudwatch.put_metric_data(
            Namespace='EmotionDetectionPerHour',
            MetricData=[
                {
                    'MetricName': 'StatsBusyHour',
                    'Timestamp': dt,
                    'Value': max_count,
                    'Unit': 'Count'
                },
            ]
        )