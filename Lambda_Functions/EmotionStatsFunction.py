import boto3
from collections import defaultdict, Counter
from datetime import datetime

def lambda_handler(event, context):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('EmotionDetectionDB')
    response = table.scan()
    items = response['Items']

    print(f"Total item readed from DynamoDB: {len(items)}") #DEBUG
    # Filter items for the current day only
    today_str = datetime.now().strftime('%Y-%m-%d')
    artwork_emotions = defaultdict(Counter)
    for item in items:
        detected_at = item.get('timestamp')
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
        if date_str != today_str:
            continue
        artwork_id = item['artwork_id']
        emotion_id = item['emotion_id']
        artwork_emotions[artwork_id][emotion_id] += 1

    # Send total emotions for each artwork to CloudWatch
    cloudwatch = boto3.client('cloudwatch')
    for artwork_id, emotions in artwork_emotions.items():
        total = sum(emotions.values())
        artwork_name = str(artwork_id)
        print(f"Artwork {artwork_id}: total emotions = {total}") #DEBUG
        cloudwatch.put_metric_data(
            Namespace='EmotionStats',
            MetricData=[
                {
                    'MetricName': 'EmotionCountPerArtwork2',
                    'Dimensions': [
                        {'Name': 'ArtworkName', 'Value': artwork_name},
                    ],
                    'Timestamp': datetime.now(),
                    'Value': total,
                    'Unit': 'Count'
                },
            ]
        )