import requests
import glob


target = glob.glob('data/preprocessed/test/images/*')[:]


url = 'http://0.0.0.0:8000/predict'  # Replace with your FastAPI server address

data = {
    "images": target[:2],
}

headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

response = requests.post(url, json=data, headers=headers)

if response.status_code == 200:
    print("Prediction successful:")
    print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.text)