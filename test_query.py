import requests
import json

url = "http://localhost:8000/ask"
payload = {"question": "what are algorithms", "top_k": 5}
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(url, json=payload, headers=headers)
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
