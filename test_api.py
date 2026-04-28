import requests

BASE = "http://127.0.0.1:5001"

print("Test 1: Health")
r = requests.get(f"{BASE}/health")
print(r.status_code, r.json())

print("\nTest 2: Single prediction")
r = requests.post(f"{BASE}/predict", json={"feature1": 5})
print(r.status_code, r.json())

print("\nTest 3: Batch prediction")
batch = [{"feature1": i} for i in range(5)]
r = requests.post(f"{BASE}/predict/batch", json=batch)
print(r.status_code, r.json())

print("\nTest 4: Missing field")
r = requests.post(f"{BASE}/predict", json={})
print(r.status_code, r.json())

print("\nTest 5: Invalid type")
r = requests.post(f"{BASE}/predict", json={"feature1": "bad"})
print(r.status_code, r.json())
