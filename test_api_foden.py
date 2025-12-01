import requests
import os

import time

def test_api():
    # Target the Backend Directly to measure time
    url = "http://localhost:8000/v1/universal/universal-correct"
    file_path = "foden_winner.srt"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Sending {file_path} to {url}...")
    start_time = time.time()
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "text/plain")}
        data = {
            "topic": "Football",
            "industry": "Sports",
            "country": "UK"
        }
        try:
            response = requests.post(url, files=files, data=data, timeout=120)
            duration = time.time() - start_time
            print(f"Status Code: {response.status_code}")
            print(f"Duration: {duration:.2f} seconds")
            
            if response.status_code != 200:
                print("Error Response:", response.text)
            else:
                print("Success! Response received.")
                data = response.json()
                corrections = data.get("applied_corrections", [])
                print(f"Applied Corrections Count: {len(corrections)}")
                if corrections:
                    print("First 3 corrections:")
                    for c in corrections[:3]:
                        print(c)
                
                changes = data.get("changes", [])
                print(f"Changes Count: {len(changes)}")
                if changes:
                    print("First 3 changes:")
                    for c in changes[:3]:
                        print(c)
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
