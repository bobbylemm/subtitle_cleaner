import json
import requests

# Read the SRT file
with open('two_deals.srt', 'r', encoding='utf-8') as f:
    content = f.read()

# Prepare the request
url = 'http://localhost:8080/v1/clean/'
headers = {
    'X-API-Key': 'sk-dev-key-1234567890',
    'Content-Type': 'application/json'
}
data = {
    'content': content,
    'format': 'srt',
    'language': 'en',
    'enable_holistic_correction': True
}

# Make the request
response = requests.post(url, headers=headers, json=data, timeout=60)

# Check the response
if response.status_code == 200:
    result = response.json()
    if result.get('success'):
        # Save the corrected content
        with open('two_deals_corrected.srt', 'w', encoding='utf-8') as f:
            f.write(result['content'])
        print("✅ Successfully processed and saved to two_deals_corrected.srt")
        
        # Check for key corrections
        corrected = result['content']
        print("\n📊 Correction Summary:")
        print(f"  - 'contrast' found: {'contrast' in content.lower()}")
        print(f"  - 'contract' in output: {'contract' in corrected.lower()}")
        print(f"  - 'May United' found: {'May United' in content}")
        print(f"  - 'Man United' in output: {'Man United' in corrected}")
        print(f"  - 'Mecano' found: {'Mecano' in content}")
        print(f"  - 'Upamecano' in output: {'Upamecano' in corrected}")
        print(f"  - 'Thebal' found: {'Thebal' in content}")
        print(f"  - 'the ball' in output: {'the ball' in corrected.lower()}")
    else:
        print(f"❌ Processing failed: {result.get('errors')}")
else:
    print(f"❌ Request failed with status {response.status_code}")
    print(response.text)
