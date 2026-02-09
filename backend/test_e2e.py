"""
E2E Test Script for AutoArchitect AI
"""
import httpx
import time

print('=== E2E TEST: AutoArchitect AI ===')

# Step 1: Upload
print('\n[1/4] Uploading test image...')
with open('test_plot.png', 'rb') as f:
    files = {'file': ('test_plot.png', f, 'image/png')}
    r = httpx.post('http://localhost:8000/api/upload', files=files)
    upload = r.json()
    job_id = upload['job_id']
    print(f'    Job ID: {job_id}')
    print(f'    Status: {upload["status"]}')

# Step 2: Trigger design
print('\n[2/4] Triggering design generation...')
r = httpx.post('http://localhost:8000/api/design/generate', json={
    'job_id': job_id,
    'requirements': {'bedrooms': 3, 'bathrooms': 2, 'style': 'modern', 'features': []}
})
print(f'    Response: {r.json()["message"]}')

# Step 3: Wait for completion
print('\n[3/4] Waiting for pipeline to complete...')
for i in range(15):
    time.sleep(1)
    r = httpx.get(f'http://localhost:8000/api/job/{job_id}/status')
    status = r.json()
    print(f'    Progress: {status["progress"]}% - {status["message"]}')
    if status['status'] == 'completed':
        break

# Step 4: Verify results
print('\n[4/4] Verifying results...')
r = httpx.get(f'http://localhost:8000/api/results/{job_id}')
results = r.json()
print(f'    Status: {results["status"]}')
png_url = results["files"]["png_url"]
print(f'    PNG URL: {png_url}')

# Verify image is accessible
if png_url:
    img = httpx.get(f'http://localhost:8000{png_url}')
    print(f'    Image accessible: {img.status_code == 200} ({len(img.content)} bytes)')

print('\n=== TEST COMPLETE ===')
print(f'Open browser to: http://localhost:3000/results/{job_id}')
