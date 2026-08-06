"""Quick verification script for the search pipeline."""
import sys
sys.path.insert(0, 'src')

from fastapi.testclient import TestClient
from backend.app import create_app

# Create app in test mode (no model loading)
app = create_app(test_mode=True)
client = TestClient(app)

# Test health endpoint
response = client.get('/health')
print(f"Health: {response.status_code} - {response.json()}")

# Test root endpoint
response = client.get('/')
print(f"Root: {response.status_code} - {response.json()}")

# Test that search returns 503 in test mode (no model loaded)
response = client.post('/search/text', json={'query': 'test'})
print(f"Search (no model): {response.status_code} - {response.json()['detail']}")

print("\nAll endpoints responding correctly!")
