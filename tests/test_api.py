from fastapi.testclient import TestClient
from truthlens.api.main import app

client = TestClient(app)

def test_index_ok():
    r = client.get("/")
    assert r.status_code == 200
