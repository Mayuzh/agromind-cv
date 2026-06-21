import sys
from pathlib import Path

from fastapi.testclient import TestClient

from agromind_cv.api import app

image_path = Path(sys.argv[1])
with TestClient(app) as client, image_path.open("rb") as image:
    health = client.get("/health")
    prediction = client.post(
        "/predict",
        data={"plantId": "plant_042"},
        files={"file": (image_path.name, image, "image/jpeg")},
    )
print("health:", health.status_code, health.json())
print("predict:", prediction.status_code, prediction.json())
