"""Tests for dynamic feature route mounting and auth protection."""

from fastapi.testclient import TestClient

from xencode.api.main import app

client = TestClient(app)


def test_dynamic_feature_routes_are_mounted():
    mounted_paths = {route.path for route in app.routes if hasattr(route, 'path')}

    assert "/api/v1/collab/start" in mounted_paths
    assert "/api/v1/review/file" in mounted_paths
    assert "/api/v1/security/scan" in mounted_paths


def test_dynamic_feature_route_requires_auth():
    response = client.post(
        "/api/v1/collab/start",
        json={"name": "room-1", "owner_id": "u1", "username": "alice"},
    )

    assert response.status_code == 401
