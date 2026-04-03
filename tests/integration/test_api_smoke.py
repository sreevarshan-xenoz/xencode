"""Smoke tests for core API availability."""

import pytest
from fastapi.testclient import TestClient

from xencode.api.main import app


pytestmark = [pytest.mark.integration, pytest.mark.e2e]


client = TestClient(app)


def test_health_endpoint_is_available():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"


def test_info_endpoint_is_available():
    response = client.get("/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "Xencode API"
