"""Tests for authentication protection on core API routes."""

from fastapi.testclient import TestClient

from xencode.api.main import app

client = TestClient(app)


def test_code_analyze_requires_auth():
    response = client.post(
        "/api/v1/code/analyze",
        json={"code": "print('hello')", "language": "python"},
    )

    assert response.status_code == 401


def test_document_list_requires_auth():
    response = client.get("/api/v1/documents/")

    assert response.status_code == 401
