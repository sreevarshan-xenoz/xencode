#!/usr/bin/env python3
"""
Compatibility wrapper for the analytics API.

The canonical FastAPI router now lives in api.routers.analytics.
This module exposes an app for standalone usage.
"""

from typing import Optional

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from .api.routers.analytics import router as analytics_router
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class AnalyticsAPI:
    """Main Analytics API class (router-backed)."""

    def __init__(self, cors_origins: Optional[list] = None):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for Analytics API. Install with: pip install fastapi uvicorn")

        self.app = FastAPI(
            title="Xencode Analytics API",
            description="RESTful API for Xencode analytics system",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        if cors_origins is None:
            cors_origins = ["*"]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["analytics"])


def create_app() -> "FastAPI":
    """Create a standalone FastAPI app with analytics routes."""
    return AnalyticsAPI().app
