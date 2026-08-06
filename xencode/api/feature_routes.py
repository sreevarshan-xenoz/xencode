#!/usr/bin/env python3
"""Dynamic feature route registration for FastAPI."""

import inspect
import logging
from typing import Any, Dict, Union, get_args, get_origin

from fastapi import Depends, FastAPI, HTTPException, Request

from xencode.api.auth import verify_jwt_token
from xencode.features.manager import FeatureManager

logger = logging.getLogger(__name__)


def _normalize_feature_path(path: str) -> str:
    """Normalize feature endpoint paths under /api/v1."""
    if not path:
        return "/api/v1/features/unknown"
    if not path.startswith("/"):
        path = f"/{path}"
    if path.startswith("/api/v1/"):
        return path
    if path.startswith("/api/"):
        return f"/api/v1/{path[len('/api/') :]}"
    return f"/api/v1{path}"


def _coerce_value(value: Any, annotation: Any) -> Any:
    """Coerce request values based on function annotations."""
    if annotation in (inspect.Signature.empty, Any):
        return value

    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if args:
            return _coerce_value(value, args[0])
        return value

    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
    if annotation is int:
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)

    return value


async def _extract_payload(request: Request) -> Dict[str, Any]:
    """Extract JSON payload for write requests."""
    if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return {}
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _build_handler_kwargs(handler: Any, request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build kwargs for the feature handler from request data."""
    signature = inspect.signature(handler)

    values: Dict[str, Any] = {}
    values.update(payload)
    values.update(dict(request.query_params))
    values.update(request.path_params)

    kwargs: Dict[str, Any] = {}
    missing = []

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if name in values:
            kwargs[name] = _coerce_value(values[name], parameter.annotation)
        elif parameter.default is inspect.Signature.empty:
            missing.append(name)

    if missing:
        missing_str = ", ".join(missing)
        raise HTTPException(
            status_code=422,
            detail=f"Missing required parameters for feature endpoint: {missing_str}",
        )

    return kwargs


def register_dynamic_feature_routes(app: FastAPI) -> int:
    """Register feature endpoint descriptors as live FastAPI routes."""
    try:
        manager = FeatureManager()
    except Exception as exc:
        logger.warning("Unable to create feature manager for dynamic routes: %s", exc)
        return 0

    existing_routes = {
        (route.path, method)
        for route in app.routes
        if hasattr(route, "methods")
        for method in (route.methods or set())
    }

    registered_count = 0

    for feature_key in manager.get_available_features():
        try:
            feature = manager.get_feature(feature_key) or manager.load_feature(feature_key)
            if not feature:
                continue

            endpoints = feature.get_api_endpoints() or []
            for endpoint in endpoints:
                raw_path = endpoint.get("path")
                raw_method = endpoint.get("method", "GET")
                handler = endpoint.get("handler")

                if not raw_path or not callable(handler):
                    continue

                method = str(raw_method).upper()
                path = _normalize_feature_path(str(raw_path))

                if (path, method) in existing_routes:
                    continue

                async def route_handler(
                    request: Request,
                    _feature=feature,
                    _handler=handler,
                    _auth: Dict[str, Any] = Depends(verify_jwt_token),
                ):
                    try:
                        if not _feature.is_initialized:
                            await _feature.initialize()

                        payload = await _extract_payload(request)
                        kwargs = _build_handler_kwargs(_handler, request, payload)
                        result = _handler(**kwargs)
                        if inspect.isawaitable(result):
                            result = await result
                        return result
                    except HTTPException:
                        raise
                    except Exception as exc:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Feature endpoint execution failed: {exc}",
                        )  from exc

                route_name = f"feature_{feature.name}_{method.lower()}_{path.strip('/').replace('/', '_').replace('{', '').replace('}', '')}"
                app.add_api_route(
                    path,
                    route_handler,
                    methods=[method],
                    name=route_name,
                    tags=["Features", "Dynamic Feature Endpoints"],
                )
                existing_routes.add((path, method))
                registered_count += 1
        except Exception as exc:
            logger.warning("Skipping feature '%s' route registration: %s", feature_key, exc)

    return registered_count
