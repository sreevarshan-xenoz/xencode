#!/usr/bin/env python3
"""Pre-deployment secret validation for CI/CD pipelines."""

import argparse
import os
import sys


REQUIRED_ENV_VARS = [
    "DATABASE_URL",
    "REDIS_URL",
    "JWT_SECRET_KEY",
    "POSTGRES_USERNAME",
    "POSTGRES_PASSWORD",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate required deploy secrets.")
    parser.add_argument(
        "--environment",
        required=True,
        choices=["staging", "production"],
        help="Deployment environment name for logging.",
    )
    args = parser.parse_args()

    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        print(f"Missing required secrets for {args.environment}:")
        for name in missing:
            print(f"  - {name}")
        return 1

    database_url = os.getenv("DATABASE_URL", "")
    redis_url = os.getenv("REDIS_URL", "")
    jwt_secret = os.getenv("JWT_SECRET_KEY", "")

    if not database_url.startswith(("postgres://", "postgresql://")):
        print("DATABASE_URL must use a PostgreSQL scheme (postgres:// or postgresql://)")
        return 1

    if not redis_url.startswith("redis://"):
        print("REDIS_URL must use redis:// scheme")
        return 1

    if len(jwt_secret) < 16:
        print("JWT_SECRET_KEY must be at least 16 characters long")
        return 1

    print(f"Secret validation passed for {args.environment}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
