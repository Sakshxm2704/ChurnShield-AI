"""
backend/auth/password.py
-------------------------
Password hashing and verification using Werkzeug's scrypt implementation.
"""
from __future__ import annotations
from werkzeug.security import generate_password_hash, check_password_hash


def hash_password(plain: str) -> str:
    """Return a secure scrypt hash of *plain*."""
    return generate_password_hash(plain, method="scrypt")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    return check_password_hash(hashed, plain)
