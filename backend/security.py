import os
from cryptography.fernet import Fernet

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
except Exception:  # pragma: no cover - fallback for minimal deployments
    class Limiter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def limit(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    def get_remote_address(*args, **kwargs):
        return "local"

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY") or Fernet.generate_key().decode()

_fernet = Fernet(ENCRYPTION_KEY.encode())


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypts data before it touches disk (e.g. saved training pairs)."""
    return _fernet.encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    return _fernet.decrypt(token)


# Rate limiter — applied per-route in main.py.
# Keyed by client IP; protects login/signup from brute force and /chat from cost abuse
# (every /chat call on Smart Mode costs a Groq API call).
limiter = Limiter(key_func=get_remote_address)
