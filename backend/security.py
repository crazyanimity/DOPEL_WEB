import os
from cryptography.fernet import Fernet
from slowapi import Limiter
from slowapi.util import get_remote_address

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY or ENCRYPTION_KEY == "CHANGE_ME_FERNET_KEY":
    raise RuntimeError("ENCRYPTION_KEY is not set (or still the placeholder). Set a real Fernet key in .env.")

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
