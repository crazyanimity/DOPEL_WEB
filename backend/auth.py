import os
import ssl
import random
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from sqlalchemy.orm import Session

from database import get_db
from models import User

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET or JWT_SECRET == "CHANGE_ME_TO_A_LONG_RANDOM_STRING":
    raise RuntimeError("JWT_SECRET is not set (or still the placeholder). Set a real random value in .env.")

JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))

bearer_scheme = HTTPBearer()


def hash_password(password: str) -> str:
    # bcrypt has a hard 72-byte input limit — truncate defensively so long
    # passwords don't raise instead of just being (harmlessly) capped.
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))


def create_access_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid session token.")
    return payload["sub"]


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency: verifies the JWT and returns the logged-in User row.
    Every protected endpoint depends on this — no endpoint trusts a user_id from the client body."""
    user_id = decode_access_token(credentials.credentials)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists.")
    return user


# ---------------------------------------------------------------------------
# OTP generation + email delivery for signup verification
# ---------------------------------------------------------------------------
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
OTP_EXPIRE_MINUTES = int(os.getenv("OTP_EXPIRE_MINUTES", "10"))


def generate_otp() -> str:
    return f"{random.randint(0, 999999):06d}"


def otp_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)


def send_otp_email(to_email: str, otp: str) -> None:
    """Sends the verification code via Gmail SMTP.

    Uses port 587 with STARTTLS rather than port 465 with implicit SSL.
    Several hosting platforms (Render, some Railway plans) block outbound
    port 465 as an anti-spam measure but leave 587 open, since it's the
    standard authenticated-submission port apps are expected to use.

    If GMAIL_* env vars aren't set, falls back to printing the code to the
    server console — lets you test the flow locally before wiring up SMTP.
    """
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print(f"[DEV MODE — no email sent] OTP for {to_email}: {otp}")
        return

    msg = MIMEText(f"Your Dopel verification code is {otp}. It expires in {OTP_EXPIRE_MINUTES} minutes.")
    msg["Subject"] = "Your Dopel verification code"
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = to_email

    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to_email, msg.as_string())