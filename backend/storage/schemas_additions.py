# ============================================================================
# APPEND-ONLY additions to your existing schemas.py
# ============================================================================
from pydantic import BaseModel


class SignupInitResponse(BaseModel):
    message: str


class VerifyOtpRequest(BaseModel):
    email: str
    otp: str


class ShareTokenResponse(BaseModel):
    share_token: str


class ShareInfoResponse(BaseModel):
    name: str
    plan: str


class PublicChatRequest(BaseModel):
    message: str
