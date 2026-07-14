from pydantic import BaseModel, Field


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120,
                       description="Must match the sender name exactly as it appears in the WhatsApp export.")
    email: str
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    name: str
    email: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    plan: str = Field(..., pattern="^(quick|smart)$")
    contact: str | None = Field(None, description="Required for Smart Mode — which contact's style to reply in.")


class TrainResponse(BaseModel):
    plan: str
    status: str
    pairs_trained: int
    your_name: str


class PersonaStatusResponse(BaseModel):
    plan: str
    status: str
    pairs_trained: int
    contacts: list[str] = []
    error_message: str | None = None


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