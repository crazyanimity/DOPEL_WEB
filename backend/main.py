"""
Dopel backend — auth + Postgres + Quick/Smart training + chat.

Run:
    cp .env.example .env      # then fill in real values, including GMAIL_ADDRESS / GMAIL_APP_PASSWORD
    pip install -r requirements.txt
    python -c "from database import Base, engine; import models; Base.metadata.create_all(engine)"
    uvicorn main:app --reload --port 8000
"""
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database import get_db, Base, engine
import models
from schemas import (
    SignupRequest, LoginRequest, AuthResponse, ChatRequest, TrainResponse, PersonaStatusResponse,
    SignupInitResponse, VerifyOtpRequest, ShareTokenResponse, ShareInfoResponse, PublicChatRequest,
)
from auth import (
    hash_password, verify_password, create_access_token, get_current_user,
    generate_otp, otp_expiry, send_otp_email,
)
from security import limiter
import ml_engine
import uuid

Base.metadata.create_all(bind=engine)  # safe no-op if tables already exist

app = FastAPI(title="Dopel API")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    # slowapi's default handler doesn't return a "detail" key, which made the frontend
    # fall back to a confusing generic message. This makes the real reason visible.
    return JSONResponse(
        status_code=429,
        content={"detail": f"Too many attempts — please wait a bit before trying again ({exc.detail})."},
    )

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth — signup is now two steps: request OTP, then verify it.
# ---------------------------------------------------------------------------
@app.post("/api/auth/signup", response_model=SignupInitResponse)
@limiter.limit("5/minute")
def signup(request: Request, body: SignupRequest, db: Session = Depends(get_db)):
    email = body.email.lower()

    existing = db.query(models.User).filter(models.User.email == email).first()
    if existing:
        raise HTTPException(409, "An account with this email already exists.")

    otp = generate_otp()
    pending = db.query(models.PendingSignup).filter(models.PendingSignup.email == email).first()
    if pending:
        pending.name = body.name.strip()
        pending.password_hash = hash_password(body.password)
        pending.otp = otp
        pending.expires_at = otp_expiry()
    else:
        pending = models.PendingSignup(
            email=email,
            name=body.name.strip(),
            password_hash=hash_password(body.password),
            otp=otp,
            expires_at=otp_expiry(),
        )
        db.add(pending)
    db.commit()

    send_otp_email(email, otp)
    return SignupInitResponse(message="Verification code sent. Check your email.")


@app.post("/api/auth/verify-otp", response_model=AuthResponse)
@limiter.limit("10/minute")
def verify_otp(request: Request, body: VerifyOtpRequest, db: Session = Depends(get_db)):
    email = body.email.lower()
    pending = db.query(models.PendingSignup).filter(models.PendingSignup.email == email).first()

    if not pending:
        raise HTTPException(400, "No pending signup for this email — sign up again.")

    expires_at = pending.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) > expires_at:
        db.delete(pending)
        db.commit()
        raise HTTPException(400, "That code expired. Sign up again for a new one.")

    if body.otp.strip() != pending.otp:
        raise HTTPException(400, "Incorrect code.")

    user = models.User(name=pending.name, email=email, password_hash=pending.password_hash)
    db.add(user)
    db.delete(pending)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "An account with this email already exists.")
    db.refresh(user)

    token = create_access_token(user.id)
    return AuthResponse(access_token=token, name=user.name, email=user.email)


@app.post("/api/auth/login", response_model=AuthResponse)
@limiter.limit("10/minute")
def login(request: Request, body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == body.email.lower()).first()
    # Same error for "no such user" and "wrong password" — don't leak which one it was.
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Incorrect email or password.")

    token = create_access_token(user.id)
    return AuthResponse(access_token=token, name=user.name, email=user.email)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _run_training(user_id: str, plan: str, files: list, your_name: str):
    """Runs in the background so the upload request returns immediately.
    files: list of (filename, raw_bytes) tuples."""
    from database import SessionLocal
    import json
    db = SessionLocal()
    try:
        persona = db.query(models.Persona).filter(
            models.Persona.user_id == user_id, models.Persona.plan == plan
        ).first()
        persona.status = "training"
        db.commit()

        combined_df, per_contact = ml_engine.parse_and_pair_uploads(files, your_name)
        if plan == "quick":
            ml_engine.train_quick(user_id, combined_df)
        else:
            ml_engine.train_smart(user_id, combined_df)

        persona.status = "ready"
        persona.pairs_trained = len(combined_df)
        persona.contacts = json.dumps(sorted(per_contact.keys()))
        persona.error_message = None
        db.commit()
    except HTTPException as e:
        persona.status = "error"
        persona.error_message = e.detail
        db.commit()
    except Exception as e:
        persona.status = "error"
        persona.error_message = "Training failed unexpectedly. Please try again."
        db.commit()
    finally:
        db.close()


@app.post("/api/train", response_model=PersonaStatusResponse)
@limiter.limit("30/hour")
async def train(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    plan: str = Form(...),
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    if plan not in ("quick", "smart"):
        raise HTTPException(400, "plan must be 'quick' or 'smart'")
    if len(files) < 1:
        raise HTTPException(400, "Upload at least one .txt chat export.")

    file_tuples = [(f.filename, await f.read()) for f in files]

    persona = db.query(models.Persona).filter(
        models.Persona.user_id == user.id, models.Persona.plan == plan
    ).first()
    if not persona:
        persona = models.Persona(user_id=user.id, plan=plan, status="pending")
        db.add(persona)
        db.commit()
        db.refresh(persona)

    persona.status = "training"
    db.commit()

    # your_name always comes from the authenticated account, never re-typed by the user.
    background_tasks.add_task(_run_training, user.id, plan, file_tuples, user.name)

    return PersonaStatusResponse(plan=plan, status="training", pairs_trained=0)


@app.get("/api/persona/{plan}/status", response_model=PersonaStatusResponse)
def persona_status(plan: str, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    import json
    persona = db.query(models.Persona).filter(
        models.Persona.user_id == user.id, models.Persona.plan == plan
    ).first()
    if not persona:
        raise HTTPException(404, "No training found for this plan yet.")
    return PersonaStatusResponse(
        plan=persona.plan, status=persona.status,
        pairs_trained=persona.pairs_trained,
        contacts=json.loads(persona.contacts) if persona.contacts else [],
        error_message=persona.error_message,
    )


# ---------------------------------------------------------------------------
# NEW: Share links — generate one per trained persona, then let anyone chat
# with it with no login. This is the "send it to a friend on WhatsApp" piece.
# ---------------------------------------------------------------------------
@app.post("/api/persona/{plan}/share", response_model=ShareTokenResponse)
def create_share_link(plan: str, db: Session = Depends(get_db), user: models.User = Depends(get_current_user)):
    persona = db.query(models.Persona).filter(
        models.Persona.user_id == user.id, models.Persona.plan == plan
    ).first()
    if not persona or persona.status != "ready":
        raise HTTPException(409, "Train this persona before sharing it.")

    if not persona.share_token:
        persona.share_token = uuid.uuid4().hex[:10]
        db.commit()

    return ShareTokenResponse(share_token=persona.share_token)


@app.get("/api/share/{token}", response_model=ShareInfoResponse)
def share_info(token: str, db: Session = Depends(get_db)):
    persona = db.query(models.Persona).filter(models.Persona.share_token == token).first()
    if not persona or persona.status != "ready":
        raise HTTPException(404, "This Dopel link isn't available.")
    owner = db.query(models.User).filter(models.User.id == persona.user_id).first()
    return ShareInfoResponse(name=owner.name, plan=persona.plan)


@app.post("/api/share/{token}/chat")
@limiter.limit("20/minute")
def share_chat(request: Request, token: str, body: PublicChatRequest, db: Session = Depends(get_db)):
    persona = db.query(models.Persona).filter(models.Persona.share_token == token).first()
    if not persona or persona.status != "ready":
        raise HTTPException(404, "This Dopel link isn't available.")
    owner = db.query(models.User).filter(models.User.id == persona.user_id).first()

    if ml_engine.is_asking_if_ai(body.message):
        return {"reply": ml_engine.disclosure_reply(owner.name)}

    if persona.plan == "quick":
        reply = ml_engine.reply_quick(owner.id, body.message)
    else:
        reply = ml_engine.reply_smart(owner.id, owner.name, body.message)
    return {"reply": reply}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@app.post("/api/chat")
@limiter.limit("30/minute")
def chat(
    request: Request,
    body: ChatRequest,
    db: Session = Depends(get_db),
    user: models.User = Depends(get_current_user),
):
    persona = db.query(models.Persona).filter(
        models.Persona.user_id == user.id, models.Persona.plan == body.plan
    ).first()
    if not persona or persona.status != "ready":
        raise HTTPException(409, "This persona isn't trained yet. Upload chat exports first.")

    # AI-disclosure guardrail — only responds this way when directly asked.
    if ml_engine.is_asking_if_ai(body.message):
        return {"reply": ml_engine.disclosure_reply(user.name)}

    if body.plan == "quick":
        reply = ml_engine.reply_quick(user.id, body.message)
    else:
        reply = ml_engine.reply_smart(user.id, user.name, body.message)

    return {"reply": reply}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ===== STARTUP: Read PORT from environment =====
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)