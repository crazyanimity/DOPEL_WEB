import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from database import Base


def gen_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)          # exact name used in their chat exports
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    personas = relationship("Persona", back_populates="user", cascade="all, delete-orphan")


class Persona(Base):
    """One row per (user, plan). Tracks training status and where the trained
    artifact lives on disk (never store the raw model weights/index in Postgres)."""
    __tablename__ = "personas"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    user_id = Column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False, index=True)
    plan = Column(String, nullable=False)          # "quick" or "smart"
    status = Column(String, nullable=False, default="pending")  # pending|training|ready|error
    pairs_trained = Column(Integer, default=0)
    artifact_path = Column(String, nullable=True)  # folder under storage/ holding the model files
    contacts = Column(Text, nullable=True)  # JSON list of contact names this persona was trained on
    error_message = Column(Text, nullable=True)
    share_token = Column(String, unique=True, nullable=True, index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                         onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="personas")
class PendingSignup(Base):
    __tablename__ = "pending_signups"

    email = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    password_hash = Column(String, nullable=False)
    otp = Column(String, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)